from typing import List, Tuple, Dict, Any
import torch
from torch import nn


class Device(object):
    """Node
    """

    def __init__(self, raw_feature: torch.Tensor, hidden_dim: int, label: int, label_one_hot: torch.Tensor) -> None:
        self.client: Client = None
        self.devices_set_name: str = ''
        self._neighbors: List[Device] = []
        self._raw_feature: torch.Tensor = raw_feature.requires_grad_(False)
        # self._embedding: nn.Module = nn.Parameter(torch.Tensor(hidden_dim), requires_grad=True)
        # nn.init.normal_(self._embedding)
        self._label: int = label
        self._label_one_hot: torch.Tensor = label_one_hot.requires_grad_(False)

    def to(self, torch_device: torch.device) -> None:
        self._raw_feature = self._raw_feature.to(torch_device)
        # self._embedding = self._embedding.to(torch_device)
        self._label_one_hot = self._label_one_hot.to(torch_device)

    def add_edge(self, neighbor: Any) -> None:
        assert neighbor.__class__ is Device
        if neighbor not in self._neighbors:
            self._neighbors.append(neighbor)

    def init_layer_embeddings(self) -> None:
        # layer_embedding_1 = torch.concat([self.client.local_models['encoder_raw'](self._raw_feature), self._embedding], dim=0)
        # layer_embedding_1 = self.client.local_models['encoder_fea'](layer_embedding_1)
        # self._layer_embeddings = [layer_embedding_1]
        self._layer_embeddings = [self._raw_feature]

    def init_neighbor_embeddings(self) -> None:
        self._intra_neighbor_embeddings = [self._layer_embeddings[-1]]
        self._cross_neighbor_embeddings = []

    def message_passing(self, set_name: str, layer: int) -> None:
        for neighbor in self._neighbors:
            if neighbor.devices_set_name == set_name:
                self_embedding = self._layer_embeddings[layer - 1] / ((1 + len(self._neighbors)) ** 0.5)  # GCN
                # self_embedding = self._layer_embeddings[layer - 1]  # GraphSage
                neighbor.receive(self_embedding, intra=True if neighbor.client == self.client else False)

    def receive(self, neighbor_embedding: torch.Tensor, intra: bool) -> None:
        if intra:
            self._intra_neighbor_embeddings.append(neighbor_embedding)
        else:
            self._cross_neighbor_embeddings.append(neighbor_embedding)

    def aggregate_and_update(self, layer: int) -> None:  # TODO secure aggregating
        # aggregate
        neighbor_embeddings = self._intra_neighbor_embeddings + self._cross_neighbor_embeddings
        aggregated_embedding = self.client.local_models[f'aggregator_{layer}'](neighbor_embeddings)
        # update
        updated_embedding = aggregated_embedding
        self._layer_embeddings.append(updated_embedding)

    def classify(self) -> torch.Tensor:
        # pred: torch.Tensor = self.client.local_models['classifier'](self._layer_embeddings[-1])
        self.pred: torch.Tensor = self._layer_embeddings[-1].clone()
        # pred: torch.Tensor = self.client.local_models['MLP'](self._raw_feature)
        loss: torch.Tensor = self.client.local_models['loss_function'](self.pred.unsqueeze(0), self._label_one_hot.unsqueeze(0))
        _, pred_label = self.pred.max(dim=-1)
        return loss, int(pred_label), int(self._label)


class Client(object):
    """Subgraph
    """

    def __init__(self, local_models: nn.ModuleDict, num_classes: int, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler) -> None:
        self._devices: List[Device] = []
        self.local_models: nn.ModuleDict = local_models
        self.num_classes: int = num_classes
        self.optimizer: torch.optim.Optimizer = optimizer
        self.scheduler: torch.optim.lr_scheduler._LRScheduler = scheduler

    def to(self, torch_device: torch.device) -> None:
        self.local_models = self.local_models.to(torch_device)
        for device in self._devices:
            device.to(torch_device)

    def add_device(self, device: Device) -> None:
        self._devices.append(device)
        device.client = self

    def split_devices_set(self, train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor) -> None:
        self._devices_sets_dict: Dict[str, List[Device]] = {
            'train': [device for i, device in enumerate(self._devices) if train_mask[i]],
            'val': [device for i, device in enumerate(self._devices) if val_mask[i]],
            'test': [device for i, device in enumerate(self._devices) if test_mask[i]],
        }
        for set_name, devices in self._devices_sets_dict.items():
            for device in devices:
                device.devices_set_name = set_name

    def init_layer_embeddings(self, set_name: str) -> None:
        for device in self._devices_sets_dict[set_name]:
            device.init_layer_embeddings()

    def init_neighbor_embeddings(self, set_name: str) -> None:
        for device in self._devices_sets_dict[set_name]:
            device.init_neighbor_embeddings()

    def update_local_models(self, global_models: nn.ModuleDict) -> None:
        for module in self.local_models:
            self.local_models[module].load_state_dict(global_models[module].state_dict())

    def local_message_passing(self, set_name: str, layer: int) -> None:
        for device in self._devices_sets_dict[set_name]:
            device.message_passing(set_name, layer)

    def local_aggregate_and_update(self, set_name: str, layer: int) -> None:
        for device in self._devices_sets_dict[set_name]:
            device.aggregate_and_update(layer)

    def local_clf_loss(self, set_name: str = 'train') -> torch.Tensor:
        result = [device.classify() for device in self._devices_sets_dict[set_name]]
        losses = 0.0
        for l in [l for l, _, _ in result]:
            losses = losses + l
        return torch.true_divide(losses, len(result))
    
    def local_cross_loss(self, set_name: str = 'train') -> torch.Tensor:
        cross_loss = 0.0
        for device in self._devices_sets_dict[set_name]:
            for neighbor in device._neighbors:
                if neighbor.devices_set_name == set_name and not neighbor.client == self:
                    cross_loss += torch.mean(device._layer_embeddings[-2] - neighbor._layer_embeddings[-2])
        return cross_loss

    def local_train(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.scheduler.step()

    def local_validate(self) -> None:
        result = [device.classify() for device in self._devices_sets_dict['test']]
        pred = [p for _, p, t in result]
        true = [t for _, p, t in result]
        return pred, true
