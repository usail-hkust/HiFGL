from typing import *
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from copy import deepcopy

from modules import *
from models import *


class TrainHiFGL(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # initialization
        self.clients_data = self._get_clients()
        num_classes = self._count_classes(self.clients_data)
        feature_dim = self.clients_data[-1].x.size(1)
        self.client_dict: Dict[int, Client] = {}
        self.device_dict: Dict[int, Client] = {}
        self.global_models = get_module_dict(self.hparams.num_layers, feature_dim, self.hparams.hidden_dim, num_classes, self.hparams.dropout)

        # construct hierarchical federated graph
        self._construct_graph(num_classes)

        # log
        # self.acc = {'1': [], '2': [], '3': [], '4': [], '5': []}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.global_models.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader([0], batch_size=1)

    def val_dataloader(self):
        return torch.utils.data.DataLoader([0], batch_size=1)

    def test_dataloader(self):
        return torch.utils.data.DataLoader([0], batch_size=1)

    def on_fit_start(self):
        self.global_models = self.global_models.to(self.device)
        for client in self.client_dict.values():
            client.to(self.device)

    def training_step(self, *args, **kwargs):

        # local
        # self._global_message_passing(set_name='train')
        # client_clf_loss = {}
        # for client in self.client_dict.values():
        #     client_clf_loss[client] = client.local_clf_loss(set_name='train')
        # # client_cross_loss = {}
        # # for client in self.client_dict.values():
        # #     client_cross_loss[client] = client.local_cross_loss(set_name='train')
        # for client in self.client_dict.values():
        #     loss = client_clf_loss[client]  # + client_cross_loss[client]
        #     client.local_train(loss)
        # return

        # global
        overall_loss = 0.0
        self._global_message_passing(set_name='train')
        for client in self.client_dict.values():
            loss = client.local_clf_loss(set_name='train')
            overall_loss = overall_loss + loss
        # for client in self.client_dict.values():
        #     loss = client.local_cross_loss(set_name='train')
        #     overall_loss = overall_loss + 1e-3 * loss
        overall_loss = torch.true_divide(overall_loss, len(self.client_dict))

        self.log('train_loss', overall_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return overall_loss

    def training_epoch_end(self, outputs) -> None:
        for client in self.client_dict.values():
            client.update_local_models(self.global_models)
        return

    def validation_step(self, *args, **kwargs):
        self._global_message_passing(set_name='test')
        overall_pred = []
        overall_true = []
        for i, client in enumerate(self.client_dict.values()):
            pred, true = client.local_validate()
            acc = accuracy(torch.tensor(pred), torch.tensor(true))
            # self.acc[str(i + 1)].append(acc.item())
            overall_pred.extend(pred)
            overall_true.extend(true)
        overall_acc = accuracy(torch.tensor(overall_pred), torch.tensor(overall_true))
        # print(f'Overall Accuracy={overall_acc}')
        self.log('ACC', overall_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, *args, **kwargs):
        self.validation_step(*args, **kwargs)

    def _get_clients(self):
        return {i: torch.load(f"./data/{self.hparams.dataset_name}/{i}_clients.pt") for i in ([-1] + list(range(1, self.hparams.num_clients + 1)))}

    def _count_classes(self, clients: dict):
        classes = set()
        client_classes = [set(client.y.tolist()) for client in clients.values()]
        for i in client_classes:
            classes = classes | i
        return len(classes)

    def _construct_graph(self, num_classes: int) -> None:

        # intra client
        for client_id in range(1, self.hparams.num_clients + 1):

            # client: subgraph
            # local_models = deepcopy(self.global_models)
            local_models = self.global_models
            optimizer = torch.optim.Adam(local_models.parameters(), lr=self.hparams.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
            client = Client(
                local_models=local_models,
                num_classes=num_classes,
                optimizer=optimizer,
                scheduler=scheduler
            )
            self.client_dict[client_id] = client
            # client data
            X: torch.Tensor = self.clients_data[client_id].x
            Y: torch.Tensor = self.clients_data[client_id].y
            Y_one_hot: torch.Tensor = F.one_hot(Y, num_classes=num_classes).float()
            index_orig: torch.Tensor = self.clients_data[client_id].index_orig.tolist()
            num_nodes, feature_dim = X.size()

            # device: node
            for i in range(num_nodes):
                device = Device(
                    raw_feature=X[i],
                    hidden_dim=self.hparams.hidden_dim,
                    label=Y[i],
                    label_one_hot=Y_one_hot[i],
                )
                client.add_device(device)
                self.device_dict[index_orig[i]] = device

            # edges
            src_indices, tgt_indices = self.clients_data[client_id].edge_index
            for src_index, tgt_index in zip(src_indices.tolist(), tgt_indices.tolist()):
                src_device: Device = self.device_dict[index_orig[src_index]]
                tgt_device: Device = self.device_dict[index_orig[tgt_index]]
                src_device.add_edge(tgt_device)
                tgt_device.add_edge(src_device)

            # split dataset
            client.split_devices_set(
                train_mask=self.clients_data[client_id].train_mask,
                val_mask=self.clients_data[client_id].val_mask,
                test_mask=self.clients_data[client_id].test_mask,
            )

        # cross client
        if self.hparams.cross_client:
            src_indices, tgt_indices = self.clients_data[-1].edge_index
            index_orig: torch.Tensor = self.clients_data[-1].index_orig.tolist()
            for src_index, tgt_index in zip(src_indices.tolist(), tgt_indices.tolist()):
                src_device: Device = self.device_dict[index_orig[src_index]]
                tgt_device: Device = self.device_dict[index_orig[tgt_index]]
                src_device.add_edge(tgt_device)
                tgt_device.add_edge(src_device)

    def _global_message_passing(self, set_name: str) -> None:
        self.client_local_models: List[nn.ModuleDict] = []
        for client in self.client_dict.values():
            client.init_layer_embeddings(set_name)
        for layer in range(self.hparams.num_layers):
            for client in self.client_dict.values():
                client.init_neighbor_embeddings(set_name)
            for client in self.client_dict.values():
                client.local_message_passing(set_name, layer+1)
            for client in self.client_dict.values():
                client.local_aggregate_and_update(set_name, layer+1)
