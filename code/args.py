from typing import *
import argparse
import pytorch_lightning as pl
import yaml
import os


def parse_global_args():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument('--root', type=str, default='./')
    parser.add_argument('--data_folder', type=str, default='data')
    parser.add_argument('--args_folder', type=str, default='args')
    parser.add_argument('-D', '--dataset_name', type=str, default='cora')
    parser.add_argument('-E', '--epochs', type=int, default=1024, help='epoch size')
    parser.add_argument('--train_batch_size', type=int, default=1, help='train batch size.')
    parser.add_argument('--val_batch_size', type=int, default=1, help='validation batch size.')
    parser.add_argument('--test_batch_size', type=int, default=1, help='test batch size.')
    parser.add_argument('--monitor', type=str, default='ACC', help='training monitor')
    parser.add_argument('--cross_client', type=bool, default=True, help='whether the graph remains the cross-client edges')
    parser.add_argument('--num_layers', type=int, default=2, help='number of gnn layers')

    global_args, _ = parser.parse_known_args()
    return parser, global_args


def parse_dataset_specific_args(parent_args: argparse.Namespace):
    with open(os.path.join(parent_args.root, parent_args.args_folder, f'{parent_args.dataset_name}.yaml'), 'r') as f:
        dataset_specific_args = yaml.load(f, Loader=yaml.FullLoader)
    parent_args.__dict__.update(**dataset_specific_args)
    return parent_args
