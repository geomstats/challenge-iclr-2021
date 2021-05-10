import os.path as osp
import sys
import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.io import read_tu_data
from torch_geometric.utils import remove_self_loops, to_undirected

class MyTUDataset(InMemoryDataset):
    r"""
    Adapted from
    https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/datasets/tu_dataset.py
    Skips download
    """

    def __init__(
        self,
        root,
        name,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        use_node_attr=False,
        use_edge_attr=False,
        cleaned=False,
        apply_ORC=True,
    ):
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

    @property
    def raw_dir(self):
        name = "raw{}".format("_cleaned" if self.cleaned else "")
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = "processed{}".format("_cleaned" if self.cleaned else "")
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self):
        names = ["A", "graph_indicator"]
        return ["{}_{}.txt".format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        """
        Skip downloads
        """

        pass

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return "{}({})".format(self.name, len(self))