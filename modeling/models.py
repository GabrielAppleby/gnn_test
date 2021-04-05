import torch
import torch.nn.functional as F
from torch.nn import Linear, Embedding, ModuleList
from torch_geometric.nn import GCNConv, global_mean_pool, Set2Set


class BasicGCN(torch.nn.Module):
    """
    A basic GCN.

    Consists of:

    Some GCNConv layers (see https://tkipf.github.io/graph-convolutional-networks/ for
    a good explanation.

    A global mean pooling layer.

    And then some linear layers.

    """
    def __init__(self,
                 num_node_features,
                 atom_embeddings,
                 num_convs,
                 conv_hidden,
                 num_linear,
                 num_outputs):
        """
        Construct the Basic GCN.
        :param num_node_features: The number of node features.
        :param num_outputs: The number of outputs.
        :param num_hidden_neurons: The number of neurons to use in all the intermediate layers. This
        is very simplistic but works fine for now.
        """
        super(BasicGCN, self).__init__()
        self.embedding = Embedding(118, atom_embeddings)
        self.convs = ModuleList([])
        size = atom_embeddings
        for conv in range(num_convs):
            self.convs.append(GCNConv(size, conv_hidden))
            size = conv_hidden
        self.pooling = Set2Set(conv_hidden, 2)
        self.lins = ModuleList([])
        lin_neurons = conv_hidden * 2
        for lin in range(num_linear-1):
            self.lins.append(Linear(lin_neurons, lin_neurons // 2))
            lin_neurons = lin_neurons // 2
        self.lin3 = Linear(lin_neurons, num_outputs)

    def forward(self, x, edge_index, batch, edge_weight=None):
        """
        Perform the forward pass through all of the layers. Relu is used after each layer, and
        global mean pooling is used between conv and linear layers. Finally some dropout is applied
        in between the final hidden layer and the output layer.
        :param x: The batch of node features.
        :param edge_index: The batch of edge indices
        :param batch: The tensor describing which nodes belong to which graph within the batch.
        :param edge_weight: Edge weightings, not really used at the moment.
        :return: The outputs, as well as the embeddings (last hidden layer).
        """
        x = self.embedding(x)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
        x = self.pooling(x, batch)
        for lin in self.lins:
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(lin(x))
        x = self.lin3(x)
        return x
