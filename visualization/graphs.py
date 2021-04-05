import torch
import numpy as np
from captum.attr import Saliency

from pymatgen import Element
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx


def draw_molecule(g, edge_mask=None, draw_edge_labels=False):
    g = g.copy().to_undirected()
    node_labels = {}
    for u, data in g.nodes(data=True):
        node_labels[u] = data['name']
    pos = nx.spring_layout(g)
    if edge_mask is None:
        edge_color = 'black'
        widths = None
    else:
        edge_color = [edge_mask[(u, v)] for u, v in g.edges()]
        widths = [x * 10 for x in edge_color]
    nx.draw(g, pos=pos, labels=node_labels, width=widths,
            edge_color=edge_color, edge_cmap=plt.cm.Blues,
            node_color='azure')

    if draw_edge_labels and edge_mask is not None:
        edge_labels = {k: ('%.2f' % v) for k, v in edge_mask.items()}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                     font_color='red')
    plt.show()



def to_molecule_graph(data):
    g = to_networkx(data, node_attrs=['x'])
    for u, data in g.nodes(data=True):
        data['name'] = Element.from_Z(data['x'].index(1.0) + 1).symbol
        del data['x']
    return g


def explain(model, data, device, target=0):
    input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(device)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    def model_forward(edge_mask, data):
        batch = torch.zeros(data.x.shape[0], dtype=int).to(device)
        out = model(x, edge_index, batch, edge_mask)
        return out

    saliency = Saliency(model_forward)
    mask = saliency.attribute(input_mask, target=target,
                              additional_forward_args=(data,))
    edge_mask = np.abs(mask.cpu().detach().numpy())
    if edge_mask.max() > 0:  # avoid division by zero
        edge_mask = edge_mask / edge_mask.max()
    return edge_mask


# from collections import defaultdict
    #
    # def aggregate_edge_directions(edge_mask, data):
    #     edge_mask_dict = defaultdict(float)
    #     for val, u, v in list(zip(edge_mask, *data.edge_index)):
    #         u, v = u.item(), v.item()
    #         if u > v:
    #             u, v = v, u
    #         edge_mask_dict[(u, v)] += val
    #     return edge_mask_dict
    #
    # for data in test_loader:
    #     mol = to_molecule_graph(data)
    #
    #     for title, method in [('Saliency', 'saliency')]:
    #         edge_mask = explain(model, data, device, target=0)
    #         edge_mask_dict = aggregate_edge_directions(edge_mask, data)
    #         plt.figure(figsize=(10, 5))
    #         plt.title(title)
    #         draw_molecule(mol, edge_mask_dict)
    #     break