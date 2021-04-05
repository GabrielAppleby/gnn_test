import numpy as np
import pathlib

import torch
from pymatgen import Structure
from sklearn.datasets import load_iris
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm


class Iris(InMemoryDataset):
    """

    """

    def __init__(self):
        """
        I can host the raw url until Tufts yells at me. Much faster than redownloading from
        materials project..
        """
        self.dataset_name: str = 'iris'
        self.raw_data_npz: str = 'iris.npz'
        self.processed_data_pt: str = 'iris.pt'
        self.data_folder_path: pathlib.Path = pathlib.Path(pathlib.Path(__file__).parent,
                                                           self.dataset_name)
        super().__init__(root=self.data_folder_path)
        self.raw_data_path: pathlib.Path = pathlib.Path(self.data_folder_path, self.raw_data_npz)
        self.processed_data_path: pathlib.Path = pathlib.Path(self.data_folder_path,
                                                              self.processed_data_pt)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """
        Raw file names that need to exist to prevent redownload.
        :return: The list of file names.
        """
        return [self.raw_data_npz]

    @property
    def processed_file_names(self):
        """
        Processed file names that need to exist to prevent reprocess.
        :return: The list of file names.
        """
        return [self.processed_data_pt]

    def download(self):
        """
        Download the files from the raw_url.
        :return: None
        """
        raise NotImplementedError

    def process(self):
        """
        Process the files. Currently used the crystalnn approach to generating edges, this seems
        to take a fill 12 hours or so to run on a decent computer. If we need to do this often it
        shouldn't be too much work to parallelize it.

        Constructing adjacency matrix:

            General idea is to run through the all of the cifs.
                Get the neighbors of each 'site' of the crystal
                Then run through each site of the crystal
                    Add an edge between that site and each of its neighbors if that edge does not
                    already exist.

        Construct node features:

            Currently, this is actually just a one hot encoding of each atom stacked together.
            In the past I have also tried using the atom features given from the mt-cgnn paper, but
            did not notice a real difference in performance. Need to revisit this.

        Constructing Targets:
            Taken as given in the mt-cgnn paper.


        :return: None. Saves the processed files.
        """
        data = np.load(str(self.raw_data_path))
        features = torch.tensor(data['features'])
        targets = data['targets']

        for i in range(targets.shape[0]):
            data = Data(x=torch.tensor(features[i], dtype=torch.float),
                        y=torch.tensor(targets[i], dtype=torch.float),
                        edge_weight=torch.tensor(edge_weights, dtype=torch.float),
                        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous())



        raw_atom_embedding = self._get_raw_atom_embedding()
        target_vars_by_id = self._get_target_variables()
        data_list = []
        with ZipFile(pathlib.Path(self.raw_dir, 'cifs.zip'), 'r') as zip_file:
            for file_name in tqdm(zip_file.namelist()):
                nodes_embedding = []
                nodes_species_number = []
                nodes_positions = []
                edges = []
                edge_weights = []
                with zip_file.open(file_name, 'r') as file:
                    crystal = Structure.from_str(file.read().decode("utf-8"), fmt='cif')
                    neighbors_by_atom = self._get_neighbors_crystalnn(crystal)
                    cid = file_name.split('.')[0]
                    for idx, site in enumerate(crystal.sites):
                        nodes_embedding.append(raw_atom_embedding[site.specie.number])
                        nodes_species_number.append(site.specie.number)
                        nodes_positions.append(site.coords)
                        for neighbor in neighbors_by_atom[idx]:
                            edge = [idx, neighbor[0]]
                            if edge not in edges:
                                edges.append(edge)
                                edge_weights.append(neighbor[1])
                    data = Data(x=torch.tensor(nodes_embedding, dtype=torch.float),
                                y=torch.tensor([target_vars_by_id[cid]], dtype=torch.float),
                                z=torch.tensor(nodes_species_number, dtype=torch.long),
                                pos=torch.tensor(nodes_positions, dtype=torch.float),
                                edge_weight=torch.tensor(edge_weights, dtype=torch.float),
                                edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous())
                    data.id = cid
                    data_list.append(data)
        data, slice = self.collate(data_list)
        torch.save((data, slice), self.processed_paths[0])

    def pairwise_distance(self, features: torch.Tensor):
        squared_norm = features.multiply(features).sum(1, keepdim=True)
        two_x_squared = features.matmul(features.t()).multiply(2.0)
        return squared_norm.subtract(two_x_squared).add(squared_norm.t())


