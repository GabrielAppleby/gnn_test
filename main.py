import os
import pathlib
from functools import partial
from typing import Tuple

import torch
import numpy as np
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch_geometric.data import DataLoader, Dataset

from data.utils import MetaDataset, load_data
from modeling.models import BasicGCN

CURRENT_DIR = pathlib.Path(__file__).parent
SAVED_MODEL_PATH = pathlib.Path(CURRENT_DIR, 'basic_gcn.pt')
SAVED_EMBEDDINGS_PATH = pathlib.Path(CURRENT_DIR, 'gcn_embeddings.npz')

RANDOM_SEED: int = 42


def set_random_seeds(random_seed: int) -> None:
    """
    Set the random seed for any libraries used.
    :param random_seed: The random seed to set.
    :return: None.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


def train(config,
          checkpoint_dir,
          num_epochs: int = 60) -> torch.nn.Module:
    """
    Train the model for num_epochs epochs on the given device using the given data.

    Right now a lot of stuff is hardcoded for this specific model / dataset.
    Most importantly only the first column of the y target matrix is used.

    :param model: The model to train.
    :param training_data: The training data to use.
    :param device: The device to train on.
    :param num_epochs: The number of epochs to train for.
    :return: The trained model.
    """
    dataset: MetaDataset = load_data()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BasicGCN(dataset.num_node_features,
                     atom_embeddings=config['atom_embedding'],
                     num_convs=config['num_convs'],
                     conv_hidden=config['conv_hidden'],
                     num_linear=config['num_linear'],
                     num_outputs=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_loader = DataLoader(dataset.train, batch_size=32, shuffle=True)
    loss_func = torch.nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for data in train_loader:
            z = data.z.to(device)
            edge_index = data.edge_index.to(device)
            batch = data.batch.to(device)

            # This dataset is for multitask learning, but
            # lets just stick to formation energy for now.
            targets = data.y#[:, 0]
            targets = torch.reshape(targets, (-1, 1)).to(device)

            preds = model(z, edge_index, batch)
            loss = loss_func(preds, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / len(train_loader)
        print('{epoch} loss: {loss}'.format(epoch=epoch, loss=epoch_loss))

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=epoch_loss)


def create_embeddings(model: torch.nn.Module,
                      dataset: Dataset,
                      device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates an embedding from data by forward pass through a trained model that returns an embedding
    as its second output.
    :param model: The trained model that ouputs embeddings.
    :param dataset: The dataset to embed.
    :param device: The device to do the forward pass on.
    :return: The embeddings and their corresponding cids.
    """
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    all_embeddings = []
    all_cids = []
    for data in data_loader:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)

        _, embeddings = model(x, edge_index, batch)
        all_embeddings.append(embeddings.detach().cpu().numpy())
        all_cids.append([int(x) for x in data.id])
    return np.concatenate(all_embeddings), np.concatenate(all_cids)


def main():
    # set_random_seeds(RANDOM_SEED)
    config = {
        "atom_embedding": tune.grid_search([128, 256]),
        "num_convs": tune.grid_search([1, 2, 3, 4]),
        "conv_hidden": tune.grid_search([32, 64, 128]),
        "num_linear": tune.grid_search([4, 3, 2, 1]),
        "linear_hidden": tune.grid_search([32, 64, 128, 256]),
    }
    reporter = CLIReporter(
        metric_columns=["loss", "training_iteration"])
    result = tune.run(
        partial(train, checkpoint_dir='checkpoint/'),
        config=config,
        resources_per_trial={'gpu': 1},
        num_samples=2,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    # torch.save(model, SAVED_MODEL_PATH)

    # model: BasicGCN = torch.load(SAVED_MODEL_PATH)
    # embeddings, cids = create_embeddings(model, dataset.train, device)
    # np.savez(SAVED_EMBEDDINGS_PATH, embedding=embeddings, cids=cids)


if __name__ == '__main__':
    main()
