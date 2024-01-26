import os
import random

import matplotlib.pyplot as plt
import numpy as np

from utils import utils


def create(
    num_clients: int,
    num_vehicles: int,
    capacity: int,
    name: str,
    folder: str,
    radius: int,
):
    """Creates a CVRP instance and saves it.

    Args:
        num_clients (int):
        num_vehicles (int):
        capacity (int):
        name (str):
        folder (str):
        radius (int): value to create a square with the origin in the middle
    """

    assert num_clients >= 1
    assert num_vehicles >= 1
    assert capacity >= 1

    instance = dict()
    instance["K"] = num_vehicles
    instance["N"] = num_clients
    instance["Q"] = capacity

    # Compute all the points in a cartesian grid and then pick
    # some of them randomly
    x = list(range(-radius, radius + 1, 1))
    y = list(range(-radius, radius + 1, 1))
    pos = []
    for i in x:
        for j in y:
            if abs(i) + abs(j) != 0:  # avoid the origin
                pos.append((i, j))
    random.shuffle(pos)

    # xy is a np.ndarray of shape n+1 rows and 2 columns
    xy = np.zeros((num_clients + 1, 2))
    # The first row is the depot at the origin
    for i in range(num_clients):
        xy[i + 1][0] = int(pos[i][0])  # x
        xy[i + 1][1] = int(pos[i][1])  # y

    # Add it to the instance
    instance["coordinates"] = xy.tolist()

    # Create the demands
    demands = [1] * (num_clients + 1)
    demands[0] = 0  # the demand of the depot is 0
    instance["demand"] = demands

    # Folder of this instance
    folder_instance = os.path.join(folder, name)
    if not os.path.exists(folder_instance):
        os.makedirs(folder_instance)

    # Plot it and save the figure to inspect it
    plt.scatter(xy[:, 0], xy[:, 1])
    plt.savefig(
        os.path.join(folder_instance, "map.png"),
        bbox_inches="tight",
        pad_inches=0.1,
    )

    # Save the instance file
    utils.save_dictionary(data=instance, file_name=name, folder=folder_instance)


def load(filepath: str) -> dict:
    """Loads an instance and returns the data of it

    Args:
        filepath (str):

    Returns:
        (dict):
    """
    return utils.load_file(filepath)
