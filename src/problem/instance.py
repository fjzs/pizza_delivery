import os
import random

import matplotlib.pyplot as plt
import numpy as np

from utils import utils


def create(
    num_clients: int,
    capacity: int,
    name: str,
    folder: str,
    radius: int,
):
    """Creates a CVRP instance and saves it.

    Args:
        num_clients (int):
        capacity (int):
        name (str):
        folder (str):
        radius (int): value to create a square with the origin in the middle
    """

    assert num_clients >= 1
    assert capacity >= 1

    instance = dict()
    instance["N"] = num_clients + 1
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

    save_instance(instance=instance, file_name=name, folder=folder)


def save_instance(instance: dict, file_name: str, folder: str):
    utils.save_dictionary(data=instance, file_name=file_name, folder=folder)


def load(filepath: str) -> dict:
    """Loads an instance and returns the data of it

    Args:
        filepath (str):

    Returns:
        (dict):
    """
    return utils.load_file(filepath)


def generate_from_vrp_file(folder: str, instance_name: str):
    filepath = os.path.join(folder, instance_name, instance_name + ".vrp")
    N = None
    q = None
    node_coords_section_active = False
    demand_section_active = False
    xy = None
    demands = None

    with open(filepath, "r") as file:
        for line in file:
            if line.startswith("DIMENSION"):
                N = int(line.split(":")[1].strip())
                xy = np.zeros((N, 2))
                demands = [0] * N
            elif line.startswith("CAPACITY"):
                q = int(line.split(":")[1].strip())
            elif line.startswith("NODE_COORD_SECTION"):
                node_coords_section_active = True
            elif line.startswith("DEMAND_SECTION"):
                demand_section_active = True
                node_coords_section_active = False
            elif line.startswith("DEPOT_SECTION"):
                break
            elif node_coords_section_active:
                id, x, y = [float(p) for p in line.split()]
                xy[int(id) - 1] = [int(x), int(y)]
            elif demand_section_active:
                id, d_i = [int(p) for p in line.split()]
                demands[id - 1] = d_i

    instance = dict()
    instance["N"] = N
    instance["Q"] = q
    instance["coordinates"] = xy.tolist()
    instance["demand"] = demands

    # Plot it and save the figure to inspect it
    min_size = 5
    size_per_demand = 3
    plt.scatter(xy[0, 0], xy[0, 1], c="blue", s=30)  # depot
    plt.scatter(
        xy[1:, 0],
        xy[1:, 1],
        c="red",
        s=[(min_size + d * size_per_demand) for d in demands[1:]],
    )  # customers
    plt.savefig(
        os.path.join(folder, instance_name, "map.png"),
        bbox_inches="tight",
        pad_inches=0.1,
    )
    folder_instance = os.path.join(folder, instance_name)
    save_instance(instance=instance, file_name=instance_name, folder=folder_instance)
