import matplotlib.pyplot as plt
import numpy as np

from utils import utils

_RADIUS = 20  # radius from the depot (0,0) in 4 directions
_DECIMALS = 0
_FOLDER_INSTANCES = "instances"


def create(
    num_clients: int,
    num_vehicles: int,
    capacity: int,
    name: str,
    decimals=_DECIMALS,
    radius=_RADIUS,
    folder=_FOLDER_INSTANCES,
):
    """Creates a CVRP instance and saves it. It starts to rotate around origin (0,0)
    with given radius and angle to create the required number of clients

    Args:
        num_clients (int): number of clients without considering the depot
        num_vehicles (int):
        capacity (int):
    """

    assert num_clients >= 1
    assert num_vehicles >= 1
    assert capacity >= 1

    instance = dict()
    instance["K"] = num_vehicles
    instance["N"] = num_clients
    instance["Q"] = capacity

    # Fill the coordinates randomly
    random_u = np.random.rand(num_clients + 1, 2)
    xy = np.ones((num_clients + 1, 2)) * -radius
    xy = xy + 2 * radius * random_u

    # The first row is the depot at the origin
    xy[0, :] = 0

    # Round the positions
    xy = np.round(xy, decimals=decimals)
    print(xy)

    # Add it to the instance
    instance["coordinates"] = xy.tolist()

    # Create the demands
    demands = [1] * (num_clients + 1)
    demands[0] = 0  # the demand of the depot is 0
    instance["demand"] = demands

    # Plot it to check if its fine
    plt.scatter(xy[:, 0], xy[:, 1])
    plt.show()

    # Save the instance
    utils.save_dictionary(instance, name, folder)


def load(instance_name: str) -> dict:
    """Loads an instance and returns the data of it

    Args:
        instance_name (str):

    Returns:
        (dict):
    """
    return utils.load_file(instance_name, folder=_FOLDER_INSTANCES)
