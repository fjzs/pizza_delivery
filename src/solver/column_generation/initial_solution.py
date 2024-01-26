from typing import Dict, List

import numpy as np

from problem.cvrp import CVRP
from problem.route import Route
from problem.solution import Solution


def get_initial_solution(data: CVRP) -> Solution:
    """This heuristic finds the closest client iteratively.
    It provides at most 1 route for each vehicle in the dictionary.
    It may not provide a route for every vehicle.

    Args:
        data (CVRP): instance

    Returns:
        Solution:
    """
    open_clients = list(range(1, data.N + 1))  # from 1,..., N
    vehicles_ids = list(range(1, data.K + 1))  # from 1,..., K
    routes: Dict[int, Route] = dict()

    for k in vehicles_ids:
        nodes = []

        # Get a route for this vehicle
        if len(open_clients) > 0:
            nodes = [0]  # start at the depot
            capacity = data.Q

            # Look iterateviely for next closest client that fits in the vehicle
            while capacity > 0 and len(open_clients) > 0:
                next_client = _get_closest_open_client(
                    origin=nodes[-1],
                    open_clients=open_clients,
                    distances=data.distance,
                    capacity_left=capacity,
                    demand=data.demand,
                )
                nodes.append(next_client)
                capacity -= data.demand[next_client]
                open_clients.remove(next_client)

            # there is no more capacity or clients left, close the route
            nodes.append(0)
            route = Route(nodes)
            # Add the route to the solution
            routes[k] = route

    return Solution(data, routes)


def _get_closest_open_client(
    origin: int,
    open_clients: List[int],
    distances: np.ndarray,
    capacity_left: int,
    demand: np.ndarray,
) -> int:
    """Given a list of open clients, an origin and a distance
    matrix, it retrieves the closest client which the demand fits in the vehicle

    Args:
        origin (int):
        open_clients (List[int]):
        distances (np.ndarray):
        capacity_left (int):
        demand (np.ndarray): client demand

    Returns:
        closest_client (int):
    """
    subset = [i for i in open_clients if (i != origin) and (capacity_left >= demand[i])]
    min_index = np.argmin(distances[origin, subset])
    closest_client = subset[min_index]
    return closest_client
