from typing import Dict, List

import numpy as np

from problem.cvrp import CVRP
from problem.route import Route
from problem.solution import Solution


def one_route_per_client(data: CVRP) -> Solution:
    """This heuristic sets one route per client

    Args:
        data (CVRP):

    Returns:
        Solution:
    """
    routes: Dict[int, Route] = dict()
    for c in data.customers:
        r = Route([0, c, 0])
        routes[len(routes)] = r
    return Solution(data, routes)


def closest_client(data: CVRP) -> Solution:
    """This heuristic finds the closest client iteratively.

    Args:
        data (CVRP): instance

    Returns:
        Solution:
    """
    open_clients = data.customers.copy()
    routes: Dict[int, Route] = dict()
    capacity = data.q

    while len(open_clients) > 0:
        # this is the current route
        nodes = [0]  # start at the depot
        route_cap = capacity

        while route_cap > 0 and len(open_clients) > 0:
            next_client = _get_closest_open_client(
                origin=nodes[-1],
                open_clients=open_clients,
                distances=data.distance,
                capacity_left=route_cap,
                demand=data.demand,
            )
            nodes.append(next_client)
            route_cap -= data.demand[next_client]
            open_clients.remove(next_client)

        # there is no more capacity or clients left, close the route
        nodes.append(0)
        route = Route(nodes)
        routes[len(routes)] = route  # Add the route to the solution

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
