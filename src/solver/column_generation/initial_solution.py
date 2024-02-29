from typing import Dict, List, Tuple

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

            # No client potentialy left to fit in the route
            if next_client is None:
                break

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
        closest_client (int): could be None
    """
    subset = [i for i in open_clients if (i != origin) and (capacity_left >= demand[i])]
    closest_client = None
    if len(subset) > 0:
        min_index = np.argmin(distances[origin, subset])
        closest_client = subset[min_index]
    return closest_client


def clarke_and_wright(data: CVRP) -> Solution:
    """Applies Clarke & Wright heuristic

    Args:
        data (CVRP):

    Returns:
        Solution:
    """
    print(f"\nRunning Clarke & Wright:")
    # First start from the one-route-per-client initial solution
    # Each element in this list is a tuple of ([0,c1,c2,...,0], cap_used)
    routes = []
    for c in data.customers:
        routes.append(([0, c, 0], data.demand[c]))
    # print(f"Initial routes:")
    # for r in routes:
    #     print(f"\t{r}")

    # Compute savings
    # Each element is (saving $, i, j)
    savings: List[float, Tuple[int, int]] = []
    for i in data.customers:
        for j in data.customers:
            if i < j:
                saving = data.distance[0, i] + data.distance[0, j] - data.distance[i, j]
                if saving > 0:
                    savings.append((saving, i, j))
    savings.sort(reverse=True)
    # print(f"\nSavings:")
    # for s in savings:
    #     print(f"\t{s}")

    while len(savings) > 0:

        savings_to_remove = []
        for s in savings:
            saving, i, j = s

            # Determine whether we can put the link (i,j) or (j,i) together
            index_i = _cw_get_route_index_of_customer(i, routes)
            index_j = _cw_get_route_index_of_customer(j, routes)
            ri, ki = routes[index_i]  # route and cap
            rj, kj = routes[index_j]  # route and cap

            # First check: combining the capacities is feasible
            if ki + kj > data.q:
                savings_to_remove.append(s)
                continue

            # Find which is the route_ini and route_end to be merged
            route_ini = None
            route_end = None

            # Case (1): rj starts with j (0,j,....) and ri ends with i (0,...,i,0)
            if rj[1] == j and ri[-2] == i:
                route_ini = rj
                route_end = ri

            # Case (2): ri starts with i (0,i,....) and rj ends with j (0,...,j,0)
            elif ri[1] == i and rj[-2] == j:
                route_ini = ri
                route_end = rj

            # Case (3): i or j are interior, cant be merged
            else:
                continue

            # Apply the merge
            route_merge = _cw_merge_routes(route_ini, route_end)
            routes.remove((ri, ki))
            routes.remove((rj, kj))
            routes.append((route_merge, ki + kj))
            savings_to_remove.append(s)

        # Remove unfeasible and applied savings
        savings = [s for s in savings if s not in savings_to_remove]
        # print(f"{len(savings)} savings left")

    # print(f"The final routes are:")
    # for r in routes:
    #     print(r)

    # Assemble solution
    routes_dict = {}
    for i, (nodes, cap) in enumerate(routes):
        routes_dict[i + 1] = Route(nodes)
    solution = Solution(data, routes_dict)
    return solution


def _cw_get_route_index_of_customer(id: int, routes: list):
    """Internal to Clarke & Wright.
    Retrieves the index of the route having this customer

    Args:
        id (int):
        routes (list):
    """
    for i, (r, cap) in enumerate(routes):
        if id in r:
            return i
    return None


def _cw_merge_routes(route_ini: list, route_end: list) -> list:
    """Merges route_ini with route_end through the last link of
    route_ini with the initial link of route_end. We remove link
    (i,0) from the first, remove (j,0) from the second and merge them
    through (i,j)

    Args:
        route_ini (list): this is the structure [0,...,i,0]
        route_end (list): this is the structure [0,j....,0]

    Returns:
        route_merge (list): the structure is [0,...,i,j,...,0]
    """
    assert len(route_ini) >= 3
    assert len(route_end) >= 3
    route_merge = route_ini.copy()
    route_merge = route_merge[0:-1]  # remove the last node
    second_part = route_end.copy()[1:]  # starting from j
    route_merge.extend(second_part)
    return route_merge
