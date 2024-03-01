from typing import List

from problem.cvrp import CVRP, Route
from problem.solution import Solution


def get_neighborhood(solution: Solution) -> List[Solution]:
    instance = solution.instance
    neighborhood = []  # list of potential solutions
    for route_id, route in solution.routes.items():
        improved_route = get_2_opt_intra_route_solutions(route, instance)
        new_route_dict = solution.routes.copy()
        new_route_dict[route_id] = improved_route
        new_sol = Solution(instance, new_route_dict)
        neighborhood.append(new_sol)
    return neighborhood


def get_2_opt_intra_route_solutions(route: Route, instance: CVRP) -> Route:
    """Given a route, applies 2-opt intra-route heuristic and retrieves
    the best option.

    Args:
        route (Route):
        instance (CVRP):

    Returns:
        best_route (Route):
    """

    # Base case with no optimization
    if len(route.clients) == 1:
        return route

    n = len(route.nodes)  # total number of nodes
    min_cost = instance.get_route_cost(route)
    best_route = route
    for i in range(1, n - 1):  # pivot customer, try to put it in every other position
        for j in range(1, n - 1):  # swap i with j
            if i != j:
                nodes = route.nodes.copy()
                temp_j = nodes[j]
                nodes[j] = nodes[i]
                nodes[i] = temp_j
                new_route = Route(nodes)
                cost = instance.get_route_cost(new_route)
                if cost < min_cost:
                    min_cost = cost
                    best_route = new_route
    return best_route
