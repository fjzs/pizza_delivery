from typing import List, Tuple

from problem.cvrp import CVRP, Route
from problem.solution import Solution


def get_neighborhood(solution: Solution) -> List[Solution]:
    instance = solution.instance
    neighborhood = []  # list of potential solutions
    for id_i, ri in solution.routes.items():
        for id_j, rj in solution.routes.items():
            if id_i != id_j:
                new_ri, new_rj = get_2_opt_inter_route_solutions(ri, rj, instance)

                # Assemble this new solution
                new_route_dict = solution.routes.copy()
                del new_route_dict[id_i]
                del new_route_dict[id_j]
                if new_ri is not None:
                    new_route_dict[id_i] = new_ri
                if new_rj is not None:
                    new_route_dict[id_j] = new_rj
                new_sol = Solution(instance, new_route_dict)
                neighborhood.append(new_sol)
    return neighborhood


def get_2_opt_inter_route_solutions(
    route_i: Route, route_j: Route, instance: CVRP
) -> Tuple[Route, Route]:
    """Given a route, applies 2-opt inter-route heuristic and retrieves
    the best option.

    Args:
        route_i (Route):
        route_j (Route):
        instance (CVRP):

    Returns:
        new_route_i, new_route_j
    """
    ni = route_i.nodes
    nj = route_j.nodes
    min_cost = instance.get_route_cost(route_i) + instance.get_route_cost(route_j)
    best_route_i = route_i
    best_route_j = route_j
    for i in range(0, len(ni) - 1):
        for j in range(0, len(nj) - 1):
            new_nodes_i = ni[0 : i + 1].copy() + nj[j + 1 :].copy()
            new_nodes_j = nj[0 : j + 1].copy() + ni[i + 1 :].copy()
            new_route_i = Route(new_nodes_i)
            new_route_j = Route(new_nodes_j)
            if instance.is_valid_route(new_route_i) and instance.is_valid_route(
                new_route_j
            ):
                cost = instance.get_route_cost(new_route_i) + instance.get_route_cost(
                    new_route_j
                )
                if cost < min_cost:
                    min_cost = cost
                    best_route_i, best_route_j = new_route_i, new_route_j
    return best_route_i, best_route_j
