from typing import Dict

from problem.cvrp import CVRP
from problem.route import Route


class Solution:
    def __init__(self, instance: CVRP, routes: Dict[int, Route]):
        """This is a feasible solution

        Args:
            instance (CVRP):
            routes (Dict[int, Route]): route per vehicle
        """
        self.instance: CVRP = instance
        self.routes: Dict[int, Route] = routes

        # Check if they are feasible, error if not
        for r in self.routes.values():
            assert self.instance.is_valid_route(r)

        # Check if the routes cover all the clients
        clients_covered = set()
        for r in self.routes.values():
            clients_covered.update(r.clients)
        assert clients_covered == set(self.instance.customers)

    def get_cost(self) -> float:
        """Retrieves the cost of this solution

        Returns:
            cost (float):
        """
        cost = 0
        for r in self.routes.values():
            cost += self.instance.get_route_cost(r)
        return cost
