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
        [self.instance.is_valid_route(r) for r in self.routes.values()]

        # Check if the routes cover all the clients
        clients_covered = set()
        for r in self.routes.values():
            clients_covered.update(r.clients)
        assert clients_covered == set(self.instance.customers)

        # Compute the cost per route and solution cost
        self.cost_per_route: Dict[int, float] = dict()
        for id, r in self.routes.items():
            cost = instance.get_route_cost(r)
            self.cost_per_route[id] = cost

        self.total_cost: float = sum(self.cost_per_route.values())
