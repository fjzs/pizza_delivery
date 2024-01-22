from typing import Dict, Set

import gurobipy as gp
from gurobipy import GRB


class MasterProblem:
    """
    This class has the responsibility of solving the linear Master Problem
    and then to provide the dual variables.
    """

    def __init__(self):
        # Define non-optimization attributes
        # This is maintained to check quickly of duplicates
        self.set_routes = Set[tuple[int, ...]]

        # Define optimization attributes (set_, par_, var_, )
        # Define the sets
        self.set_R = Set[int]

        # Define the parameters
        self.par_clients_per_route = Dict[int, tuple[int, ...]]

    def add_route(self, r: tuple[int, ...]):
        """If the route complies with the API and is unique, its added

        Args:
            r (tuple[int, ...]):
        """
        if self._is_valid_route(r):
            if r not in self.set_routes:
                self.set_routes.add(r)
                id = len(self.set_R)
                self.set_R.add(id)
                self.par_clients_per_route[id] = r
                print(f"route {r} added with id {id}")

            else:
                print(f"route {r} is not unique")

    def _is_valid_route(self, route: tuple[int, ...]) -> bool:
        """Checks if the route complies with the API

        Args:
            route (tuple[int, ...]):

        Returns:
            bool:
        """
        assert route is not None
        assert isinstance(route, tuple)
        assert len(route) >= 3  # depot -> client -> depot is the shortest
        assert route[0] == 0  # start is the depot
        assert route[-1] == 0  # end is the depot
        clients = route[1 : len(route) - 1]
        assert len(clients) == len(set(clients))  # clients must be unique in the route
        return True
