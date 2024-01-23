from typing import Dict, List, Set

import gurobipy as gp
from gurobipy import GRB

from problem.route import Route
from problem.solution import Solution


class MasterProblem:
    """
    This class has the responsibility of solving the linear Master Problem
    and then to provide the dual variables.
    """

    def __init__(self, solution: Solution):
        """Creates the master problem

        Args:
            solution (Solution): this is a feasible solution
        """
        assert solution is not None

        # Define non-optimization attributes
        self.initial_solution = solution
        # This is maintained to check quickly of duplicates
        self.routes: Set[tuple[int, ...]] = set()
        # route id -> list of nodes
        self.clients_per_route: Dict[int, tuple[int, ...]] = dict()

        # Define model attributes (set_, par_, var_, )
        self.model = gp.Model("MasterProblem")

        # Define the sets
        # ids of routes
        self.set_R: Set[int] = set()
        self.set_N: Set[int] = set(range(1, self.initial_solution.instance.N + 1))

        # Define the parameters
        # client_id -> list of routes that visit it
        self.par_routes_per_client: Dict[int, List[int]] = dict()
        for i in self.set_N:
            self.par_routes_per_client[i] = []

        # route id -> cost of route ($)
        self.par_cost_per_route: Dict[int, float] = dict()

        # Load the initial solution
        for route_id in self.initial_solution.routes.keys():
            route = self.initial_solution.routes[route_id]
            cost = self.initial_solution.cost_per_route[route_id]
            self.add_route(route, cost)

    def add_route(self, r: Route, cost: float):
        """If the route is unique its added to the model. The route
        is something like (0, 5,1,..., 0)

        Args:
            r (tuple[int, ...]): the route
            cost (float): cost of the route
        """
        assert r is not None
        assert r[0] == 0
        assert r[-1] == 0
        assert len(r) >= 3

        if r not in self.routes:
            # Update non-model attributes
            id = len(self.set_R)
            self.routes.add(r)
            self.clients_per_route[id] = r

            # Update set
            self.set_R.add(id)

            # Update parameters
            clients_per_route = r[1 : len(r) - 1]
            for c in clients_per_route:
                self.par_routes_per_client[c].append(c)
            self.par_cost_per_route[id] = cost
            print(f"route {r} added with id {id}")

        else:
            print(f"route {r} is not unique")

    def build_model(self):
        """Rebuild the model with the updated routes"""
        self.model = gp.Model("MasterProblem")

        # Add variables
        R = self.set_R
        X = self.model.addVars(R, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="X")

        # Add constraints
        M = self.set_N
        self.model.addConstrs(
            (gp.quicksum(X[r] for r in self.par_routes_per_client[i]) >= 1.0) for i in M
        )

        # Objective function
        self.model.setObjective(
            gp.quicksum(X[r] * self.par_cost_per_route[r] for r in R),
            sense=GRB.MINIMIZE,
        )

        # Update the model after the creation
        self.model.update()
        print(f"\n\nCONSTRAINTS:")
        for i, con in enumerate(self.model.getConstrs()):
            print(f"\n{con}:\n{self.model.getRow(con)} {con.Sense} {con.RHS}")
