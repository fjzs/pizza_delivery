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
        self.initial_solution: Solution = solution
        self.routes: Set[tuple[int, ...]] = set()
        """This is maintained to check quickly of duplicates, ex:
        {
            (0,3,4,0),
            (0,4,5,0)
        }
        """
        # route id -> list of nodes
        self.clients_per_route: Dict[int, tuple[int, ...]] = dict()
        """This is maintained to quickly build the constraints, ex:
        1: (3,4)
        2: (4,5)
        """

        # Define model attributes (set_, par_, var_, )
        self.model = gp.Model("MasterProblem")

        # Store the constraints here
        self.constraints: Dict[int, gp.Constr] = None

        # Define the sets
        # ids of routes
        num_clients = self.initial_solution.instance.N
        self.set_R: Set[int] = set()
        self.set_N: Set[int] = set(range(1, num_clients + 1))

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
        assert isinstance(r, Route)
        assert cost >= 0

        # This representation is hashable
        nodes_tuple = tuple(r.nodes)
        clients_per_route = nodes_tuple[1:-1]

        if nodes_tuple not in self.routes:
            # Update non-model attributes
            id = len(self.set_R)
            self.routes.add(nodes_tuple)
            self.clients_per_route[id] = clients_per_route

            # Update set
            self.set_R.add(id)

            # Update parameters
            for c in clients_per_route:
                self.par_routes_per_client[c].append(id)
            self.par_cost_per_route[id] = cost
            print(f"route {nodes_tuple} added with id {id}")

        else:
            print(f"route {nodes_tuple} is not unique")

    def build_model(self):
        """Rebuild the model with the updated routes"""
        self.model = gp.Model("MasterProblem")

        # Add variables
        R = self.set_R
        X = self.model.addVars(R, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="X")

        # Add constraints
        N = self.set_N
        self.constraints = self.model.addConstrs(
            (
                (gp.quicksum(X[r] for r in self.par_routes_per_client[i]) >= 1.0)
                for i in N
            ),
            name="serve_client",
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

    def get_duals(self) -> Dict[int, float]:
        """Retrieves the shadow prices associated to the primal constraint
        of client service.

        Returns:
            client_dual (Dict[int, float]): client id -> shadow price
        """

        self.model.optimize()
        # self.model.display()

        # https://www.gurobi.com/documentation/9.5/refman/pi.html
        # https://support.gurobi.com/hc/en-us/community/posts/15340462880785-How-to-retrieve-the-dual-variable-value-for-a-linear-constraint-in-a-SOCP-problem
        client_dual: Dict[int, float] = dict()

        for client_id, c in self.constraints.items():
            print(f"client id: {client_id}, dual: {c.Pi}")
            client_dual[client_id] = c.Pi

        assert len(client_dual) == len(self.set_N)
        return client_dual
