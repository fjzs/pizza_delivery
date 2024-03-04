from typing import Dict, List, Set

import gurobipy as gp
from gurobipy import GRB

from problem.cvrp import CVRP
from problem.route import Route
from problem.solution import Solution


class MasterProblem:
    """
    This class has the responsibility of solving the restricted Master Problem,
    both in its linear and integer form. The problem is:

    Minimize Σ_r X_r * c_r

    subject to:

    * Σ_r X_r * a_ir >= 1, Ɐ i ∈ C (attend the clients)

    * X_r ∈ {0,1}

    """

    def __init__(self, solution: Solution):
        """Creates the master problem

        Args:
            solution (Solution): this is a feasible solution
        """
        assert solution is not None

        # Define non-model attributes
        self.initial_solution: Solution = solution
        self.instance: CVRP = solution.instance
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

        # Define the sets
        self.set_R: Set[int] = set()  # set of routes
        self.set_C: Set[int] = set(self.instance.customers)  # set of customers

        # Define the parameters
        # client_id -> list of routes that visit it
        self.par_routes_per_client: Dict[int, List[int]] = dict()
        for i in self.set_C:
            self.par_routes_per_client[i] = []

        # route id -> cost of route ($)
        self.par_cost_per_route: Dict[int, float] = dict()

        # Load the initial solution
        for route_id in self.initial_solution.routes.keys():
            route = self.initial_solution.routes[route_id]
            cost = self.instance.get_route_cost(route)
            self.add_route(route, cost)

    def add_route(self, r: Route, cost: float):
        """If the route is unique its added to the model. The route
        is something like (0, a, b, c, 0)

        Args:
            r (tuple[int, ...]): the route
            cost (float): cost of the route
        """
        assert r is not None
        assert isinstance(r, Route)
        assert cost >= 0

        # This representation is hashable and so it can check if the route is repeated
        # in O(1)
        route_nodes = tuple(r.nodes)
        route_nodes_reversed = route_nodes[::-1]
        clients_per_route = route_nodes[1:-1]  # don't loose the order

        if (route_nodes not in self.routes) and (
            route_nodes_reversed not in self.routes
        ):
            # Update non-model attributes
            id = len(self.set_R)
            self.routes.add(route_nodes)
            self.clients_per_route[id] = clients_per_route

            # Update set R
            self.set_R.add(id)

            # Update parameters
            for c in clients_per_route:
                self.par_routes_per_client[c].append(id)
            self.par_cost_per_route[id] = cost
            print(f"\troute {route_nodes} added with id {id}")

        else:
            print(f"\troute {route_nodes} is not unique")

    def build_model(self, is_linear: bool):
        """Builds the model with the updated routes.

        Args:
            is_linear (bool): If true, then solve the linear version, otherwise
            solve the MIP
        """
        self.model = gp.Model("MasterProblem")

        # Set variables, constraint and objective function
        self._add_variables(is_linear)
        self._add_constraints(is_linear)
        self._set_objective_function()
        self.model.update()
        # print(f"\n\nCONSTRAINTS:")
        # for i, con in enumerate(self.model.getConstrs()):
        #     print(f"\n{con}:\n{self.model.getRow(con)} {con.Sense} {con.RHS}")

        # self._print_routes()

    def _add_variables(self, is_linear: bool):
        """Add the variables of the model:

        X_r: 1 if route r is used, 0 if not.

        Args:
            linear (bool):
        """
        if is_linear:
            self.var_X = self.model.addVars(
                self.set_R, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="X"
            )
        else:
            self.var_X = self.model.addVars(
                self.set_R, lb=0, ub=1, vtype=GRB.BINARY, name="X"
            )

    def _add_constraints(self, is_linear: bool):
        """Adds the contraints of the model

        Args:
            linear (bool):
        """
        self._add_constraint_serve_clients(is_linear)

    def _add_constraint_serve_clients(self, is_linear: bool):
        """Each customer must be served

        Args:
            linear (bool):
        """
        if is_linear:
            # Serve each client
            self.con_serve_clients = self.model.addConstrs(
                (
                    (
                        gp.quicksum(
                            self.var_X[r] for r in self.par_routes_per_client[i]
                        )
                        >= 1.0
                    )
                    for i in self.set_C
                ),
                name="serve_client",
            )
        else:
            # Server each client once
            self.model.addConstrs(
                (
                    (
                        gp.quicksum(
                            self.var_X[r] for r in self.par_routes_per_client[i]
                        )
                        == 1.0
                    )
                    for i in self.set_C
                ),
                name="serve_client",
            )

    def _set_objective_function(self):
        self.model.setObjective(
            gp.quicksum(self.var_X[r] * self.par_cost_per_route[r] for r in self.set_R),
            sense=GRB.MINIMIZE,
        )

    def solve(self):
        """Solves the master problem"""
        self.model.optimize()

    def get_Obj_Value(self) -> float:
        """Retrieves:
        * ObjVal (Objective value for current solution)

        Source: https://www.gurobi.com/documentation/current/refman/attributes.html
        https://www.gurobi.com/documentation/current/refman/objval.html

        Returns:
            * ObjVal (float)
        """
        return self.model.ObjVal

    def get_solution(self) -> Solution:
        """Get the solution after solving the integer problem

        Returns:
            Solution:
        """

        def remove_0_values(d: Dict, tolerance=float("1.0e-10")) -> Dict:
            return {k: v for (k, v) in d.items() if v > tolerance}

        routes_ids_used = remove_0_values(self.model.getAttr("X", self.var_X))
        routes_solution: Dict[int, Route] = dict()
        for i, r_id in enumerate(routes_ids_used):
            clients_of_route = self.clients_per_route[r_id]
            nodes = [0] + list(clients_of_route) + [0]
            routes_solution[i] = Route(nodes)

        solution = Solution(
            instance=self.initial_solution.instance, routes=routes_solution
        )
        return solution

    def get_duals(self) -> Dict[int, float]:
        """After solving the linear relaxation of the Master Problem
        retrieve the shadow prices associated to the constraint. We
        have |C| constraints and duals (λi) associated with serving the
        clients.

        Returns:
            client_dual (Dict[int, float]): {i: λi}
        """

        # self.model.display()

        # https://www.gurobi.com/documentation/9.5/refman/pi.html
        # https://support.gurobi.com/hc/en-us/community/posts/15340462880785-How-to-retrieve-the-dual-variable-value-for-a-linear-constraint-in-a-SOCP-problem
        client_dual: Dict[int, float] = dict()
        for client_id, c in self.con_serve_clients.items():
            client_dual[client_id] = c.Pi

        assert len(client_dual) == len(self.set_C)
        return client_dual

    def _print_routes(self):
        print(f"Clients per route in the Master Problem:")
        for id, clients in self.clients_per_route.items():
            print(f"\tr={id+1}: {clients}")
