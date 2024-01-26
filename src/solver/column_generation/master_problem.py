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

    Σ_r X_r * a_ir >= 1, Ɐ i ∈ N - {0} (attend the clients)

    Σ_r X_r <= K (vehicles capacity)

    X_r ∈ {0,1}

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

        # Store the constraints here
        self.constraints: Dict[int, gp.Constr] = None

        # Define the sets
        # ids of routes
        num_clients = self.instance.N
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

            # Update set R
            self.set_R.add(id)

            # Update parameters
            for c in clients_per_route:
                self.par_routes_per_client[c].append(id)
            self.par_cost_per_route[id] = cost
            print(f"\troute {nodes_tuple} added with id {id}")

        else:
            print(f"\troute {nodes_tuple} is not unique")

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
        self._add_constraint_vehicles_limit()

    def _add_constraint_vehicles_limit(self):
        """Use at most K vehicles"""
        self.con_vehicle_limit = self.model.addConstr(
            -gp.quicksum(self.var_X[r] for r in self.set_R) >= -self.instance.K,
            name="vehicles_limit",
        )

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
                    for i in self.set_N
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
                    for i in self.set_N
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

    def get_solution(self) -> Solution:
        """Get the solution after solving the integer problem

        Returns:
            Solution:
        """

        def remove_0_values(d: dict(), tolerance=float("1.0e-10")) -> dict():
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

    def get_duals(self) -> tuple[Dict[int, float], float]:
        """After solving the linear relaxation of the Master Problem
        retrieve the shadow prices associated to the constraints. We
        have |N|-1 constraints and duals (λi) associated with serving the
        clients and 1 constraint and dual (μ) with the vehicle capacity.

        Returns:
            client_dual (tuple[Dict[int, float], float]): {i: λi}, μ
        """

        # self.model.display()

        # https://www.gurobi.com/documentation/9.5/refman/pi.html
        # https://support.gurobi.com/hc/en-us/community/posts/15340462880785-How-to-retrieve-the-dual-variable-value-for-a-linear-constraint-in-a-SOCP-problem
        client_dual: Dict[int, float] = dict()
        for client_id, c in self.con_serve_clients.items():
            # print(f"client id: {client_id}, dual: {c.Pi}")
            client_dual[client_id] = c.Pi

        vehicle_cap_dual = self.con_vehicle_limit.Pi
        # print(f"vehicle cap dual: {vehicle_cap_dual}")

        assert len(client_dual) == len(self.set_N)
        return client_dual, vehicle_cap_dual
