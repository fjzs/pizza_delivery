from typing import List, Set

import gurobipy as gp
import numpy as np
from gurobipy import GRB


class RCSP:
    """This class has the responsibility of solving the
    Resource-Constrained Shortest Path Problem.
    Source: Network Flows - Theory, Algorithms and Applications (1993)
    book, page 598.
    """

    def __init__(
        self, costs: np.ndarray, times: np.ndarray, T: float, source: int, target: int
    ):
        """Initialization of the problem

        Args:
            - costs (np.ndarray): this is the cost to minimize, it has shape (N, N)
            - times (np.ndarray): this is the resource, it has shape (N, N)
            - T (float): this is the total amount of resource available
            - source (int): starting node index
            - target (int): ending node index
        """
        # Check costs
        assert costs is not None
        assert isinstance(costs, np.ndarray)
        assert len(costs.shape) == 2
        assert costs.shape[0] == costs.shape[1]  # square matrix
        N = costs.shape[0]

        # Check times
        assert times is not None
        assert isinstance(times, np.ndarray)
        assert len(times.shape) == 2
        assert times.shape == costs.shape
        assert np.all(times >= 0)
        assert T >= 0

        # Check source and target
        assert source is not None
        assert target is not None
        assert 0 <= source < N
        assert 0 <= target < N
        assert source != target

        # Create optimization model
        self.model = gp.Model("RCSP")
        self.set_N = set(range(N))
        self.par_cost = costs
        self.set_A = self._get_set_arcs()
        self.par_time = times
        self.par_source = source
        self.par_target = target
        self.par_max_time = T
        self.var_X = self._add_var_X()
        self._add_constraint_net_flow()
        self._add_constraint_max_out_is_1()
        self._add_constraint_time()
        self._set_objective_function()
        self.model.update()

        print(f"\n\nCONSTRAINTS:")
        for i, con in enumerate(self.model.getConstrs()):
            print(f"\n{con}:\n{self.model.getRow(con)} {con.Sense} {con.RHS}")

    def _get_set_arcs(self) -> Set[tuple[int, int]]:
        """Creates the set of arcs where the cost to travel is < ∞

        Returns:
            Set[tuple[int,int]]: arcs (i,j)
        """
        arcs = set()
        for i in self.set_N:
            for j in self.set_N:
                if self.par_cost[i, j] < np.inf:
                    arcs.add((i, j))
        return arcs

    def _add_var_X(self):
        """
        X_ij: 1 if the arc (i,j) is used, 0 if not
        """
        return self.model.addVars(self.set_A, lb=0, ub=1, vtype=GRB.BINARY, name="X")

    def _add_constraint_net_flow(self):
        """
        Everything going out - Everything entering = netflow. netflow of node i =
        * 1 if i = source
        * -1 if i = target
        * 0 in other case
        """
        A = self.set_A
        N = self.set_N
        netflow = [0] * len(N)
        netflow[self.par_source] = 1
        netflow[self.par_target] = -1
        self.model.addConstrs(
            (
                gp.quicksum(self.var_X[(i, j)] for j in N if (i, j) in A)
                - gp.quicksum(self.var_X[(j, i)] for j in N if (j, i) in A)
                == netflow[i]
                for i in N
            ),
            name="flow_conservation",
        )

    def _add_constraint_max_out_is_1(self):
        """Because there could be negative costs, here we ensure that
        the maximum number of arcs coming out from a node is 1
        * Σ_{j:(i,j) ∈ A} X_ij <= 1, Ɐ i ∈ N
        """
        A = self.set_A
        N = self.set_N
        self.model.addConstrs(
            (gp.quicksum(self.var_X[(i, j)] for j in N if (i, j) in A) <= 1 for i in N),
            name="max_outflow_is_1",
        )

    def _add_constraint_time(self):
        """
        The total time used must be <= than the availability
        """
        A = self.set_A
        self.model.addConstr(
            gp.quicksum(self.var_X[(i, j)] * self.par_time[i, j] for (i, j) in A)
            <= self.par_max_time,
            name="resource_constraint",
        )

    def _set_objective_function(self):
        """
        Minimize total cost
        """
        A = self.set_A
        self.model.setObjective(
            gp.quicksum(self.var_X[(i, j)] * self.par_cost[i, j] for (i, j) in A),
            sense=GRB.MINIMIZE,
        )

    def solve(self):
        print("\n\n\nSOLVING RCSP\n")
        self.model.optimize()

        def remove_0_values(d: dict(), tolerance=float("1.0e-10")) -> dict():
            return {str(k): v for (k, v) in d.items() if v > tolerance}

        solution = None
        if self.model.status == GRB.OPTIMAL:
            objective_function_value = self.model.ObjVal
            arcs_used = remove_0_values(self.model.getAttr("X", self.var_X))
            print(arcs_used)
