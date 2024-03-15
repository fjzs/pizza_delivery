from typing import Dict, List

import numpy as np

from .espcc import ESPCC

ALPHA_BEST_EDGES = [None, 0.3, 0.4, 0.5, 0.7, 0.9]


class PricingProblem:
    """
    This class has the responsibility of solving the Pricing Problem of the CVRP, which is an
    Elementary Shortest Path Problem with Resource Constraints
    """

    def __init__(self, distances: np.ndarray, capacity: float, demand: List[int]):
        """This class assumes the index 0 is the depot

        Args:
            distances (np.ndarray): 2D matrix having all the distances between each pair of nodes
            capacity (float): capacity of the vehicle
            demand (List[int]): demand of each client, goes from 0 to n-1, n being the number of nodes
        """
        assert len(distances.shape) == 2
        assert capacity >= 1

        self.distances: np.ndarray = distances
        self.capacity: float = capacity
        self.num_nodes: int = len(distances)
        self.demand: List[int] = demand
        self.client_duals: Dict[int, float] = dict()
        self.max_clients_per_route = int(np.floor(self.capacity / min(self.demand[1:])))
        print(f"max clients per route: {self.max_clients_per_route}")

    def set_duals(self, client_duals: Dict[int, float]):
        """Set the shadow prices of each client

        Args:
            duals (Dict[int, float]):
        """
        assert client_duals is not None
        assert len(client_duals) == self.num_nodes - 1
        self.client_duals = client_duals

    def _set_up_cost_and_times(self):
        """Set up the cost matrix and the times used on each arc.
        It will initialize the self.cost and self.time attributes
        of this class
        """
        # Number of nodes in the original network
        n = self.num_nodes

        # Create the adapted network of the ESPCC, that has a depot connected to every
        # client, all the clients connected to each other, and all the clients connected
        # to the target (which is the depot)

        # the last column is returning to the depot
        cost = np.zeros((n + 1, n + 1))
        # regular distances
        cost[0:n, 0:n] = self.distances
        # from the artificial depot (last row) you cant go anywhere
        cost[-1, :] = np.inf
        # depot to depot is not allowed
        cost[0, -1] = np.inf
        # distance when returning to artificial depot is symmetric
        cost[1:, -1] = cost[1:, 0]
        # Nobody can go to node 0
        cost[:, 0] = np.inf

        # Substract the duals to obtain the reduced cost:
        # reduced_c_ij = c_ij - pi_i (this is the dual)
        # duals of each node [0, lambda_1, lambda_2, ..., lambda_n, 0]
        duals_col = np.zeros((n + 1, 1))
        for id_client, dual_value in self.client_duals.items():
            duals_col[id_client, 0] = dual_value
        # print(f"duals_col:\n{duals_col}")
        # print(f"\ncost before duals:\n{cost}")
        cost -= duals_col
        # From node i you cant go to node i
        np.fill_diagonal(cost, np.inf)
        # print(f"\ncost after duals:\n{cost}")
        self.cost = cost

        # Resources (demand of each client)
        times = np.zeros((n + 1, n + 1))
        demands = np.zeros((1, n + 1))
        demands[0, 0:n] = self.demand
        times += demands
        np.fill_diagonal(times, np.inf)
        self.time = times

    def _get_cost_BestEdges(self, alpha: float = None) -> np.ndarray:
        """Sets to infinity those edges which d_ij > alpha * max_dual

        Args:
            alpha (float, optional):

        Returns:
            cost (np.ndarray):
        """
        cost = self.cost.copy()
        if alpha:
            max_dual = max(self.client_duals.values())
            n = self.num_nodes
            for i in range(n):
                for j in range(n):
                    d_ij = self.distances[i, j]
                    if i != j and d_ij > alpha * max_dual:
                        # by setting this to inf, the edge wont be available
                        cost[i, j] = np.inf
                        if i == 0:
                            # this is the cost from the customer to the virtual depot
                            cost[j, n] = np.inf
        return cost

    def solve(self) -> List[tuple[float, List[int]]]:
        """Solves this problem with the goal of finding reduced-cost
        routes

        Returns:
            solutions (List[tuple[float, List[int]]]): List of (reduced_cost, path)
        """
        self._set_up_cost_and_times()

        # Apply the 'Best Edges Strategy' from https://pubsonline.informs.org/doi/10.1287/trsc.1050.0118
        # --> remove edges such that cost_ij > alpha * max_dual_variable
        # 0 < alpha < 1 is a parameter

        cost_paths = []
        print(f"\nStarting Alpha-Pricing")
        for alpha in ALPHA_BEST_EDGES:
            print(f"\n\tAlpha = {alpha}")
            cost = self._get_cost_BestEdges(alpha)
            elementary_path_problem = ESPCC(
                costs=cost,
                times=self.time,
                T=self.capacity,
                source=0,
                target=self.num_nodes,
            )
            # Solve the problem
            cost_path_solutions = elementary_path_problem.solve()
            cost_paths.extend(cost_path_solutions)

        return cost_paths
