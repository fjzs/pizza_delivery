from typing import Dict, List

import numpy as np

from .rcsp import RCSP

_MAX_NUM_SOLUTIONS = 100


class PricingProblem:
    """
    This class has the responsibility of solving the Pricing Problem of the CVRP, which is a
    Resource-Constrained Shortest Path Problem
    """

    def __init__(self, distances: np.ndarray, capacity: float, demand: List[int]):
        """Object initialization

        Args:
            distances (np.ndarray): 2D matrix having all the distances between each pair of nodes
            capacity (float): capacity of the vehicle
            demand (List[int]): demand of each client, goes from 0 to n
        """
        assert len(distances.shape) == 2
        assert capacity >= 1

        self.distances: np.ndarray = distances
        self.capacity: float = capacity
        self.num_clients: int = len(distances) - 1
        self.demand: List[int] = demand
        self.client_duals: Dict[int, float] = dict()
        self.vehicle_cap_dual: float = 0

    def set_duals(self, client_duals: Dict[int, float], vehicle_cap_dual: float):
        """Set the shadow prices of each client and the vehicle cap

        Args:
            duals (Dict[int, float]):
        """
        assert client_duals is not None
        assert vehicle_cap_dual is not None
        assert len(client_duals) == self.num_clients
        assert isinstance(vehicle_cap_dual, float)
        self.client_duals = client_duals
        self.vehicle_cap_dual = vehicle_cap_dual

    def solve(self) -> List[tuple[float, List[int]]]:
        """Solves this problem with the goal of finding reduced-cost
        paths

        Returns:
            solutions (List[tuple[float, List[int]]]): List of (reduced_cost, path)
        """
        # total number of nodes (0...n+1) the last one is the returning depot
        n = self.num_clients + 1

        # the last column is returning to the depot
        cost = np.zeros((n + 1, n + 1))
        # regular distances
        cost[0:n, 0:n] = self.distances
        # from the artificial depot (last row) you cant go anywhere
        cost[-1, :] = np.inf
        # depot to depot is not allowed
        cost[0, -1] = np.inf
        # returning to depot is symmetric
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

        # Resources (demand of each client)
        times = np.zeros((n + 1, n + 1))
        demands = np.zeros((1, n + 1))
        demands[0, 0:n] = self.demand
        times += demands
        # From node i you cant go to node i
        np.fill_diagonal(times, np.inf)
        # print(f"\ntimes:\n{times}")

        rcsp = RCSP(costs=cost, times=times, T=self.capacity, source=0, target=n)
        cost_path_solutions = rcsp.solve_enumeration(
            vehicle_cap_dual=self.vehicle_cap_dual,
            max_clients_in_route=self.capacity,
            max_num_solutions=_MAX_NUM_SOLUTIONS,
        )
        return cost_path_solutions
