from typing import Dict, List

import numpy as np

from .espcc import ESPCC

_MAX_NUM_SOLUTIONS = 1


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
        

    def solve(self) -> List[tuple[float, List[int]]]:
        """Solves this problem with the goal of finding reduced-cost
        paths from depot to clients and to depot

        Returns:
            solutions (List[tuple[float, List[int]]]): List of (reduced_cost, path)
        """
        # Number of nodes in the original network
        n = self.num_nodes

        # Create the adapted network of the ESPCC, that has a depot connected to every
        # client, all the clients connected to each other, and all the clients connected
        # to the target (which is the depot)
        
        # the last column is returning to the depot
        cost = np.zeros((n + 1, n + 1))
        # regular distances
        cost[0: n, 0: n] = self.distances
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
        print(f"duals_col:\n{duals_col}")
        print(f"\ncost before duals:\n{cost}")
        cost -= duals_col
        # From node i you cant go to node i
        np.fill_diagonal(cost, np.inf)
        print(f"\ncost after duals:\n{cost}")

        # Resources (demand of each client)
        times = np.zeros((n + 1, n + 1))
        demands = np.zeros((1, n + 1))
        demands[0, 0:n] = self.demand
        times += demands
        # From node i you cant go to node i
        np.fill_diagonal(times, np.inf)
        print(f"\ntimes:\n{times}")


        elementary_path_problem = ESPCC(costs=cost, times=times, T=self.capacity, source=0, target=n)
        cost_path_solutions = elementary_path_problem.solve(
            method="enumeration",
            max_num_solutions=_MAX_NUM_SOLUTIONS,
            max_clients_per_path= self.max_clients_per_route
        )
        #cost_path_solutions = elementary_path_problem._solve_exact_bidirectional()
        
        return cost_path_solutions
