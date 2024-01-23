from typing import Dict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class PricingProblem:
    """
    This class has the responsibility of solving the Pricing Problem of the CVRP, which is a
    Resource-Constrained Shortest Path Problem
    """

    def __init__(
        self, distances: np.ndarray, capacity: float, demand: Dict[int, float]
    ):
        """Object initialization

        Args:
            distances (np.ndarray): 2D matrix having all the distances between each pair of nodes
            capacity (float): capacity of the vehicle
            demand (Dict[int, float]): demand of each client
        """
        assert len(distances.shape) == 2
        assert capacity >= 1

        self.distances: np.ndarray = distances
        self.capacity: float = capacity
        self.num_clients: int = len(distances) - 1
        self.demand: Dict[int, float] = demand
        self.duals: Dict[int, float] = dict()  # client -> dual
        self.mu = 1.0

    def set_duals(self, duals: Dict[int, float]):
        """Set the shadow prices of each client

        Args:
            duals (Dict[int, float]):
        """
        assert duals is not None
        assert len(duals) == self.num_clients
        self.duals = duals

    def _create_graph(self):
        n = self.num_clients + 1  # total number of nodes
        dist_matrix = np.zeros(
            (n + 1, n + 1)
        )  # the last column is returning to the depot
        dist_matrix[0:n, 0:n] = self.distances  # regular distances
        dist_matrix[-1, :] = np.inf  # from the artificial depot you cant go anywhere
        dist_matrix[0, -1] = np.inf  # depot to depot is not allowed
        dist_matrix[1:, -1] = dist_matrix[1:, 0]  # returning to depot is symmetric

        # Substract the duals
        duals_row = np.zeros((1, n + 1))
        for id_client, dual_value in self.duals.items():
            duals_row[0, id_client] = dual_value
        print(f"dual row: {duals_row}")
        print(f"\nmatrix so far:\n{dist_matrix}")
        dist_matrix -= duals_row

        print(f"\nafter duals:\n{dist_matrix}")

        graph = csr_matrix(dist_matrix)
        dist_matrix, predecessors = dijkstra(
            csgraph=graph, directed=False, indices=0, return_predecessors=True
        )
        print(dist_matrix)
        print(predecessors)

    def solve(self):
        self._create_graph()
