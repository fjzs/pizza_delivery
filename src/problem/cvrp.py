import os
from typing import Dict, List

import numpy as np
from scipy.spatial.distance import cdist

from problem import instance
from problem.route import Route


class CVRP:
    """This class has all the data for an instance of the Capacitated
    Vehicle Routing Problem, (but does not solve it)"""

    def __init__(self, filepath: str):
        data = instance.load(filepath)
        self.q: int = data["Q"]
        """
        Capacity of vehicles
        """
        self.n: int = data["N"]
        """
        Number of nodes of this instance. Those are {0, 1, .., n-1}
        """
        self.customers: List[int] = list(range(1, self.n))
        """
        List of customers, they have ids {1, ..., n-1}
        """
        self.demand: List[int] = data["demand"]
        """
        Demand of each node. d[0] = 0
        """
        self.coordinates: np.ndarray = np.asarray(data["coordinates"])
        """
        [x_i, y_i] of each node i
        """
        self.distance: np.ndarray = self._get_distances(self.coordinates)
        """
        Distance matrix with shape (n,n)
        """
        assert self.q >= 1
        assert self.n >= 2
        assert len(self.demand) == self.n
        assert self.demand[0] == 0
        assert len(self.coordinates.shape) == 2
        assert self.coordinates.shape[0] == self.n
        assert len(self.coordinates.shape) == 2
        assert all(d >= 0 for d in self.demand)

    def is_valid_route(self, r: Route) -> bool:
        """Checks the validity of the route:
        * Has to start at the depot
        * Has to end at the depot
        * The number of total nodes visited must be >= 3
        * The clients (different than the depot) must be unique
        * Total demand covered must be <= capacity

        Args:
            route (Route):

        Returns:
            bool: True if route is valid, Error if not
        """
        if r is None:
            return False
        if len(r.nodes) <= 2:
            return False
        if r.nodes[0] != 0:  # start is the depot
            return False
        if r.nodes[-1] != 0:  # end is the depot
            return False
        if len(set(r.nodes[1:-1])) != len(
            r.nodes[1:-1]
        ):  # there are duplicated clients
            return False

        demand_covered = sum([self.demand[i] for i in r.clients])
        if demand_covered > self.q:  # constraint capacity
            return False

        return True

    def get_route_cost(self, r: Route) -> float:
        """Compute the cost of a given route.

        Args:
            r (Route):

        Returns:
            cost(float):
        """
        cost = 0
        i = r.nodes[0]
        for j in r.nodes[1:]:
            cost += self.distance[i, j]
            i = j
        return cost

    def _get_distances(self, points: np.ndarray, decimals=1) -> np.ndarray:
        """Computes the euclidean distances between each pair of points

        Args:
            points (np.ndarray): (n, 2) array

        Returns:
            distances (np.ndarray): (n, n) matrix
        """
        assert isinstance(points, np.ndarray)
        assert len(points.shape) == 2
        assert points.shape[1] == 2
        n = len(points)
        distances = np.round(cdist(points, points, metric="euclidean"), decimals)
        assert distances.shape == (n, n)
        assert np.all(distances >= 0)
        return distances
