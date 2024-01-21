import numpy as np
from scipy.spatial.distance import cdist

from problem import instance


class CVRP:
    """This class has all the data for an instance of the Capacitated
    Vehicle Routing Problem, (but does not solve it)"""

    def __init__(self, filepath: str):
        data = instance.load(filepath)
        self.K = data["K"]  # number of vehicles
        self.Q = data["Q"]  # capacity of vehicles
        self.N = data["N"]  # number of clients (not considering depot)
        self.demand = data["demand"]  # demand of each node
        coordinates = np.asarray(data["coordinates"])
        self.distance = self._get_distances(coordinates)
        print(f"distances:\n{self.distance}")
        print(f"demand:\n{self.demand}")

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
