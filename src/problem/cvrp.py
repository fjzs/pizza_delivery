from typing import Dict, List

import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path
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
        self.coordinates = np.asarray(data["coordinates"])
        self.distance = self._get_distances(self.coordinates)
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

    def draw(self, routes: Dict[int, List[int]], save_file=False, file_name=None):
        fig, ax = plt.subplots()
        pos = self.coordinates

        # Plot nodes ids
        for i in range(self.N + 1):
            x, y = self.coordinates[i, :]
            color = "red" if i == 0 else "blue"
            ax.scatter(x, y, s=200, color=color)
            plt.annotate(str(i), (x - 0.3, y - 0.35), color="white", size=10)

        # Plot the routes
        for route in routes.values():
            verts = [pos[n] for n in route]
            path = Path(verts)
            patch = patches.PathPatch(path, facecolor="none", lw=1, zorder=0)
            ax.add_patch(patch)

        if save_file:
            file_name = (
                file_name
                if file_name is not None
                else f"{len(pos.keys())}_nodes_{len(routes)}_routes"
            )
            plt.savefig(
                f"{file_name}.pdf",
                bbox_inches="tight",
                transparent=True,
                pad_inches=0.1,
            )

        plt.show()
