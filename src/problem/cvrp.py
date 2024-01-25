from typing import Dict, List

import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path
from scipy.spatial.distance import cdist

from problem import instance
from problem.route import Route


class CVRP:
    """This class has all the data for an instance of the Capacitated
    Vehicle Routing Problem, (but does not solve it)"""

    def __init__(self, filepath: str):
        data = instance.load(filepath)
        self.K: int = data["K"]  # number of vehicles
        self.Q: int = data["Q"]  # capacity of vehicles
        self.N: int = data["N"]  # number of clients (not considering depot)
        self.demand: List[int] = data["demand"]  # demand of each node
        self.coordinates: np.ndarray = np.asarray(data["coordinates"])
        self.distance: np.ndarray = self._get_distances(self.coordinates)
        print(f"distances:\n{self.distance}")
        print(f"demand:\n{self.demand}")

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
        assert r is not None
        assert len(r.nodes) >= 3  # depot -> client -> depot is the shortest
        assert r.nodes[0] == 0  # start is the depot
        assert r.nodes[-1] == 0  # end is the depot
        demand_covered = sum([self.demand[i] for i in r.clients])
        assert demand_covered <= self.Q  # constraint capacity
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

    def draw(
        self, routes: Dict[int, Route], title: str, save_file=False, file_name=None
    ):
        fig, ax = plt.subplots()
        pos = self.coordinates

        plt.title(title)

        # Plot nodes ids
        for i in range(self.N + 1):
            x, y = self.coordinates[i, :]
            color = "red" if i == 0 else "blue"
            ax.scatter(x, y, s=200, color=color)
            plt.annotate(str(i), (x - 0.3, y - 0.35), color="white", size=10)

        # Plot the routes
        for route in routes.values():
            if len(route.nodes) > 0:
                verts = [pos[n] for n in route.nodes]
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
                f"{file_name}.png",
                bbox_inches="tight",
                pad_inches=0.1,
            )

        plt.show()
