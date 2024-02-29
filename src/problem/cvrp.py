import os
from typing import Dict, List

import matplotlib.colors as mcolors
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
        assert r is not None
        assert len(r.nodes) >= 3  # depot -> client -> depot is the shortest
        assert r.nodes[0] == 0  # start is the depot
        assert r.nodes[-1] == 0  # end is the depot
        demand_covered = sum([self.demand[i] for i in r.clients])
        assert demand_covered <= self.q  # constraint capacity
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
        self, routes: Dict[int, Route], title: str, filename: str, folder_to_save: str
    ):
        """Draw a set of routes

        Args:
            routes (Dict[int, Route]):
            title (str):
            filename (str):
            folder_to_save (str):
        """
        fig, ax = plt.subplots()
        pos = self.coordinates
        plt.title(title)

        # Plot the nodes proportional to its demand
        min_node_size = 5
        max_node_size = 200
        min_demand = min(self.demand[1:])
        max_demand = max(self.demand[1:])
        range_demand = max_demand - min_demand + 1

        # Plot nodes ids
        for i in range(self.n):
            x, y = self.coordinates[i, :]
            color = "black"
            marker = "s" if i == 0 else "o"
            size = (
                100
                if i == 0
                else min_node_size
                + max_node_size * (self.demand[i] - min_demand) / range_demand
            )
            ax.scatter(x, y, s=size, color=color, marker=marker)
            # y_displacement = -0.5
            # x_displacement = -0.4 if i < 10 else -0.6
            # plt.annotate(
            #     str(i), (x + x_displacement, y + y_displacement), color="white", size=10
            # )  # id

        # Plot the routes
        colormap = plt.cm.viridis
        route_ids = routes.keys()
        norm = mcolors.Normalize(vmin=min(route_ids), vmax=max(route_ids))
        colors = {
            id: colormap(norm(id)) for id in route_ids
        }  # Get a unique color for each route_id

        for id, route in routes.items():
            if len(route.nodes) > 0:
                col = colors[id]
                verts = [pos[n] for n in route.nodes]
                path = Path(verts)
                patch = patches.PathPatch(
                    path, facecolor="none", lw=1, zorder=0, edgecolor=col
                )
                ax.add_patch(patch)

        # Save the fig
        filepath = os.path.join(folder_to_save, filename) + ".png"
        plt.savefig(filepath, bbox_inches="tight", pad_inches=0.1)
        fig.clear()
        plt.clf()
        print(f"Saved {filepath}")
