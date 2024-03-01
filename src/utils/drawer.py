import os

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path

from problem.cvrp import CVRP
from problem.solution import Solution

MIN_NODE_SIZE = 10
MAX_NODE_SIZE = 200
DEPOT_SIZE = 100
ZFILL_DIGITS = 3


class Drawer:

    def __init__(self, folder: str, instance: CVRP):
        self.instance = instance
        self.iteration = 1
        self.folder = folder

    def draw_solution(
        self,
        solution: Solution,
        filename: str,
        save_iteration: bool = False,
    ):
        """Draw a solution

        Args:
            solution (Solution): the solution to plot
            filename (str): (Optional name), for ex: "heuristic"
            folder_to_save (str): The folder to save all the plots
            save_iteration (bool): True to save the plot with the iteration evolution
        """
        assert (filename is not None) or save_iteration

        fig, ax = plt.subplots()
        pos = self.instance.coordinates

        # Plot the nodes proportional to its demand
        min_demand = min(self.instance.demand[1:])
        max_demand = max(self.instance.demand[1:])
        range_demand = max_demand - min_demand + 1

        # Plot nodes ids
        for i in range(self.instance.n):
            x, y = self.instance.coordinates[i, :]
            color = "black"
            marker = "s" if i == 0 else "o"
            size = (
                DEPOT_SIZE
                if i == 0
                else MIN_NODE_SIZE
                + MAX_NODE_SIZE * (self.instance.demand[i] - min_demand) / range_demand
            )
            ax.scatter(x, y, s=size, color=color, marker=marker)
            # y_displacement = -0.5
            # x_displacement = -0.4 if i < 10 else -0.6
            # plt.annotate(
            #     str(i), (x + x_displacement, y + y_displacement), color="white", size=10
            # )  # id

        # Plot the routes
        colormap = plt.cm.viridis
        route_ids = solution.routes.keys()
        norm = mcolors.Normalize(vmin=min(route_ids), vmax=max(route_ids))
        colors = {
            id: colormap(norm(id)) for id in route_ids
        }  # Get a unique color for each route_id

        for id, route in solution.routes.items():
            if len(route.nodes) > 0:
                col = colors[id]
                verts = [pos[n] for n in route.nodes]
                path = Path(verts)
                patch = patches.PathPatch(
                    path, facecolor="none", lw=1, zorder=0, edgecolor=col
                )
                ax.add_patch(patch)

        # Save the fig
        if filename:
            filepath = os.path.join(self.folder, filename) + ".png"
            plt.title(f"{filename}, cost = {round(solution.get_cost(), 1)}")
            plt.savefig(filepath, bbox_inches="tight", pad_inches=0.1)
            print(f"Saved {filepath}")

        if save_iteration:
            filename = str(self.iteration).zfill(ZFILL_DIGITS)
            self.iteration += 1
            filepath = os.path.join(self.folder, filename) + ".png"
            plt.title(f"Iteration #{filename}, cost = {round(solution.get_cost(), 1)}")
            plt.savefig(filepath, bbox_inches="tight", pad_inches=0.1)
            filepath = os.path.join(self.folder, "current") + ".png"
            plt.savefig(filepath, bbox_inches="tight", pad_inches=0.1)  # current.png
            print(f"Saved {filepath}")

        fig.clear()
        plt.close()
