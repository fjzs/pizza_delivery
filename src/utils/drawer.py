import os

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path

from problem.cvrp import CVRP
from problem.solution import Solution

from .logger import Log

MIN_NODE_SIZE = 10
MAX_NODE_SIZE = 200
DEPOT_SIZE = 100
ZFILL_DIGITS = 3
FIGSIZE = (12, 6)

# TODO: https://matplotlib.org/stable/gallery/animation/unchained.html#sphx-glr-gallery-animation-unchained-py


class Drawer:

    def __init__(self, folder: str, instance: CVRP):
        self.instance = instance
        self.folder = folder

    def _save_figure(self, iteration: int):
        """Saves the figure in the specified folder

        Args:
            iteration (int):
        """
        filepath = os.path.join(self.folder, "current") + ".png"
        plt.savefig(filepath, bbox_inches="tight", pad_inches=0.1)
        filepath = (
            os.path.join(self.folder, str(iteration).zfill(ZFILL_DIGITS)) + ".png"
        )
        plt.savefig(filepath, bbox_inches="tight", pad_inches=0.1)

    def draw_of_and_solution(self, solution: Solution, log: Log):
        """Draws both the objective function evolution and the solution

        Args:
            solution (Solution):
            log (Log):
        """
        fig, axs = plt.subplots(1, 2, layout="constrained", figsize=FIGSIZE)
        iteration = log.get_last_iteration()
        self._set_of_plot(log, axs[0], iteration)
        self._set_solution_plot(solution, axs[1], iteration)
        self._save_figure(iteration)
        fig.clear()
        plt.close()

    def _set_solution_plot(self, solution: Solution, ax, iteration: int):

        # Plot the nodes proportional to its demand
        min_demand = min(self.instance.demand[1:])
        max_demand = max(self.instance.demand[1:])
        range_demand = max_demand - min_demand + 1

        # Plot nodes ids
        pos = self.instance.coordinates
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

        ax.set_title(
            f"Iteration #{str(iteration).zfill(ZFILL_DIGITS)}, cost = {round(solution.get_cost(), 1)}"
        )

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        return ax

    def _set_of_plot(self, log: Log, ax, last_iteration: int):
        best_know_of_value = self.instance.best_known_value
        iterations = range(1, last_iteration + 1)
        of_integer_values = [x["of_integer_optimal_value"] for x in log.data]

        # Plot the objective function integer values
        ax.plot(
            iterations,
            of_integer_values,
            label="Current Value",
            color="red",
            marker="o",
        )

        # Plot the best known solution if it is known
        if best_know_of_value:
            ax.plot(
                iterations,
                [best_know_of_value for i in iterations],
                label="Best Known Value",
                color="black",
                linestyle="dashed",
            )
        ax.set_title("Objective Function value per iteration")
        ax.legend(loc="upper right")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("MIP Objective Function Value")
        return ax

    def draw_solution(self, solution: Solution, filename: str):
        """Draw a solution

        Args:
            solution (Solution): the solution to plot
            filename (str): (Optional name), for ex: "heuristic"
            folder_to_save (str): The folder to save all the plots
        """
        assert filename is not None

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

        # if save_iteration:
        #     filename = str(self.iteration).zfill(ZFILL_DIGITS)
        #     self.iteration += 1
        #     filepath = os.path.join(self.folder, filename) + ".png"
        #     plt.title(f"Iteration #{filename}, cost = {round(solution.get_cost(), 1)}")
        #     plt.savefig(filepath, bbox_inches="tight", pad_inches=0.1)
        #     filepath = os.path.join(self.folder, "current") + ".png"
        #     plt.savefig(filepath, bbox_inches="tight", pad_inches=0.1)  # current.png
        #     print(f"Saved {filepath}")

        fig.clear()
        plt.close()
