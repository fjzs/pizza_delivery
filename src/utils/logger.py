import os

import matplotlib.pyplot as plt

from . import utils


class Log:
    def __init__(self):
        self.data = []

    def get_last_iteration(self) -> int:
        if len(self.data) > 0:
            return self.data[-1]["iteration"]
        else:
            return 0

    def add(
        self,
        of_linear_lower_bound: float,
        of_integer_optimal_value: float,
        number_routes: int,
        min_reduced_cost: float,
    ):
        iteration = len(self.data) + 1
        row = dict()
        row["iteration"] = iteration
        row["of_linear_lower_bound"] = of_linear_lower_bound
        row["of_integer_optimal_value"] = of_integer_optimal_value
        row["number_routes"] = number_routes
        row["min_reduced_cost"] = min_reduced_cost
        self.data.append(row)

    def save(self, folder: str):
        utils.save_data(self.data, "log", folder)

    def plot(self, folder: str):
        self._plot_of(folder)

    def _plot_of(self, folder: str):
        iterations = self.data.keys()
        of_linear_lower_bound = [
            self.data[k]["of_linear_lower_bound"] for k, v in self.data.items()
        ]
        of_integer_optimal_value = [
            self.data[k]["of_integer_optimal_value"] for k, v in self.data.items()
        ]
        plt.plot(
            iterations,
            of_linear_lower_bound,
            label="Linear Lower Bound",
            color="blue",
            marker="o",
        )
        plt.plot(
            iterations,
            of_integer_optimal_value,
            label="MIP Value",
            color="red",
            marker="o",
        )
        plt.title("Objective Function value per iteration")
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function")
        plt.savefig(os.path.join(folder, "OF.png"), bbox_inches="tight", pad_inches=0.1)
        plt.clf()
