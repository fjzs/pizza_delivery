import os

from problem.cvrp import CVRP
from problem.instance import create
from solver.column_generation.method import SolverColumnGeneration

FOLDER_INSTANCES = ".\\src\\instances\\"


if __name__ == "__main__":
    # # Create a new instance
    # create(
    #     num_clients=20,
    #     capacity=4,
    #     name="03",
    #     folder=FOLDER_INSTANCES,
    #     radius=50,
    # )

    # Solve
    instance = "01"
    instance_folder = os.path.join(FOLDER_INSTANCES, instance)
    filepath = os.path.join(instance_folder, instance) + ".json"
    data = CVRP(filepath)
    solver = SolverColumnGeneration(
        instance=data, folder=instance_folder, max_iterations=5
    )
