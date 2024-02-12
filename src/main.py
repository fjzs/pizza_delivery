import os

from problem.cvrp import CVRP
from problem.instance import create, generate_from_vrp_file
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

    # # Generate the .json from a .vrp file
    # generate_from_vrp_file(folder=FOLDER_INSTANCES, instance_name="CMT1")

    # Solve
    instance = "demand_test"
    instance_folder = os.path.join(FOLDER_INSTANCES, instance)
    filepath = os.path.join(instance_folder, instance) + ".json"
    data = CVRP(filepath)
    solver = SolverColumnGeneration(
        instance=data, folder=instance_folder, max_iterations=100
    )
