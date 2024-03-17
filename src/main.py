import os

from problem.cvrp import CVRP
from problem.instance import create, generate_from_vrp_file
from solver.column_generation.method import SolverColumnGeneration

FOLDER_INSTANCES = ".\\src\\instances\\"

if __name__ == "__main__":

    instance_name = "CMT1"

    # # Generate the .json from the original .vrp file
    # generate_from_vrp_file(folder=FOLDER_INSTANCES, instance_name=instance_name)

    # Solve
    instance_folder = os.path.join(FOLDER_INSTANCES, instance_name)
    filepath = os.path.join(instance_folder, instance_name) + ".json"
    instance = CVRP(filepath)
    solver = SolverColumnGeneration(
        instance=instance,
        folder=instance_folder,
        cg_max_iterations=0,
        improvement_iterations=100,
        heuristic="cw",
    )
