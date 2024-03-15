import os

from problem.cvrp import CVRP
from problem.instance import create, generate_from_vrp_file
from solver.column_generation.method import SolverColumnGeneration

FOLDER_INSTANCES = ".\\src\\instances\\"

if __name__ == "__main__":

    instance = "A-n33-k6"

    # # Generate the .json from the original .vrp file
    # generate_from_vrp_file(folder=FOLDER_INSTANCES, instance_name=instance)

    # Solve
    instance_folder = os.path.join(FOLDER_INSTANCES, instance)
    filepath = os.path.join(instance_folder, instance) + ".json"
    data = CVRP(filepath)
    solver = SolverColumnGeneration(
        instance=data,
        folder=instance_folder,
        cg_max_iterations=100,
        improvement_iterations=0,
        heuristic="cw",
    )
