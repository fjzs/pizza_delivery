import os

from problem.cvrp import CVRP
from problem.instance import create
from solver.column_generation.method import SolverColumnGeneration

FOLDER_INSTANCES = ".\\src\\instances\\"


if __name__ == "__main__":
    instance = "02.json"
    filepath = os.path.join(FOLDER_INSTANCES, instance)
    data = CVRP(filepath)
    solver = SolverColumnGeneration(data)
