from problem.cvrp import CVRP, Route
from problem.solution import Solution

from .initial_solution import get_initial_solution
from .master_problem import MasterProblem
from .pricing_problem import PricingProblem


class SolverColumnGeneration:
    """This class has the responsibility of solving the CVRP
    with a column generation approach
    """

    def __init__(self, instance: CVRP):
        self.instance: CVRP = instance
        self.initial_solution: Solution = None

        # heuristic solution
        # self.initial_solution = get_initial_solution(instance)
        # instance.draw(
        #     self.initial_solution.routes,
        #     title="Heuristic Initial Solution",
        #     save_file=True,
        #     file_name="heuristic",
        # )

        # bad solution
        r1 = Route([0, 2, 4, 0])
        r2 = Route([0, 5, 3, 1, 0])
        self.initial_solution = Solution(instance, {1: r1, 2: r2})
        instance.draw(
            self.initial_solution.routes,
            title="Bad Initial Solution",
            save_file=True,
            file_name="bad",
        )

        # Create the master problem and fill it with the initial solution
        self.master = MasterProblem(self.initial_solution)
        self.master.build_model()
        duals = self.master.get_duals()

        # Create the pricing problem
        self.pricing = PricingProblem(
            distances=self.instance.distance,
            capacity=self.instance.Q,
            demand=self.instance.demand,
        )
        self.pricing.set_duals(duals)
        self.pricing.solve()
