from problem.cvrp import CVRP
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
        self.initial_solution: Solution = get_initial_solution(instance)
        # instance.draw(self.initial_solution)

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

        # COnsider Johnson or Bellman Ford for negative edges
