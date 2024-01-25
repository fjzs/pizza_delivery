from problem.cvrp import CVRP, Route
from problem.solution import Solution

from .initial_solution import get_initial_solution
from .master_problem import MasterProblem
from .pricing_problem import PricingProblem


class SolverColumnGeneration:
    """This class has the responsibility of solving the CVRP
    with a column generation approach
    """

    def __init__(self, instance: CVRP, max_iterations: int = 10):
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
        # instance.draw(
        #     self.initial_solution.routes,
        #     title="Bad Initial Solution",
        #     save_file=True,
        #     file_name="bad",
        # )

        # Create the master problem and fill it with the initial solution
        self.master = MasterProblem(self.initial_solution)

        # Create the pricing problem
        self.pricing = PricingProblem(
            distances=self.instance.distance,
            capacity=self.instance.Q,
            demand=self.instance.demand,
        )

        for i in range(max_iterations):
            print(f"It: {i+1}")

            # Get the duals
            self.master.build_model(linear=True)
            self.master.solve()
            duals = self.master.get_duals()
            self.pricing.set_duals(duals)

            # Find a negative reduced-cost path
            reduced_cost, path = self.pricing.solve()
            print("Pricing problem:")
            print(f"\tmin reduced-cost: {reduced_cost}")
            print(f"\tbest path: {path}")

            if reduced_cost < 0:
                # add route to the master problem
                path[-1] = 0  # replace the last virtual node with the actual depot
                route = Route(path)
                self.instance.is_valid_route(route)
                cost = self.instance.get_route_cost(route)
                self.master.add_route(route, cost)
            else:
                break

        print(f"Now solving the MIP Master:")
        self.master.build_model(linear=False)
        self.master.solve()
        final_solution: Solution = self.master.get_solution()
        instance.draw(
            self.initial_solution.routes,
            title=f"Final Solution Z={final_solution.total_cost}",
            save_file=True,
            file_name="final",
        )
