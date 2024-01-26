from problem.cvrp import CVRP, Route
from problem.solution import Solution

from .initial_solution import get_initial_solution
from .master_problem import MasterProblem
from .pricing_problem import PricingProblem


class SolverColumnGeneration:
    """This class has the responsibility of solving the CVRP
    with a column generation approach
    """

    def __init__(
        self,
        instance: CVRP,
        folder: str,
        max_iterations: int = 10,
    ):
        """Initializes the solver

        Args:
            instance (CVRP): data for the problem
            folder (str): folder to save results
            max_iterations (int, optional): Max iterations to run column generation
        """
        self.instance: CVRP = instance
        self.initial_solution: Solution = None

        # heuristic solution
        self.initial_solution = get_initial_solution(instance)
        instance.draw(
            self.initial_solution.routes,
            title=f"Heuristic Cost={round(self.initial_solution.total_cost)}",
            filename="heuristic",
            folder_to_save=folder,
        )

        # bad solution
        # r1 = Route([0, 2, 4, 0])
        # r2 = Route([0, 5, 3, 1, 0])
        # self.initial_solution = Solution(instance, {1: r1, 2: r2})
        # instance.draw(
        #     self.initial_solution.routes,
        #     title=f"Bad Initial Solution Z={self.initial_solution.total_cost}",
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
            print(f"\nMASTER ITERATION: {i+1} -------------------\n")

            # Get the duals
            self.master.build_model(is_linear=True)
            self.master.solve()
            client_duals, vehicle_cap_dual = self.master.get_duals()
            self.pricing.set_duals(client_duals, vehicle_cap_dual)

            # Find a negative reduced-cost path
            cost_path_solutions = self.pricing.solve()
            if len(cost_path_solutions) > 0:
                print("\n Results of Pricing Problem:")
                for i, (reduced_cost, path) in enumerate(cost_path_solutions):
                    print(f"\t {i+1}: red-cost: {reduced_cost} path: {path}")
                    # add route to the master problem
                    path[-1] = 0  # replace the last virtual node with the actual depot
                    route = Route(path)
                    self.instance.is_valid_route(route)
                    cost = self.instance.get_route_cost(route)
                    self.master.add_route(route, cost)
            else:
                break

        print(f"\nCOLUMN GENERATION ENDED!!!")

        print(f"Now solving the MIP Master:")
        self.master.build_model(is_linear=False)
        self.master.solve()
        final_solution: Solution = self.master.get_solution()
        instance.draw(
            routes=final_solution.routes,
            title=f"Final Solution Z={final_solution.total_cost}",
            filename="final",
            folder_to_save=folder,
        )
