from problem.cvrp import CVRP, Route
from problem.solution import Solution

from . import initial_solution
from .master_problem import MasterProblem
from .pricing_problem import PricingProblem
from utils.logger import Log

class SolverColumnGeneration:
    """This class has the responsibility of solving the CVRP
    with a column generation approach
    """

    def __init__(
        self,
        instance: CVRP,
        folder: str,
        max_iterations: int,
    ):
        """Initializes the solver

        Args:
            instance (CVRP): data for the problem
            folder (str): folder to save results
            max_iterations (int, optional): Max iterations to run column generation
        """
        self.instance: CVRP = instance
        self.initial_solution: Solution = None
        self.log = Log()
        
        # heuristic solution
        # self.initial_solution = closest_client(instance)
        self.initial_solution = initial_solution.one_route_per_client(instance)
        instance.draw(
            self.initial_solution.routes,
            title=f"Heuristic Cost = {round(self.initial_solution.total_cost, 1)}",
            filename="heuristic",
            folder_to_save=folder,
        )

        # Create the master problem and fill it with the initial solution
        self.master = MasterProblem(self.initial_solution)

        # Create the pricing problem
        self.pricing = PricingProblem(
            distances=self.instance.distance,
            capacity=self.instance.q,
            demand=self.instance.demand,
        )

        last_min_reduced_cost_entered = None
        
        for i in range(max_iterations):
            print(f"\nMASTER ITERATION: {i+1} ---------------------------------------------\n")
            
            # Get the current state to show performance evolution
            self.master.build_model(is_linear=False)
            self.master.solve()
            obj_bound, obj_value = self.master.get_Obj_Values()
            self.log.add(iteration=i,
                         of_linear_lower_bound=obj_bound,
                         of_integer_optimal_value=obj_value,
                         number_routes=len(self.master.routes),
                         min_reduced_cost=last_min_reduced_cost_entered)
            
            # Get the duals
            self.master.build_model(is_linear=True)
            self.master.solve()
            client_duals = self.master.get_duals()
            self.pricing.set_duals(client_duals)

            # Find a negative reduced-cost path
            print(f"\nSolving the pricing problem...")
            cost_path_solutions = self.pricing.solve()
            last_min_reduced_cost_entered = None
            if len(cost_path_solutions) > 0:
                last_min_reduced_cost_entered = min([c for c,p in cost_path_solutions])
                print("\nResults of Pricing Problem:")
                for i, (reduced_cost, path) in enumerate(cost_path_solutions):
                    # add route to the master problem
                    path[-1] = 0  # replace the last virtual node with the actual depot
                    route = Route(path)
                    self.instance.is_valid_route(route)
                    cost = self.instance.get_route_cost(route)
                    self.master.add_route(route, cost)
                    print(f"\t {i+1}: red-cost: {reduced_cost} path: {path}, cost: {cost}")
            else:
                print(f"\nNo reduced-cost routes found! Ending Column Generation...")
                break

        print(f"\n\nCOLUMN GENERATION ENDED!!!")

        print(f"Now solving the MIP Master:")
        self.master.build_model(is_linear=False)
        self.master.solve()
        obj_bound, obj_value = self.master.get_Obj_Values()
        self.log.add(iteration=i+1,
                    of_linear_lower_bound=obj_bound,
                    of_integer_optimal_value=obj_value,
                    number_routes=len(self.master.routes),
                    min_reduced_cost=None)

        # Now save the log
        self.log.save(folder=folder)
        self.log.plot(folder=folder)
        
        final_solution: Solution = self.master.get_solution()
        instance.draw(
            routes=final_solution.routes,
            title=f"Final Cost = {round(final_solution.total_cost, 1)}",
            filename="final",
            folder_to_save=folder,
        )
