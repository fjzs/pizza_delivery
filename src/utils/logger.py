from . import utils
import matplotlib.pyplot as plt
import os

class Log:
    
    def __init__(self):
        self.data = dict()
        
    def add(self, iteration: int, 
            of_linear_lower_bound: float, 
            of_integer_optimal_value: float, 
            number_routes: int, 
            min_reduced_cost: float):
        row = dict()
        row["iteration"] = iteration
        row["of_linear_lower_bound"] = of_linear_lower_bound
        row["of_integer_optimal_value"] = of_integer_optimal_value
        row["number_routes"] = number_routes
        row["min_reduced_cost"] = min_reduced_cost
        self.data[iteration] = row
    
    def save(self, folder: str):
        utils.save_dictionary(self.data, "log", folder)
    
    def plot(self, folder: str):
        self._plot_of(folder)
        
    def _plot_of(self, folder: str):        
        iterations = self.data.keys()
        of_linear_lower_bound = [self.data[k]["of_linear_lower_bound"] for k,v in self.data.items()]
        of_integer_optimal_value = [self.data[k]["of_integer_optimal_value"] for k,v in self.data.items()]
        plt.plot(iterations, of_linear_lower_bound, label="Linear Lower Bound", color="blue")
        plt.plot(iterations, of_integer_optimal_value, label="MIP Value", color="red")
        plt.title("Objective Function value per iteration")
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function")
        plt.savefig(
            os.path.join(folder, "OF.png"),
            bbox_inches="tight",
            pad_inches=0.1)
        plt.show()
        
        
        
        