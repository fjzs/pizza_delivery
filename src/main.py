from problem import instance

FOLDER_INSTANCES = "instances"


if __name__ == "__main__":
    instance.create(
        num_clients=10, num_vehicles=5, capacity=10, name="02", folder=FOLDER_INSTANCES
    )
