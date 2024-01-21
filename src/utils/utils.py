import json
import os


def save_dictionary(data: dict, file_name: str, folder: str) -> None:
    """Saves the dict as a json file

    Args:
        data (dict):
    """
    filepath = os.path.join(folder, file_name + ".json")

    with open(filepath, "w") as outfile:
        json.dump(data, outfile, sort_keys=True, indent=4)
        print(f"\n{file_name} saved in {filepath}")


def load_file(filepath: str) -> dict:
    """Loads a file

    Args:
        filepath (str):

    Returns:
        dict:
    """
    data = dict()
    with open(filepath) as f:
        data = json.load(f)
    return data
