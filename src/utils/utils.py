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


def load_file(file_name: str, folder: str) -> dict:
    """Loads a file

    Args:
        file_name (str): does not have extension
        folder (str): folder where it lives

    Returns:
        dict:
    """
    assert "." not in file_name
    data = dict()
    with open(os.path.join(folder, file_name + ".json")) as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    d = load_file("01", "instances")
    print(d)
