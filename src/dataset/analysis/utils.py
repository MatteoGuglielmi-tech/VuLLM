import json


def read_json(pth: str) -> dict:
    with open(file=pth, mode="r") as file:
        json_string: str = file.read()

    # Parse JSON data into a dictionary
    json_data: dict = json.loads(json_string)

    return json_data
