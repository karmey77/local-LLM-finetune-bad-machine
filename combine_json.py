import json
import random

# Load the JSON files
file_paths = [
    "./princess.json",
    "./ruozhiba_qa.json",
    "./identity.json",
    "./TMMLU.json",
]

data = []

# Read and duplicate the JSON data
for index, file_path in enumerate(file_paths):
    with open(file_path, "r", encoding="utf-8") as file:
        file_data = json.load(file)
        if index == 0:
            data.extend(file_data * 5000)  # Duplicate the first file's data 10 times
        elif index == 1:
            data.extend(file_data * 3000)  # Duplicate the second file's data 5 times
        elif index == 2:
            data.extend(file_data * 6000)
        elif index == 3:
            data.extend(file_data * 0)

# Shuffle the combined data
random.shuffle(data)

# Output the combined and shuffled data
final_json_path = "./data/for_fine_tune.json"
with open(final_json_path, "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

final_json_path

# Load the JSON files
file_paths = [
    "./data/final_for_fine_tune.json"
]

data = []

# Read and validate the JSON data
for file_path in final_json_path:
    with open(file_path, "r", encoding="utf-8") as file:
        file_data = json.load(file)
        for entry in file_data:
            if not isinstance(entry["output"], str):
                entry["output"] = str(entry["output"])
        data.extend(file_data)

# Output the validated data
validated_json_path = "./data/validated_for_fine_tune.json"
with open(validated_json_path, "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

validated_json_path