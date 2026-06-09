import json
from pathlib import Path

INPUT_JSON = "data/plantseg/json/train"

json_files = list(Path(INPUT_JSON).glob("*.json"))

all_labels = set()

for json_file in json_files:
    with open(json_file, "r") as f:
        data = json.load(f)

    for shape in data["shapes"]:
        all_labels.add(shape["label"].strip().lower())

print(f"Found {len(all_labels)} classes")

for i, cls in enumerate(sorted(all_labels)):
    print(i, cls)
