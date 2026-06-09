import json
import shutil
import re
from pathlib import Path

import pandas as pd

INPUT_ROOT = Path("data/plantseg")
OUTPUT_ROOT = Path("data/yolo_seg")


# =====================================================
# Label Normalization
# =====================================================

def normalize_label(label):
    label = str(label).strip().lower()

    label = re.sub(
        r"\b(google|bing|baidu)\b",
        "",
        label
    )

    label = re.sub(r"\s+", " ", label)

    return label.strip()


# =====================================================
# Load metadata and build class mapping
# =====================================================

df = pd.read_csv(INPUT_ROOT / "Metadata.csv")

classes = sorted(
    {
        normalize_label(d)
        for d in df["Disease"].unique()
    }
)

class_map = {
    cls: idx
    for idx, cls in enumerate(classes)
}

print(f"Found {len(classes)} classes")


# =====================================================
# Create output folders
# =====================================================

for split in ["train", "val"]:

    (OUTPUT_ROOT / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_ROOT / split / "labels").mkdir(
        parents=True,
        exist_ok=True
    )


# =====================================================
# Statistics
# =====================================================

stats = {
    "images": 0,
    "labels": 0,
    "polygons": 0,
    "bad_polygons": 0,
    "skipped": 0,
}


# =====================================================
# Convert LabelMe -> YOLO Seg
# =====================================================

def convert_split(src_split, dst_split):

    json_dir = INPUT_ROOT / "json" / src_split
    img_dir = INPUT_ROOT / "images" / src_split

    for json_file in json_dir.glob("*.json"):

        image_file = img_dir / f"{json_file.stem}.jpg"

        # Skip samples with missing images
        if not image_file.exists():
            stats["skipped"] += 1
            continue

        with open(json_file, "r") as f:
            data = json.load(f)

        width = data["imageWidth"]
        height = data["imageHeight"]

        yolo_lines = []

        for shape in data["shapes"]:

            label = normalize_label(shape["label"])

            if label not in class_map:
                continue

            class_id = class_map[label]

            polygon = []

            for x, y in shape["points"]:

                polygon.append(f"{x / width:.6f}")
                polygon.append(f"{y / height:.6f}")

            # -------------------------------------------------
            # YOLO segmentation validation
            # -------------------------------------------------

            # Must have x,y pairs
            if len(polygon) % 2 != 0:
                stats["bad_polygons"] += 1
                continue

            # Minimum 3 points = 6 coordinates
            if len(polygon) < 6:
                stats["bad_polygons"] += 1
                continue

            line = f"{class_id} " + " ".join(polygon)

            yolo_lines.append(line)

            stats["polygons"] += 1

        # Skip image if no valid polygons remain
        if not yolo_lines:
            continue

        label_file = (
            OUTPUT_ROOT
            / dst_split
            / "labels"
            / f"{json_file.stem}.txt"
        )

        with open(label_file, "w") as f:
            f.write("\n".join(yolo_lines))

        shutil.copy2(
            image_file,
            OUTPUT_ROOT
            / dst_split
            / "images"
            / image_file.name
        )

        stats["images"] += 1
        stats["labels"] += 1


# =====================================================
# Run Conversion
# =====================================================

print("Converting training set...")
convert_split("train", "train")

print("Converting test set...")
convert_split("test", "val")


# =====================================================
# Generate data.yaml
# =====================================================

yaml_file = OUTPUT_ROOT / "data.yaml"

with open(yaml_file, "w") as f:

    f.write("path: data/yolo_seg\n")
    f.write("train: train/images\n")
    f.write("val: val/images\n\n")

    f.write(f"nc: {len(classes)}\n")
    f.write("names:\n")

    for idx, cls in enumerate(classes):
        f.write(f"  {idx}: {cls}\n")


# =====================================================
# Summary
# =====================================================

print("\nConversion Complete")
print("-" * 40)
print(f"Classes      : {len(classes)}")
print(f"Images       : {stats['images']}")
print(f"Labels       : {stats['labels']}")
print(f"Polygons     : {stats['polygons']}")
print(f"Bad polygons : {stats['bad_polygons']}")
print(f"Skipped      : {stats['skipped']}")
print(f"Output       : {OUTPUT_ROOT}")
