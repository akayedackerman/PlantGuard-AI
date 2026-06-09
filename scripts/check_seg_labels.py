from pathlib import Path

for split in ["train", "val"]:

    label_dir = Path(f"data/yolo_seg/{split}/labels")

    bad = 0

    for txt in label_dir.glob("*.txt"):

        lines = txt.read_text().strip().splitlines()

        for line in lines:

            parts = line.split()

            if len(parts) < 7:
                print("Too few points:", txt)
                bad += 1
                break

            coords = parts[1:]

            if len(coords) % 2 != 0:
                print("Odd coordinate count:", txt)
                bad += 1
                break

    print(split, "bad labels =", bad)
