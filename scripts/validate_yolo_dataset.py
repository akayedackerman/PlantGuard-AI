from pathlib import Path

for split in ["train", "val"]:

    img_dir = Path(f"data/yolo_seg/{split}/images")
    lbl_dir = Path(f"data/yolo_seg/{split}/labels")

    imgs = {p.stem for p in img_dir.glob("*.jpg")}
    lbls = {p.stem for p in lbl_dir.glob("*.txt")}

    print(f"\n=== {split.upper()} ===")

    print("Images:", len(imgs))
    print("Labels:", len(lbls))

    missing_images = lbls - imgs
    missing_labels = imgs - lbls

    print("Labels without images:", len(missing_images))
    print("Images without labels:", len(missing_labels))

    if missing_images:
        print("\nExamples:")
        print(list(missing_images)[:10])
