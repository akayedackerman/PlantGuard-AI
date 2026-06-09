from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolo11n-seg.pt")

    model.train(
        data="data/yolo_seg/data.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        workers=4,
        cache=False,
        project="runs",
        name="plantseg_v1"
    )
