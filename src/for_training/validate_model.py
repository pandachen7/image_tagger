from ultralytics import YOLO

source = r"D:\ws\datasets\test_liyu_lake\imgs_test_2020_12以後"


def predict_model(model):
    """有預測到的直接放到對應資料夾內"""
    results = model.predict(
        source=source,
        imgsz=640,
        conf=0.25,
        save=True,
        save_crop=True,
        project="runs/detect",
        name="imgs_test_2020_12",
        exist_ok=True
    )
    # print(results)


def validate_model(model):
    # Validate the model
    results = model.val(data="coco128.yaml")
    print(results)


def benchmark_model(model):
    # 需要灌一對東西, 但還是跑不起來
    # NOTE: 會把torch改成cpu版的, 不要用
    # results = model.benchmark()
    # print(results)
    pass


if __name__ == "__main__":
    model = YOLO(r"D:/ws/models/train_12classes/weights/epoch300.pt")
    predict_model(model)
