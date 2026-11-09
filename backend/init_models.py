from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from app.core.config import settings


def load_models():
    motorcycle_model = settings.motorcycle_model
    motorcycle_detection_model = YOLO(motorcycle_model)

    plate_model_path = hf_hub_download(
        repo_id="ishaqinu/yolo26n-license-plate-detection",
        filename="alpr_26n.pt"
    )
    plate_detection_model = YOLO(plate_model_path)


if __name__ == "__main__":
    load_models()
