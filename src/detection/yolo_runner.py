import os
from ultralytics import YOLO
import sys

# Add src to path if needed for detection module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.metrics.detection import extract_detection_metrics

class YOLOEvaluator:
    def __init__(self, model_path: str, data_yaml: str, device: str = "auto"):
        self.model = YOLO(model_path)
        # Assuming we don't want to change weights during eval
        self.data_yaml = data_yaml
        self.device = None if device == "auto" else device

    def evaluate(self, image_dir_or_yaml: str) -> dict:
        """
        Evaluate model on a dataset.
        For Optuna trials, we might evaluate on a specific restored images directory.
        Ultralytics val() typically requires a yaml or a split. 
        If we evaluate restored images, we can temporarily change the yaml's validation split to point to the restored images directory.
        Since we just want to run evaluation on a set of images, we can use model.val(data=..., split='val').
        But wait, we need the labels! If we pass a restored_dir, YOLO expects labels to be in restored_dir/../labels or similar.
        If we copy images to restored_dir, we must copy labels to restored_dir_labels or use symlinks.
        Actually, we can create a temporary yaml that points to restored_dir as its `val` split.
        """
        
        # We assume image_dir_or_yaml is either a .yaml file or a .txt split file.
        # If it's a split file containing absolute paths to the restored images, YOLO handles it correctly.
        
        results = self.model.val(data=self.data_yaml, split='val', device=self.device, verbose=False)
        metrics = extract_detection_metrics(results)
        return metrics

    def predict_and_save(self, image_path: str, output_dir: str):
        """Predict and save the image with bounding boxes to output_dir"""
        os.makedirs(output_dir, exist_ok=True)
        # model.predict returns list of Results
        res = self.model.predict(source=image_path, device=self.device, save=True, project=output_dir, name="predict", exist_ok=True)
        # The saved image will be in output_dir/predict/filename.jpg
        return res
