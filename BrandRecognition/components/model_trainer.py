import os, sys
import yaml
import shutil
from ultralytics import YOLO
from BrandRecognition.utils.main_utils import read_yaml_file
from BrandRecognition.logger import logging
from BrandRecognition.exception import BrandException
from BrandRecognition.entity.config_entity import ModelTrainerConfig
from BrandRecognition.entity.artifact_entity import ModelTrainerArtifact


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(
        self,
    ) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Unzipping data")
            os.system("unzip dataset.zip")
            shutil.unpack_archive("dataset.zip", ".")

            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            print(model_config_file_name)

            os.system(
                "yolo task=detect mode=train epochs=10 data=/content/brand/data.yml model=yolov8l.pt imgsz=640 batch=8 patience=0"
            )
            os.system("cp runs/detect/train/weights/best.pt custom_model_weights/")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            os.system(
                f"cp runs/detect/train/weights/best.pt {self.model_trainer_config.model_trainer_dir}/"
            )
            os.system("cp yolov8n.pt custom_model_weights/")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            os.system(
                f"cp runs/detect/train/weights/best.pt {self.model_trainer_config.model_trainer_dir}/"
            )

            os.system("rm -rf runs")
            os.system("rm -rf train")
            os.system("rm -rf val")
            os.system("rm -rf data.yml")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="custom_model_weights/best.pt",
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise BrandException(e, sys)
