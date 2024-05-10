import os

# from dotenv import load_dotenv

from medical_nlp.constants import *
from medical_nlp.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
)
from medical_nlp.utils.common import create_directories, read_yaml

# load_dotenv()

# MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]


class configurationManager:
    def __init__(
        self, config_file_path=CONFIG_FILE_PATH, params_file_path=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir, source_URL=config.source_URL
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=config.root_dir,
            nlp_base_model_path=config.nlp_base_model_path,
            nlp_updated_base_model_path=config.nlp_updated_base_model_path,
            params_classes=self.params.CLASSES,
        )

        return prepare_base_model_config

    def get_training_config(self) -> TrainingConfig:
        model_training = self.config.model_training
        prepare_base_model = self.config.prepare_base_model
        data_ingestion = self.config.data_ingestion
        training_data = os.path.join(
            self.config.data_ingestion.root_dir,
            data_ingestion.source_URL.split("/")[-1] + "/",
        )

        create_directories([model_training.root_dir])

        training_config = TrainingConfig(
            root_dir=model_training.root_dir,
            nlp_trained_model_path=model_training.nlp_trained_model_path,
            nlp_updated_base_model_path=prepare_base_model.nlp_updated_base_model_path,
            training_data=training_data,
            # mlflow_uri=MLFLOW_TRACKING_URI,
            all_params=self.params,
            params_batch_size=self.params.BATCH_SIZE,
            params_epochs=self.params.EPOCHS,
            params_classes=self.params.CLASSES,
            params_learning_rate=self.params.LEARNING_RATE,
            params_model_name=self.params.MODEL_NAME,
            params_trainer=self.params.TRAINER,
        )

        return training_config
