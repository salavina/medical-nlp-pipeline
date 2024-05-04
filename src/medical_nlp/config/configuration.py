from medical_nlp.constants import *
from medical_nlp.utils.common import read_yaml, create_directories
from medical_nlp.entity.config_entity import (DataIngestionConfig)

# import os
# from dotenv import load_dotenv

# load_dotenv()

# MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
# MLFLOW_TRACKING_USERNAME = os.environ["MLFLOW_TRACKING_USERNAME"]
# MLFLOW_TRACKING_PASSWORD = os.environ["MLFLOW_TRACKING_PASSWORD"]

class configurationManager:
    def __init__(self, config_file_path = CONFIG_FILE_PATH,
                 params_file_path = PARAMS_FILE_PATH):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL
        )
        
        return data_ingestion_config