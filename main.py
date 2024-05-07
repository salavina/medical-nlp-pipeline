from medical_nlp import logger
from medical_nlp.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from medical_nlp.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from medical_nlp.pipeline.stage_03_model_training import ModelTrainingPipeline


STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<<\nx===============x")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Prepare base model"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_preparation = PrepareBaseModelTrainingPipeline()
    model_preparation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_training = ModelTrainingPipeline()
    model_training.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e