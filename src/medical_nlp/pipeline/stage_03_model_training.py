from medical_nlp.config.configuration import configurationManager
from medical_nlp.components.model_training import ModelTrainerHF
from medical_nlp import logger




STAGE_NAME = "Model Training"


class ModelTrainingPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        config = configurationManager()
        training_config = config.get_training_config()
        training = ModelTrainerHF(config=training_config)
        logger.info(f"model being trained: {training_config.params_model_name}")
        training.train()
        # turn on & off mlflow tracking here
        # training.log_into_mlflow()




if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e