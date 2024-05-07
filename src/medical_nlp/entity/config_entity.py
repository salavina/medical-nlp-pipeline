from dataclasses import dataclass
from pathlib import Path

# specifies the type of value related to the key in yaml file
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    nlp_base_model_path: Path
    nlp_updated_base_model_path: Path
    params_classes: int
    

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    nlp_trained_model_path: Path
    nlp_updated_base_model_path: Path
    training_data: Path
    # mlflow_uri: str
    all_params: dict
    params_batch_size: int
    params_epochs: int
    params_learning_rate: float
    params_model_name: str