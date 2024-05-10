# End to End NLP Classification Pipeline with Azure Cloud Deployment

This is a complete pipeline that streamlines various steps including data ingestion, model preparation, model training, experiment tracking via MLFlow, data version control (DVC), app development via flask, containarization via Docker, Azure app deplyment & CI/CD via Github Actions.

## 📋 Table of Contents

- [🧩 End to End NLP Classification Pipeline with Azure Cloud Deployment](#🧩-end-to-end-nlp-classification-pipeline-with-azure-cloud-deployment)
  - [📋 Table of Contents](#📋-table-of-contents)
  - [🧑‍⚕️ Workflows](#🧑‍⚕️-workflows)
  - [📖 Usage](#📖-usage)
    - [Downloading the dataset](#downloading-the-dataset)
    - [Dataset structure](#dataset-structure)
    - [App Running](#app-running)
    - [Training](#training)
    - [Experiment Tracking via MLFlow](#experiment-tracking-via-mlflow)
    - [Azure CI/CD](#azure-cicd)
      - [Creating/Pushing Docker Image](#creatingpushing-docker-image)
      - [Deployment Steps](#deployment-steps)
  - [✍️ Contributing](#️✍️-contributing)

## 🧑‍⚕️ Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional] - only if you have API keys
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml
10. app.py

## 📖 Usage

### Downloading the dataset

The California Independent Medical Review (CIMR) can be downloaded from [Kaggle](https://www.kaggle.com/datasets/prasad22/ca-independent-medical-review) or authomatically via the pipeline.

### Dataset structure

The dataset is provided as a CSV file. Only two columns named `Findings` and `Type` are used for training/inference purposes where `Type` is the label column with 3 labels: `Medical Necessity`, `Experimental/Investigational`, `Urgent Care`. The splits are sized as follows:

| Split        | # Row % |
| :----------- | :-----: |
| `train`      |   80    |
| `validation` |   20    |

### App Running

Clone the repository

```bash
https://github.com/salavina/medical-nlp-pipeline
```

#### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medical python=3.9 -y
```

```bash
conda activate medical
```

#### STEP 02- install the requirements

```bash
pip install -r requirements.txt
```

```bash
# Finally run the following command
python app.py
```

### Training

1. `dvc init`
2. `dvc repro`
3. `dvc dag`

### Experiment Tracking via MLFlow

- [Documentation](https://mlflow.org/docs/latest/index.html)

- MLFlow is hosted via [dagshub](https://dagshub.com/)

- You need to add your `MLFLOW_TRACKING_URI` to a `.env` file within root directory.

### Azure CI/CD

#### Creating/Pushing Docker Image

Run the following from terminal:

```bash
docker build -t medical.azurecr.io/medical:latest .

docker login medical.azurecr.io

docker push medical.azurecr.io/medical:latest
```

#### Deployment Steps

1. Build the Docker image of the Source Code
2. Push the Docker image to Container Registry
3. Launch the Web App Server in Azure
4. Pull the Docker image from the container registry to Web App server and run

## ✍️ Contributing

I welcome contributions to this repository (noticed a typo? a bug?). To propose a change:

```
git clone https://github.com/salavina/medical-nlp-pipeline
cd medical-nlp-pipeline
git checkout -b my-branch
pip install -r requirements.txt
pip install -r dev-requirements.txt
```

Once your changes are made, make sure to lint and format the code (addressing any warnings or errors):

```
isort .
black .
flake8 .
```

Then, submit your change as a pull request.
