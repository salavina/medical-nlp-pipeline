import opendatasets as od
from medical_nlp import logger
from medical_nlp.utils.common import get_size
import os
import shutil
import pandas as pd
import glob
from medical_nlp.entity.config_entity import DataIngestionConfig



class DataIngestion():
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self) -> str:
        
        try:
            file_name = 'kaggle.json'
            dataset_url = self.config.source_URL
            # Check if the folder path exists
            files_in_folder = os.listdir(self.config.root_dir)
            
            # Check if the specified file exists in the folder
            if file_name not in files_in_folder:
                shutil.copy(os.path.join('research/', file_name), self.config.root_dir)
            logger.info(f"Downloading data from {dataset_url} to {str(self.config.root_dir)}")
            os.chdir(self.config.root_dir)
            od.download(dataset_url)
            logger.info(f"Downloaded data from {dataset_url} to {str(self.config.root_dir)}")
            os.chdir('../../')
            csv_files = glob.glob(str(self.config.root_dir) + '/' +  dataset_url.split('/')[-1] + '/' + '*.csv')
            base_file = [file for file in csv_files if 'Custom' not in os.path.basename(file)][0]
            df = pd.read_csv(base_file)
            df_customzied = df[['Findings', 'Type']]
            df_customzied.to_csv(str(self.config.root_dir) + '/' +  dataset_url.split('/')[-1] + '/' + os.path.basename(base_file).split('.')[0] + '_Custom.csv' ,index=False)
            logger.info(f"Saved custom dataset to {str(self.config.root_dir) +  dataset_url.split('/')[-1]}")
        
        except Exception as e:
            raise e