# 캐글 api 파일 저장
import os   
from kaggle.api.kaggle_api_extended import KaggleApi

class KaggleDownloader:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(KaggleDownloader, cls).__new__(cls)
        return cls._instance

    def setup(self, key: str):
        
        home_folder = os.getenv('HOME')
        kaggle_config_folder = os.path.join(home_folder, '.kaggle')
        kaggle_api_file_path = os.path.join(kaggle_config_folder, 'kaggle.json')
        os.makedirs(kaggle_config_folder, exist_ok=True)
        with open(kaggle_api_file_path, 'w') as f:
            f.write(key)
        self.api = KaggleApi()
        self.api.authenticate()
        
    def download(self, dataset, unzip=True):
        dataset_name = dataset.split('/')[-1]
        self.api.dataset_download_files(dataset, './datasets/' + dataset_name, unzip=unzip)
        
