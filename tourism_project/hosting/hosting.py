from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="mlops/deployment", 
    repo_id="Jyotidarshan/tour-pckg-pred-space",
    repo_type="space",  
    path_in_repo="",
)
