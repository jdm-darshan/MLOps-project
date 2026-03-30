
import pandas as pd
import sklearn
import os

from sklearn.model_selection import train_test_split
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Jyotidarshan/tour-pckg-pred-data/tourism.csv"
tourism_data = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
num_features = [
 'Age',
 'DurationOfPitch',
 'NumberOfPersonVisiting',
 'NumberOfFollowups',
 'PreferredPropertyStar', 
 'NumberOfTrips',
 'PitchSatisfactionScore',
 'OwnCar',
 'NumberOfChildrenVisiting',
 'MonthlyIncome'
]

# List of categorical features in the dataset
cat_features = [
 'TypeofContact',
 'CityTier',             #Although numerical but actually categorical
 'Occupation',
 'Gender',
 'ProductPitched',
 'MaritalStatus',
 'Passport',             #Although numerical but actually categorical
 'Designation'
]

# Define predictor matrix (X) using selected numeric and categorical features
X = tourism_data[num_features + cat_features]

# Define target variable
y = tourism_data[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              
    test_size=0.3,     
    random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Jyotidarshan/tour-pckg-pred-data",
        repo_type="dataset",
    )
