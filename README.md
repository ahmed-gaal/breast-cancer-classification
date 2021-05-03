# Breast-Cancer-Classification

## Introduction
This is a Machine Learning workflow built using Data Version Control.

The data used in this workflow has been obtained from Kaggle.

The objective of this ML Workflow is to accurately classifiy the
diagnosis of patients to whether the cancer is **malignant** or **benign**.

### This steps of this workflow consists of

*   Extracting the data from a remote storage.

*   Feature extraction, preprocessing and transformation.

*   A machine learning algorithm is applied to the transformed data and saved in a
serialized format.

*   Finally the last step of our pipeline is model evaluation. 

--------
#### The metrics used are
*   Confusion Matrix
*   Accuracy Score
*   Mean Accuracy Score using 10 fold cross validation
*   Precision Score
*   Recall Score

--------
## How to build this workflow
*   First you need to clone this repository using:
>   ```git clone https://github.com/ahmed-gaal/breast-cancer-classification.git```
*   Then you need to create a virtual environment using:
>   ```python3 -m venv env```
*   After creating a virtual environment, install the project dependencies using:
>   ```pip install -r requirements.txt```
*   Add the original data to your environment variables:
>   ```export DATA='https://drive.google.com/uc?id=1s5vLuJ0zRq6Gjk7fkGQUI24AkR3Gjb_I'```
*   Finally, create changes in either train script or feature extraction script and 
reproduce the workflow using the following command:
>   ```dvc repro```

To launch the notebook, click ⇢ [here](https://mybinder.org/v2/gh/ahmed14-cell/breast-cancer-classification/HEAD)

--------