"""This script extracts features from the dataset."""
import pandas as pd
from param import Params

# Create directory to store extracted features.
Params.features.mkdir(parents=True, exist_ok=True)

# Loading in train and test data
train_df = pd.read_csv(str(Params.data / 'train.csv'))
test_df = pd.read_csv(str(Params.data / 'test.csv'))

# Create function to extract and preprocess features from the data.


def feature_extraction(dframe):
    """Function to extract features."""
    deff = dframe.loc[:, dframe.columns != 'diagnosis']
    return deff


train_features = feature_extraction(train_df)
test_features = feature_extraction(test_df)

#Preprocess target for training model.


def preprocess_target(dframe):
    """Function to preprocess target feature."""
    specs = {
        'M': 0, 'B': 1
    }
    dframe['diagnosis'] = dframe['diagnosis'].map(specs)
    dfr = dframe['diagnosis']
    return dfr


train_target = preprocess_target(train_df)
test_target = preprocess_target(test_df)

# Saving features and target extracted to the features directory
train_features.to_csv(str(Params.features / 'train_features.csv'), index=None)
test_features.to_csv(str(Params.features / 'test_features.csv'), index=None)
train_target.to_csv(str(Params.features / 'train_target.csv'), index=None)
test_target.to_csv(str(Params.features / 'test_target.csv'), index=None)
