"""This script extracts data from remote storage."""
import os
import gdown
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from param import Params

# Generating random seed
np.random.seed(42)
random = Params.random_state

# Create directories to store extracted data.
Params.original.parent.mkdir(parents=True, exist_ok=True)
Params.data.mkdir(parents=True, exist_ok=True)

# Extract data from remote storage.
gdown.download(
    os.environ.get('DATA'),
    str(Params.original)
)

# Loading data extracted for into pandas DataFrame.
df = pd.read_csv(str(Params.original))

df.drop(['Unnamed: 32'], axis=1, inplace=True)

# Splitting data to train and test sets.
df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)

# Saving data to file directories.
df_train.to_csv(str(Params.data / 'train.csv'), index=None)
df_test.to_csv(str(Params.data / 'test.csv'), index=None)
