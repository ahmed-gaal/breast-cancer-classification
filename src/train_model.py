"""This script performs model training on the preprocessed data."""
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from param import Params

# Create a directory to save models developed.
Params.models.mkdir(parents=True, exist_ok=True)

# Loading train features into pandas DataFrame.
X_train = pd.read_csv(str(Params.features / 'train_features.csv'))
y_train = pd.read_csv(str(Params.features / 'train_target.csv'))

# Instantiating and fitting the algorithm.
model = GradientBoostingClassifier(criterion='friedman_mse', n_estimators=400)
model.fit(X_train, y_train.to_numpy().ravel())

# Saving model in serialized format.
pickle.dump(model, open(
    str(Params.models / 'model.pickle'), 'wb'
))
