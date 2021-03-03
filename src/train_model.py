"""This script performs model training on the preprocessed data."""
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from param import Params

# Create a directory to save models developed.
Params.models.mkdir(parents=True, exist_ok=True)

# Loading train features into pandas DataFrame.
X_train = pd.read_csv(str(Params.features / 'train_features.csv'))
y_train = pd.read_csv(str(Params.features / 'train_target.csv'))

# Instantiating and fitting the algorithm.
model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.2, max_delta_step=0, max_depth=3,
              min_child_weight=1, missing=None, n_estimators=200, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
model.fit(X_train, y_train.to_numpy().ravel())

# Saving model in serialized format.
pickle.dump(model, open(
    str(Params.models / 'model.pickle'), 'wb'
))
