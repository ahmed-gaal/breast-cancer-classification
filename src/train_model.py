"""The following script performs model training on the preprocessed data."""
import pickle
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from param import Params

# Create a directory to save models developed.
Params.models.mkdir(parents=True, exist_ok=True)

# Loading train features into pandas DataFrame.
X_train = pd.read_csv(str(Params.features / 'train_features.csv'))
y_train = pd.read_csv(str(Params.features / 'train_target.csv'))

# Instantiating and fitting the algorithm.
base_rf = RandomForestClassifier(n_estimators=1000)
model = AdaBoostClassifier(
    algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
    n_estimators=1000, random_state=None
)
model.fit(X_train, y_train.to_numpy().ravel())

# Saving model in serialized format.
pickle.dump(model, open(
    str(Params.models / 'model.pickle'), 'wb'
))
