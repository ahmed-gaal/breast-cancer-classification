"""The follwoing scirpt evaluates the accuracy of the model developed."""
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from param import Params

# Create directory to store metrics for model performance.
Params.metrics.mkdir(parents=True, exist_ok=True)

# Load in test features for model evaluation.
X_test = pd.read_csv(str(Params.features / 'test_features.csv'))
y_test = pd.read_csv(str(Params.features / 'test_target.csv'))

# Load in the pretrained model.
model = pickle.load(open(
    str(Params.models / 'model.pickle'), 'rb'
))

# Perform predictions on the model.
pred = cross_val_predict(
    model, X_test, y_test.to_numpy().ravel(), cv=3, n_jobs=-1, verbose=1
)

# Calculate metrics
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
conf = pd.DataFrame(
    confusion_matrix(y_test, pred),
    index=pd.MultiIndex.from_product(
        [['Actual'], ['Negative', 'Positive']]
    ),
    columns=pd.MultiIndex.from_product(
        [['Predicted'], ['Negative', 'Positive']]
    )
)
res = cross_val_score(
    model, X_test, y_test.to_numpy().ravel(), scoring='accuracy', cv=10,
    n_jobs=-1
)
acc = accuracy_score(y_test, pred)
ave = np.mean(res)

# Store results in the metrics directory
with open(str(Params.metrics / 'metrics.json'), 'w') as outfile:
    json.dump(
        dict(zip(['Precision', 'Recall', 'Accuracy', 'Average Accuracy'],
        [round(prec, 3), round(rec, 3), round(acc, 3), round(ave, 3)])), outfile
    )
conf.to_csv(
    str(Params.metrics / 'confusion_matrix.csv')
)
pd.DataFrame(
    pred, columns=['Predictions']
).to_csv(
    str(Params.features / 'predictions.csv'), index=None
)
