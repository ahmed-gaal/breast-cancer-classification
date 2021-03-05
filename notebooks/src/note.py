"""This module stores all methods used in the notebook."""
import ppscore as pps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score


class Experiment():
    """Base class to store the methods."""


    def __init__(self):
        """Initialize the class object."""
    

    def memory_stats(df):
        """Method to compute memory stats for each feature."""
        return pd.DataFrame(
            df.memory_usage(deep=True),
            columns=['Memory']
        )


    def pps_heatmap(df):
        """
            Function for calculating the Predictive Power Score and plotting a heatmap
                Args:
                    Pandas DataFrame or Series object
                __________
                Returns:
                    figure
        """
        pps_mtrx = pps.matrix(df)
        pps_mtrx1 = pps_mtrx[['x', 'y', 'ppscore']].pivot(columns='x', index='y',
                                                  values='ppscore')
        plt.figure(figsize = (15, 8))
        ax = sb.heatmap(pps_mtrx1, vmin=0, vmax=1, cmap="afmhot_r", linewidths=0.5,
                        annot=True)
        ax.set_title("PPS matrix")
        ax.set_xlabel("feature")
        ax.set_ylabel("target")
        return ax
    

    def corr_heatmap(df, mask: bool):
        """Method to visualize correlation."""
        plt.figure(figsize=(24, 8))
        if mask is True:
            # Create mask
            mask = np.zeros_like(df.corr(), dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
            # Generate Custom diverging cmap
            sb.heatmap(df.corr(), annot=True, cmap='cividis', linewidth=.5,
                       mask=mask)
        else:
            sb.heatmap(df.corr(), annot=True, cmap='cividis')
        return plt.show()



    def mean_accuracy_score(est, X, y, cv: int):
        """Method to calculate average accuracy of the model."""
        res = cross_val_score(est, X, y, cv=cv, n_jobs=-1, verbose=1,
                                scoring='accuracy')
        score = print('Average Accuracy:', (np.mean(res)))
        std = print('Average Standard Deviation:', (np.std(res)))
        return
    

    def plot_prc_figure(precision, recall, thresh):
        """Method for plotting Precision Recall Curve"""
        plt.figure(figsize=(24, 10))
        plt.plot(thresh, precision[:-1], 'r--', label='Precision')
        plt.plot(thresh, recall[:-1], 'g--', label='Recall')
        plt.title('Precision Recall Curve')
        plt.xlabel('Threshold')
        plt.legend(loc='best')
        plt.ylim([-0.5, 1.5])
        plt.show()
        return
    

    def roc_curve_plot(fpr, tpr, truth, pred, label=None):
        """Method to plot receiver operator characteristics curve."""
        roc = print('ROC Score:', roc_auc_score(truth, pred))
        plt.figure(figsize=(18, 10))
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()
        return roc

    
    def plot_roc_curve(fpr, tpr, thresholds):
        """Method to plot roc curve with rich visualization."""
        specs = pd.DataFrame({
            'FALSE POSITIVE RATE': fpr,
            'TRUE POSITIVE RATE': tpr
        }, index=thresholds)

        specs.index.name = "Thresholds"
        specs.columns.name = "Rate"

        fig = px.line(
            specs, title='TPR AND FPR AT EVERY THRESHOLD', width=480,
            height=640
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(range=[0, 1], constrain='domain')
        return fig.show()

    
    def print_score(est, X_train, X_test, y_train, y_test, train: bool):
        """
        Method to output metrics of classifier using train or test data
        """

        lab = LabelBinarizer()
        lab.fit(y_train)
        if train is True:
            res = est.predict(X_train)
            print("Train Score:\n")
            print('Accuracy Score:{0:.4f}\n'.format(accuracy_score(y_train, res)))
            print("ROC AUC:{0:.4f}\n".format(roc_auc_score(y_train, lab.transform(res))))
            print("Classification Report: \n {} \n".format(
                classification_report(y_train, res)
            ))
            print("Confusion Matrix: \n {} \n".format(confusion_matrix(y_train, res)))
            nres = cross_val_score(
                est, X_train, y_train, cv=10, scoring='accuracy'
            )
            print("Average Accuracy: {0:.4f}\n".format(np.mean(nres)))
            print("Average Standard Deviation {0:.4f}\n".format(np.std(nres)))
            return
        else:
            res_test = est.predict(X_test)
            print("Test Score:\n")
            print('Accuracy Score:{0:.4f}\n'.format(accuracy_score(y_test, res_test)))
            print("ROC AUC:{0:.4f}\n".format(roc_auc_score(y_test, lab.transform(res_test))))
            print("Classification Report: \n {} \n".format(
                classification_report(y_test, res_test)
            ))
            print("Confusion Matrix: \n {} \n".format(confusion_matrix(y_test, res_test)))
            return

