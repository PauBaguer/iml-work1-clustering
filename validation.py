from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


def conf_matrix(gt, predicted):
    cm = metrics.confusion_matrix(gt, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(gt))
    cm_display.plot()
    plt.show()
