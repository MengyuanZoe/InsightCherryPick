import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from sklearn import metrics
import itertools

def prcurve_binary(precision,recall,area):
  pl.clf()
  pl.plot(recall, precision, label='Precision-Recall curve')
  pl.xlabel('Recall')
  pl.ylabel('Precision')
  pl.ylim([0.0, 1.05])
  pl.xlim([0.0, 1.0])
  pl.title('Precision-Recall example: AUC=%0.2f' % area)
  pl.legend(loc="lower left")
  pl.show()

def prcurve_multiclass(precision,recall,average_precision,n_classes,colors,lw=2):
  plt.clf()
  plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
          label='micro-average Precision-recall curve (area = {0:0.2f})'
          ''.format(average_precision["micro"]))
  for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=lw,
            label='Precision-recall curve of class {0} (area = {1:0.2f})'
            ''.format(i-2, average_precision[i]))

  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Precision-Recall curve on multi-class marketing funnel classification')
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':10})
  plt.show()


def plot_confusion_matrix_normalized(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  print("Normalized confusion matrix")

  print(cm)

  thresh = cm.max() / 2
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], '.2f'),
              horizontalalignment="center",
              color="white" if cm[i, j] > thresh else "black")

  #plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

    
def plot_confusion_matrix_unnormalized(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print('Confusion matrix, without normalization')
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
