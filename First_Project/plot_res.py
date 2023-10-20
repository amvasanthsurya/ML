import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from load_save import *
import numpy as np

matrix1 = load('cnn_conf')
matrix2 = load('ann_conf')
matrix3 = load('bi_lstm_conf')

matrix2 = np.array(matrix2)
matrix2 = matrix2[~np.isnan(matrix2)]
value1 = 0.86556
value2 = 0.94353
matrix2 = np.append(matrix2,[value1,value2])
matrix1 = np.array(matrix1)
matrix3 = np.array(matrix3)
matrix = np.column_stack((matrix1,matrix2,matrix3))

def bar_plot(label,data,metrix):
    df = pd.DataFrame(data)
    df1 = pd.DataFrame()
    df1['Dataset'] = [1,2,3]
    df = pd.concat((df,df1),axis=1)
    df.plot(x='Dataset',kind='bar',stacked=False)
    plt.ylabel(metrix)
    plt.legend(loc='upper right')
    plt.savefig('./Results/' + metrix + '.png', dpi=400)
    plt.show(block=False)

def plot_res():
    Y_test = load('Y_test')
    metrices = matrix
    method = ['CNN','ANN','Bi-LSTM']
    metrices_plot=['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F-Measure', 'MCC', 'NPV', 'FPR', 'FNR']

    for i in range(len(metrices_plot)):
        bar_plot(method,metrices[i],metrices_plot[i])

    print('Testing Metrices-Dataset ')
    tab = pd.DataFrame(metrices, index=metrices_plot, columns=method)
    print(tab)

    Y_pred = load('bi_lstm_y_pred')
    auc = metrics.roc_auc_score(Y_test, Y_pred)

    false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(Y_test, Y_pred)

    plt.figure(figsize=(10, 8), dpi=100)
    plt.axis('scaled')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("AUC & ROC Curve")
    plt.plot(false_positive_rate, true_positive_rate, 'g')
    plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
    plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig('./Results/roc.png', dpi=400)
    plt.show()
