import numpy as np
from load_save import *
from classifier import cnn

def obj_fun1(soln):

    X_train=load('X_train')
    X_test=load('X_test')
    Y_train=load('Y_train')
    Y_test=load('Y_test')

    # Feature selection78
    soln = np.round(soln)
    X_train = X_train[:,np.where(soln==1)[0]]
    X_test = X_test[:, np.where(soln == 1)[0]]
    pred, met,= cnn(X_train, X_test,Y_test,Y_train)
    fit = 1/met[0]
    return fit

