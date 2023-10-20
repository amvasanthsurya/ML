from objective_function import obj_fun1
from load_save import save, load
from mealpy.swarm_based.SSA import BaseSSA
from classifier import *
from plot_res import *
def full_analysis():

    X_train = load('X_train')
    X_test = load('X_test')
    Y_train = load('Y_train')
    Y_test = load('Y_test')

    ## Setting parameters
    obj_func = obj_fun1
    # lb = [-15, -10, -3, -15, -10, -3, -15, -10, -3, -15, -10, -3, -15, -10, -3]
    # ub = [15, 10, 3, 15, 10, 3, 15, 10, 3, 15, 10, 3, 15, 10, 3]
    lb = [0]
    ub = [1]
    problem_size = 2
    batch_size = 25
    verbose = True
    epoch = 10
    pop_size = 10
    cnn_val, ann_val, rf_val=[], [], []

    save('cur_X_train', X_train)
    save('cur_X_test', X_test)
    save('cur_y_train', Y_train)
    save('cur_y_test', Y_test)

    # CNN
    pred, met = cnn(X_train,X_test,Y_train,Y_test)
    cnn_val.append([met])

    # ANN
    pred, met = ann(X_train,X_test,Y_train,Y_test)
    ann_val.append([met])

    # RF
    pred, met = bi_lstm(X_train,X_test,Y_train,Y_test)
    rf_val.append([met])


full_analysis()





