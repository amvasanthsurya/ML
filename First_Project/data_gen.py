import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from load_save import *
from objective_function import *
from mealpy.swarm_based.SSA import BaseSSA

#pre_proccessing
data = pd.read_csv('C:\\Users\\kvsku\\OneDrive\\Desktop\\ISO-NE_case2.csv')
data = data.dropna(axis=1)
data = data.drop([0,1,2,3])
data = data.drop(columns='Time')
data = data.drop(data.columns[[66,67,68,69,85,86,87,88,116]], axis=1)
data = data.astype('float')
data['label'] = data.iloc[:,-1].apply(lambda x:1 if x>100 else 0)
data = data.drop(data.columns[-2], axis=1)
label = data.iloc[:,-1]
label = np.array(label)
data = data.drop(columns='label')

#feat_extraction
data['skew']=data.skew(axis=1)
data['kurtosis'] = data.kurtosis(axis=1)
columns = ['skew','kurtosis']
data['correlation'] = data[columns].corr(method='spearman').mean().mean()
feat = np.array(data)
print(feat.shape)
print(label.shape)

#split and train

feat_sel_best_pos, train_data, test_data, train_lab, test_lab = [], [], [], [], []
X_train,X_test,Y_train,Y_test = train_test_split(feat,label,test_size=0.3)
save('X_train',X_train)
save('X_test',X_test)
save('Y_train',Y_train)
save('Y_test',Y_test)

# feature Selection using SSA
lb = (np.zeros([1, X_train.shape[1]]).astype('int16'))
ub = (np.ones([1, X_train.shape[1]]).astype('int16'))
problem_dict1 = {
    "fit_func": obj_fun1,
    "lb": lb,
    "ub": ub,
    "minmax": "min",
}

epoch = 2
pop_size = 50
ST = 0.8
PD = 0.2
SD = 0.1
model = BaseSSA(epoch, pop_size, ST, PD, SD)
best_position, best_fitness = model.solve(problem_dict1)
print(f"Solution: {best_position}, Fitness: {best_fitness}")
feat_sel_best_pos.append(best_position)

## Feature selection data
best_position = np.round(best_position)
X_train = X_train[:, np.where(best_position == 1)]
X_test = X_test[:, np.where(best_position == 1)]

train_data.append(X_train)
test_data.append(X_test)
train_lab.append(Y_train)
test_lab.append(Y_test)
save('X_train', train_data)
save('X_test', test_data)
save('Y_train', train_lab)
save('Y_test', test_lab)
save('feat_sel_best_pos', feat_sel_best_pos)



