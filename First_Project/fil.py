from load_save import *
from Confusion_matrix import *
test = load("Y_test")
y_pred = test.copy()

y_pred[:100] = test[100:200]
print(y_pred)
a= confu_matrix(test,y_pred)
print(a)
