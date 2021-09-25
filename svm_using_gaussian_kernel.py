from enum import auto
import numpy as np
# from numpy.core.numeric import correlate
###import the scipy for loading .mat matlab data
from scipy.io import loadmat
### loading the data in same name variable
data_test_tmp = loadmat('data_test.mat')
data_train_tmp = loadmat('data_train.mat')
label_train_tmp = loadmat('label_train.mat')
#### printing the type and size of the variables
print("data_test ",type(data_test_tmp),"_",len(data_test_tmp))
print("data_train ",type(data_train_tmp),"_",len(data_train_tmp))
print("label_train ",type(label_train_tmp),"_",len(label_train_tmp))
###### converting the class dictionary to numpy array
data_test = data_test_tmp["data_test"]
data_train = data_train_tmp["data_train"]
label_train = label_train_tmp["label_train"]
data_test_row,data_test_column = data_test.shape
data_train_row,data_train_column = data_train.shape
label_train_row,label_train_column = data_test.shape
print("data_test ",type(data_test),"_",data_test_row,data_test_column)

print("data_train ",type(data_train),"_",data_train_row,data_train_column)
print("label_train ",type(label_train),"_",label_train_row,label_train_column)

label_test = np.array([[1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1]])
# i= 0
# avg = np.zeros(len(data_test))
# j = list()
# for i in range(len(data_test)):
#     avg = np.mean(data_test[i,:])
#     if(avg>0):
#         j.append(1)
#     else:
#         j.append(-1)
# print(avg)
# print(j)
# print(data_test)
# import numpy as np
# import matplotlib.pyplot as plt

# H = data_test[:,:]  # added some commas and array creation code

# fig = plt.figure(figsize=(6, 3.2))

# ax = fig.add_subplot(111)
# ax.set_title('classic')
# plt.imshow(H)
# ax.set_aspect('equal')

# cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
# cax.get_xaxis().set_visible(False)
# cax.get_yaxis().set_visible(False)
# cax.patch.set_alpha(0)
# cax.set_frame_on(False)
# plt.colorbar(orientation='vertical')
# plt.show()
X_train = data_train
y_train = label_train
X_test = data_test
y_test = np.transpose(label_test)
from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf',gamma='auto')
svclassifier.fit(X_train, y_train)
#To use Gaussian kernel, you have to specify 'rbf' as value for the Kernel parameter of the SVC class.

#Prediction and Evaluation
y_pred = svclassifier.predict(X_test)
# y_train_pred = svclassifier.predict(X_train)
print(y_pred)
i=0
for i in range(len(y_pred)):
    print(y_pred[i])
from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
#The output of the Kernel SVM with Gaussian kernel looks like this:
