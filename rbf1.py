import numpy as np
from numpy.core.numeric import correlate
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
##############
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# X = data_train
# y = label_train[:,0]
# kernel = 1.0 * RBF(1.0)
# gpc = GaussianProcessClassifier(kernel=kernel,
#         random_state=0).fit(X, y)
# gpc.score(X, y)

# gpc.predict_proba(X[:2,:])
# print(gpc.score(X, y))
# print(gpc.predict_proba(X[:2,:]))
####################
# class_1 = np.zeros([1,33])
# class_2 = np.zeros([1,33])
# num_class1 = 0
# num_class2 = 0

# i = 0
# for i in range(len(data_train)):
#     if(label_train[i,0] == 1):
#         class_1[0,:] += data_train[i,:]
#         num_class1 += 1
#     else:
#         class_2[0,:] += data_train[i,:]
#         num_class2 += 1
# print("Class_1",num_class1)
# print("Class_2",num_class2)
# central_neuron_1 = class_1/num_class1
# central_neuron_2 = class_2/num_class2
# print("central neuraon",central_neuron_1.shape)
import math
def radial_gaussian_function(distance_in_neurons,sigma):
    g_of_d = math.exp((-1.0*distance_in_neurons*distance_in_neurons)/(2*sigma*sigma))
    return g_of_d
# phi = np.zeros([330,2])

# for j in range(len(data_train)):
#     distance_in_neurons_1 = np.linalg.norm(data_train[j] - central_neuron_1)
#     distance_in_neurons_2 = np.linalg.norm(data_train[j] - central_neuron_2)
#     phi[j,0] = radial_gaussian_function(distance_in_neurons_1,0.707)
#     phi[j,1] = radial_gaussian_function(distance_in_neurons_2,0.707)
    
# print("PHI",phi.shape)
# weights = np.matmul(np.linalg.pinv([phi]),label_train)
# # print("weights",weights)
# f_of_x = np.matmul(phi,weights)
# # print("f_of_x",f_of_x[0,:,0])

# ijk = 0
# correct = 0
# for ijk in range(len(data_train)):
#     # tmp1= round(f_of_x[0,ijk,0] - label_train[ijk,0])
#     if(f_of_x[0,ijk,0] < 0 ):
#         tmp1 = -1
#     else:
#         tmp1 = 1
#     # print(f_of_x[0,ijk,0] - label_train[ijk,0])
#     if ((tmp1 == -1 and label_train[ijk,0] ==-1 ) or ((tmp1 == 0 or tmp1 == 1 ) and label_train[ijk,0] == 1)):
#         correct +=1
# print("Acccuracy",100*(correct/len(data_train)))
# print("wrong",len(data_train)-correct)

from scipy.spatial import distance
import random
def weight_calculation(train_dataset,train_label_dataset,number_of_centres):
    i_1, i_2,i_3 = 0 ,0 , 0
    central_neurons = np.zeros([number_of_centres,data_train_column])
    distance_in_neurons = np.zeros([len(train_dataset),number_of_centres])
    PHI_1 = np.zeros([len(train_dataset),number_of_centres])
    for i_1 in range(number_of_centres):
        central_neuron_id = round(random.random() *len(data_train))
        # print("randomly chose the neurons",central_neuron_id)
        central_neurons[i_1,:]  = train_dataset[central_neuron_id,:]
    # abc = distance.euclidean(train_dataset,central_neurons)
    
    # print("centrL nEURONS",central_neurons)
    d_max = 0
    for i_2 in range(len(train_dataset)):
        for i_3 in range(number_of_centres):
            # distance_in_neurons[i_2,i_3] = distance.euclidean(train_dataset[i_2,:],central_neurons[i_3,:])
            d_distance = np.linalg.norm(train_dataset[i_2,:] - central_neurons[i_3,:])
            if (d_max<=d_distance):
                d_max = d_distance
            
            sigma = d_max/(number_of_centres**0.5)
            PHI_1[i_2,i_3] = radial_gaussian_function(d_distance,sigma)
    # print("Max_distace",d_max)
    # print("distance",distance_in_neurons)
    weights_final = np.matmul(np.linalg.pinv([PHI_1]),train_label_dataset)
    # print("weights",weights_final)
    f_of_x_1 = np.matmul(PHI_1,weights_final)
    # print("f_of_x",f_of_x[0,:,0])

    ijk_1 = 0
    correct_1 = 0
    for ijk_1 in range(len(data_train)):
        # tmp1= round(f_of_x[0,ijk,0] - label_train[ijk,0])
        if(f_of_x_1[0,ijk_1,0] < 0 ):
            tmp3 = -1
        else:
            tmp3 = 1
        # print(f_of_x[0,ijk,0] - label_train[ijk,0])
        if ((tmp3 == -1 and train_label_dataset[ijk_1,0] ==-1 ) or ((tmp3 == 0 or tmp3 == 1 ) and train_label_dataset[ijk_1,0] == 1)):
            correct_1 +=1
    print("Acccuracy",100*(correct_1/len(train_dataset)))
    # print("wrong",len(train_dataset)-correct_1)
centres = input("Enter the number of Central neurons to be considered ")
weight_calculation(data_train,label_train,int(centres))
