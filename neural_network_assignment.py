import numpy as np
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
#########################################################
import math
def radial_gaussian_function(distance_in_neurons,sigma):
    g_of_d = math.exp((-1.0*distance_in_neurons*distance_in_neurons)/(2*sigma*sigma))
    return g_of_d
import random

def weight_calculation(train_dataset,train_label_dataset,number_of_centres):
    i_1, i_2,i_3 = 0 ,0 ,0
    if (prechoose==0):
        central_neurons = np.zeros([number_of_centres,data_train_column])
    else:
        number_of_centres = 20
        central_neurons = np.zeros([number_of_centres,data_train_column])    
    distance_in_neurons = np.zeros([len(train_dataset),number_of_centres])
    PHI_1 = np.zeros([len(train_dataset),number_of_centres])
    for i_1 in range(number_of_centres):
        if (prechoose == 0):
            central_neuron_id = round(random.random() *len(data_train))
            # print("randomly chose the neurons",central_neuron_id)
            central_neurons[i_1,:]  = train_dataset[central_neuron_id,:]
        else:
            random_central_neruon =[141, 4, 329, 147, 318, 133, 185, 90, 47, 306, 48, 60, 12, 124, 41, 213, 99, 21, 308, 325]
            number_of_centres = 20
            central_neurons[i_1,:]  = train_dataset[random_central_neruon[i_1],:]

    d_max = 0
    for i_2 in range(len(train_dataset)):
        for i_3 in range(number_of_centres):
            d_distance = np.linalg.norm(train_dataset[i_2,:] - central_neurons[i_3,:])
            if (d_max<=d_distance):
                d_max = d_distance
            else:
                d_max = d_max
            
            sigma = d_max/(number_of_centres**0.5)
            PHI_1[i_2,i_3] = radial_gaussian_function(d_distance,sigma)
    weights_final = np.matmul(np.linalg.pinv([PHI_1]),train_label_dataset)
    
    print("TRAIN WEIGHTS",weights_final[0,:,:].shape)
    print("TRAIN_PHI",PHI_1.shape)
    f_of_x_1 = np.matmul(PHI_1,weights_final)
    print("TRAIN f_of_x",f_of_x_1[0,:,:].shape)

    ijk_1 = 0
    correct_1 = 0
    for ijk_1 in range(len(data_train)):
        if(f_of_x_1[0,ijk_1,0] < 0 ):
            tmp3 = -1
        else:
            tmp3 = 1
        if ((tmp3 == -1 and train_label_dataset[ijk_1,0] ==-1 ) or ((tmp3 == 0 or tmp3 == 1 ) and train_label_dataset[ijk_1,0] == 1)):
            correct_1 +=1
    print("Training Acccuracy",100*(correct_1/len(train_dataset)))
    return weights_final,central_neurons

def test_pred(test_dataset,weights_final_from_train_,central_neurons,number_of_centres):
    distance_in_neurons = np.zeros([len(test_dataset),number_of_centres])
    PHI_2 = np.zeros([len(test_dataset),number_of_centres])
    d_max = 0
    for i_2 in range(len(test_dataset)):
        for i_3 in range(number_of_centres):
            d_distance = np.linalg.norm(test_dataset[i_2,:] - central_neurons[i_3,:])
            if (d_max<=d_distance):
                d_max = d_distance
            sigma = d_max/(number_of_centres**0.5)
            PHI_2[i_2,i_3] = radial_gaussian_function(d_distance,sigma)

    test_label_dataset_output = np.matmul(PHI_2,weights_final_from_train_)
    # print("test_label_predicition", test_label_dataset_output)
    # print("test_label_predicition",test_label_dataset_output.shape)
    test_label_dataset_output1 = test_label_dataset_output[0,:,:]
    print("test_label_predicition",test_label_dataset_output1.shape)
    pred = list()
    iem = 0
    for iem in range(len(test_label_dataset_output1)):
        if(test_label_dataset_output1[iem,0] < 0):
            pred.append(-1)
        else:
            pred.append(1)            
    # print("Prediction result ",pred)
    return pred
def svm():
    X_train = data_train
    y_train = label_train
    X_test = data_test
    # y_test = np.transpose(label_test)
    from sklearn.svm import SVC
    svclassifier = SVC(kernel='rbf',gamma='auto')
    svclassifier.fit(X_train, y_train)
    #To use Gaussian kernel, you have to specify 'rbf' as value for the Kernel parameter of the SVC class.

    #Prediction and Evaluation
    y_pred = svclassifier.predict(X_test)
    # y_train_pred = svclassifier.predict(X_train)
    print("SVM Prediction",y_pred)
    return y_pred
import sys
centres = int (sys.argv[1])
prechoose = int (sys.argv[2])
# centres = input("Enter the number of Central neurons to be considered ")
print("Number of centres", centres)
train_weights,trian_central_neurons = weight_calculation(data_train,label_train,int(centres))
label_test_prediction = test_pred(data_test,train_weights,trian_central_neurons,int(centres))
print ("Label_test",np.transpose(label_test_prediction))
svm_pred = svm()
i = 0
correct = 0
correct_1 = 0
for i in range(len(svm_pred)):
    if(svm_pred[i] ==  label_test_prediction[i]):
        correct += 1
    else:
        correct += 0
print("Considering the svm is the most coorect one")
print("rbf is ",100*( correct/len(svm_pred)),"accuracy to SVM")
