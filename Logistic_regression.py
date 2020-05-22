import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image
from scipy import ndarray

train_dataset = h5py.File('/Users/saichakradhar/Desktop/LR/train_catvnoncat.h5',"r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
plt.imshow(train_dataset["train_set_x"][7])
train_set_y = np.array(train_dataset["train_set_y"][:]) # your train set labels
test_dataset = h5py.File('/Users/saichakradhar/Desktop/LR/test_catvnoncat.h5', "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
test_set_y = np.array(test_dataset["test_set_y"][:]) # your test set labels
classes = np.array(test_dataset["list_classes"][:]) # the list of classes
train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))


#flattening the image uding numpy.reshape()
x = train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*train_set_x_orig.shape[3]
train_set_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],x).T
test_set_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

#getting the normalized data
train_set_x = train_set_flatten/255
test_set_x = test_set_flatten/255

def sigmoid(x):
  s = 1/(1+np.exp(-x)) #declaring the sigmoid function
  return s
def initalize_parameters(dim):
  w = np.zeros((dim,1)) #Making dummy weights which are updated later
  b = 0.0
  return w,b
def propagate(w , b , X , Y):  # Calculating the slopes dw and db
  m = X.shape[1]
  A = sigmoid(np.dot(w.T,X)+b)
  cost = (1/m)*np.sum((Y*np.log(A),(1-Y)*np.log(1-A)))
  #BackPropagation
  da = (A-Y)
  dw = (1/m)*(np.dot(X,da.T))
  db = np.sum(da)
  #combining dw and db in grades
  grades = {
      "dw":dw,
      "db":db 
  }
  return grades,cost
def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost =False):
 
  for i in range(0,num_iterations):				# Changing w ang b values using the slopes for num_iterations times
    grades,cost = propagate(w,b,X,Y)
    dw = grades["dw"]
    db = grades["db"]
    w = w - learning_rate*dw
    b = b - learning_rate*db
   
  params = {
      "w":w,
      "b":b
  }
  grades ={
      "dw":dw,
      "db":db
  }
  return params,grades,cost
def predict(w,b,X):
  m = X.shape[1]					#Predicting the values
  Y_prediction = np.zeros((1,m))
  w = w.reshape(X.shape[0],1)
  A = sigmoid(np.dot(w.T,X)+b)
  for i in range(0,A.shape[0]):
    Y_prediction = 1*(A>0.5)
  return Y_prediction
def model(X_train,X_test,Y_train,Y_test,num_iterations=1000,learning_rate=0.00005,print_cost=False):
  # Putting all togeather	
  w,b = initalize_parameters(X_train.shape[0])
  params,grades,cost = optimize(w,b,X_train,Y_train,num_iterations,learning_rate)
  w = params["w"]
  b = params["b"]
  Y_prediction_test = predict(w,b,X_test)
  Y_prediction_train = predict(w,b,X_train)
  test_accuracy = 100 - np.mean(np.abs(Y_prediction_test-Y_test))*100
  print("Accuary of test set is:")
  print(test_accuracy)
  return Y_prediction_test

x = model(train_set_x, test_set_x, train_set_y, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
plt.imshow(test_set_x_orig[13])
print("Predicted_vlaues:")
print(x)
print("Actual Values:")
print(test_set_y)