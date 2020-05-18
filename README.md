# Image-Classifier-using-Logistic-Regression
Logistic Regression is a starting point in understanding the neural network known as the hello world of deep learning.
This project is done from the scratch without any frameworks like tensorflow, keras, etc.
I used logistic loss function as my cost function and calculated the back propagation da,dw,dw.
I got the accuracy of 70% as I used only one node(neuron).
<br>
<img heigth = "700" src="https://miro.medium.com/max/1400/1*TvNwzBfbyvzHCR6gJM1Wrg.png"/>
<br>
I converted the image into a array of size (64*64*3,1)which is nothing but flatting the image.And then sent this to the node 
as shown in the image.
Each array element of 12288,1 has a weight and bias corresponding to it.I intialized the weights and bias to zeros in thr start 
thrn updated the value using back propagation and learning rate using the formula
w = w - learning_rate*dw
b = b - learning_rate*db
