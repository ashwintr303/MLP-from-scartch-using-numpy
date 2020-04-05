# MLP-from-scartch-using-numpy
Implementation of multi-layer perceptron from scratch using numpy for MNIST classification 

This code implements a multi-layer perceptron trained using backpropogation. The model includes regularization and learning rate scheduling to avoid overfitting and improve accuracy.

## Requirements:

1. Python 3 or above.
2. h5py  
Run:
>pip install h5py

## Running the code

### 1. Set the file path in main.
path = 'your_file_path'

### 2. Set the model and hyperparameters in the network arcitecture section.
For example, to create a network with 3 layers (input, hidden, output) with 784, 50 and 10 nodes and with activation functions for the hidden and the output layers as relu and softmax repectively: 

> nn = network([784, 50, 10],['relu', 'softmax'])

The code currently supports relu, tanh and softmax. You ccan add your activations and their derivatives in the methods activation_function() and derivatives() of the network class.

Batch size, number of epochs, learning rate and regularization factor can be set here:
>nn.train(x_train,y_train,x_val,y_val,batch_size=10,epochs=20,lr=0.0005,reg=0.001)

## References

[1] https://www.coursera.org/lecture/machine-learning/backpropagation-algorithm-1z9WW  
[2] https://medium.com/@a.mirzaei69/implement-a-neural-network-from-scratch-with-python-numpy-backpropagation-e82b70caa9bb  
[3] https://www.jeremyjordan.me/neural-networks-activation-functions/  
[4] https://stats.stackexchange.com/questions/215521/how-to-find-derivative-of-softmax-function-for-the-purpose-of-gradient-descent
