import numpy as np
import h5py
from random import shuffle

class network(object):
    
    #weights and bias initilization
    def __init__(self,layers,activations):
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []
        for i in range(len(self.layers) - 1):
            self.weights.append(np.random.randn(layers[i+1],layers[i])*0.01)
            self.biases.append(np.ones((layers[i+1],1))*0.01)
        
        print('w0',np.shape(self.weights[0]))
        print('w1',np.shape(self.weights[1]))
        print('b0',np.shape(self.biases[0]))
        print('b1',np.shape(self.biases[1]))
        
    def shuffle_data(self,x,y):
        ind_list = [i for i in range(len(y))]
        shuffle(ind_list)
        x_new  = x[ind_list, :]
        y_new = y[ind_list,]
        return x_new,y_new

    #activation functions
    def activation_function(self,x,func):
        if func == 'softmax':
            e = 10e-5
            return np.exp(x - x.max(axis=0)) / (np.exp(x - x.max(axis=0))+e).sum(axis = 0)
        elif func == 'relu':
            return np.maximum(0,x)
        elif func == 'tanh':
            return np.tanh(x)
        else:
            print('Activation function not defined')
            exit

    #derivatives of activation functions
    def derivatives(self,x,y,func):
        if func == 'relu':
            re = np.greater(x, 0).astype(int)
            return re
        elif func == 'tanh':
            return 1 - np.power(np.tanh(x),2)
        elif func == 'softmax':
            result = [(x[i] - y[i]) for i in range(len(y))]
            return result
        else:
            print('Activation function not defined')
            exit
    
    #feedforward training
    def feedforward(self,input_data):
        temp = np.copy(input_data)
        s = []
        a = [input_data]
        for i in range(len(self.weights)):
            s.append(temp @ self.weights[i].T + self.biases[i].T)
            temp = self.activation_function(s[i],self.activations[i])
            a.append(temp)
        return s,a
    
    #backpropogation
    def backprop(self,s,a,y_batch):
        dw = [] #dC/dW
        db = [] #dC/dB
        delta = [0] * len(self.weights) #dC/dS
        delta[-1] = self.derivatives(a[-1],y_batch,'softmax')
        delta[-1] = np.asarray(delta[-1])
        for i in reversed(range(len(delta)-1)):
            delta[i] = ((delta[i+1]) @ self.weights[i+1]) * self.derivatives(s[i],y_batch,self.activations[i])
            batch_size = len(y_batch)  
        dw = [(a[i].T @ d)/float(batch_size) for i,d in enumerate(delta)]
        db = [d.T.dot(np.ones((batch_size,1)))/float(batch_size) for d in delta]
        return db, dw
    
    #validation
    def test_val(self,x,y):
        s_val, a_val = self.feedforward(x)
        temp2 = []
        for k in a_val[-1]:
            observed = np.zeros(len(k))
            observed[np.argmax(k)] = 1
            temp2.append(observed)
        c2 = 0
        #print(len(y))
        for m in range(len(y)):
            if np.array_equal((temp2[m]),(y[m])):
                c2 += 1
        print('val_acc',c2/10000)
    
    #train the model    
    def train(self,x,y,x_val,y_val,batch_size=1000,epochs=20,lr=0.01,reg=0.01):
        
        for e in range(epochs):
            i = 0
            
            #learning rate scheduling
            #learning rate gets halved after every 10 epochs
            if not e%10 & e>0:
                lr = lr/2
            print('epoch '+str(e))

            while(i < len(y)):
                x_batch = x[i:i+batch_size] 
                y_batch = y[i:i+batch_size]
                
                self.shuffle_data(x_batch,y_batch)
                i += batch_size
                s, a = self.feedforward(x_batch)
                
                db, dw = self.backprop(s, a, y_batch)
                self.weights = [w - lr*(r.T + 2*reg*w) for w,r in  zip(self.weights, dw)]
                self.biases = [b - lr*(r + 2*reg*b) for b,r in  zip(self.biases, db)]
            
            s_ff, a_ff = self.feedforward(x)
            temp1 = []
            for k in a_ff[-1]:
                observed = np.zeros(len(k))
                observed[np.argmax(k)] = 1
                temp1.append(observed)
            c1 = 0
            #print(len(y))
            for m in range(len(y)):
                if np.array_equal((temp1[m]),(y[m])):
                    c1 += 1
            print('train_acc:',(c1/50000))
            
            self.test_val(x_val,y_val)
            
#driver code
if __name__=='__main__':
    
  #extract data  
  path = 'file_path/mnist_traindata.hdf5'
  f = h5py.File(path, 'r')
  x_data = f['xdata']
  y_data = f['ydata'] 
  
  #split for training and validation
  x_train, x_val = x_data[:50000,:], x_data[50000:,:]
  y_train, y_val = y_data[:50000], y_data[50000:]
  
  #reshape and normalize 
  x_train = x_train.astype(np.float32) / 255.0
  x_val = x_val.astype(np.float32) / 255.0
  print('xtrain:',np.shape(x_train))
  print('ytrain:',np.shape(y_train))
  print('xval:',np.shape(x_val))
  print('yval:',np.shape(y_val))
  
  #describe the network architecture
  nn = network([784, 50, 10],['relu', 'softmax'])
  nn.shuffle_data(x_train,y_train)
  nn.shuffle_data(x_val,y_val)
  nn.train(x_train,y_train,x_val,y_val,batch_size=10,epochs=20,lr=0.0005,reg=0.001)
  print("working!!!")
