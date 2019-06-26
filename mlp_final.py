# -*/- coding: utf-8 -*-
"""
Created on Wed Feb 13 11:12:30 2019

@author: saloni
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pylab


f=h5py.File('mnist_traindata.hdf5','r')

with h5py.File('mnist_traindata.hdf5', 'r') as df:
  xdata = df['xdata'][:]
  ydata=df['ydata'][:]
print(list(f))

xtr=xdata[:-10000]
xval=xdata[-10000:]
ytr=ydata[:-10000]
yval=ydata[-10000:]
print(np.shape(xtr))
print(np.shape(xval))
print(np.shape(ytr))
print(np.shape(yval))

def relu(x):
    return np.maximum(0,x)

def relu_der(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j]>0:
                x[i][j]=1
            else:
                x[i][j]=0
    return x

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def tanh1(x):
    return np.tanh(x)

def tanh_der(x):
    return (1-np.tanh(x)**2)
def sigmoid(x):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
    """Derivative of the sigmoid function."""
    return sigmoid(x)*(1-sigmoid(x))

def init_w(hidden):
    w={}
    for i in range(1,len(hidden)):
        w['W'+str(i)] = np.random.normal(0,2/hidden[i],(hidden[i],hidden[i-1]))*0.01
    return w
    
def init_b(hidden):
    param={}
    for i in range(1,len(hidden)):
        param['b'+str(i)] = np.zeros((hidden[i],1))
    return param  

def predict(x,y,w,b,hidden):
    
    
# =============================================================================
#     	outputs["A0"] = np.transpose(val_x[:][:])
# 	y = val_y[:][:]
# 	y = np.transpose(y)
# =============================================================================
    a={}
    z={}
    a["A0"] = np.transpose(x)
    y=y[:][:]
    val_acc=0
    y=np.transpose(y)
    for i in range(1,len(hidden)):
        if(i==len(hidden)-1):
            z['Z'+str(i)]=np.dot(w['W'+str(i)],a['A'+str(i-1)])+b['b'+str(i)]
            a['A'+str(i)]=softmax(z['Z'+str(i)])
        else:
            z['Z'+str(i)]=np.dot(w['W'+str(i)],a['A'+str(i-1)])+b['b'+str(i)]
            a['A'+str(i)]=tanh1(z['Z'+str(i)])
            
    cost1=np.sum(y*np.log(a['A'+str(len(hidden)-1)]),axis=0)
    cost=np.sum(cost1)
    cost=(-1/10000)*cost
   # print(cost)
    
    y=y.T
    y1=a['A'+str(len(hidden)-1)].T
    for j in range(y.shape[0]):
        y_pred=np.argmax(y1[j])
        y_val=next((i for i, x in enumerate(y[j]) if x), None)
    
        if(y_pred==y_val):
            val_acc+=1
    val_acc=val_acc
    return(val_acc,cost)


        
        
def model(xdata,ydata,hidden,epoch,activation,eta,reg_lambda,yval,xval,batch_size):
    #alpha=10e-6
    w=init_w(hidden)
    print(len(w))
    #print(w)
    b=init_b(hidden)
    print(len(b))
    #print(b)
    a={}
    z={}
    grad={}
    cost_final=[]
    accuracy_val=[]
    accuracy_tra=[]
    for j in range(epoch):
        
        
        #Feedforward propogation
        temp=0
        for k in range(100):
            x=xdata[temp*batch_size:(temp+1)*batch_size][:]
            y=ydata[temp*batch_size:(temp+1)*batch_size][:]
            y=np.transpose(y)

            a["A0"] = np.transpose(x)
            
           
            for i in range(1,len(hidden)):
                if(i==len(hidden)-1):
                    z['Z'+str(i)]=np.dot(w['W'+str(i)],a['A'+str(i-1)])+b['b'+str(i)]
                    a['A'+str(i)]=softmax(z['Z'+str(i)])
                else:
                    z['Z'+str(i)]=np.dot(w['W'+str(i)],a['A'+str(i-1)])+b['b'+str(i)]
                    a['A'+str(i)]=tanh1(z['Z'+str(i)])
            
            cost1=np.sum(y*np.log(a['A'+str(len(hidden)-1)]),axis=0)
            cost=np.sum(cost1)
            cost=(-1/500)*cost
            #print(cost)
            


            
            for i in range(1,len(hidden)):
                if i==1:
                    grad['dZ'+str(len(hidden)-i)]=a['A'+str(len(hidden)-i)]-y
                    grad['dA'+str(len(hidden)-i-1)]=np.dot(np.transpose(w['W'+str(len(hidden)-i)]),grad['dZ'+str(len(hidden)-i)])
                    grad['dW'+str(len(hidden)-i)]=np.dot(grad['dZ'+str(len(hidden)-i)],np.transpose(a['A'+str(len(hidden)-i-1)]))/500
                    grad['db'+str(len(hidden)-i)]=np.sum(grad['dZ'+str(len(hidden)-i)],axis=1)/500
                    w['W'+str(len(hidden)-i)]=w['W'+str(len(hidden)-i)]-eta*grad['dW'+str(len(hidden)-i)]
                    b['b'+str(len(hidden)-i)]=b['b'+str(len(hidden)-i)]-eta*np.reshape(grad['db'+str(len(hidden)-i)],(hidden[len(hidden)-i],1))
                else:
                    grad['dZ'+str(len(hidden)-i)]=grad['dA'+str(len(hidden)-i)]*tanh_der(z['Z'+str(len(hidden)-i)])
                    grad['dA'+str(len(hidden)-i-1)]=np.dot(np.transpose(w['W'+str(len(hidden)-i)]),grad['dZ'+str(len(hidden)-i)])
                    grad['dW'+str(len(hidden)-i)]=np.dot(grad['dZ'+str(len(hidden)-i)],np.transpose(a['A'+str(len(hidden)-i-1)]))/500
                    grad['db'+str(len(hidden)-i)]=np.sum(grad['dZ'+str(len(hidden)-i)],axis=1)/500
                    w['W'+str(len(hidden)-i)]=w['W'+str(len(hidden)-i)]-eta*grad['dW'+str(len(hidden)-i)]
                    b['b'+str(len(hidden)-i)]=b['b'+str(len(hidden)-i)]-eta*np.reshape(grad['db'+str(len(hidden)-i)],(hidden[len(hidden)-i],1))
            
            temp=temp+1    
                
               
        if j==20:
            eta=eta/2
        elif j==40:
            eta=eta/2

        
        print('j:',j)
        [val_acc,cost]=predict(xval,yval,w,b,hidden)
        [tra_acc,cost]=predict(xtr,ytr,w,b,hidden)
        val_acc=val_acc/100
        tra_acc=(tra_acc/50000)*100
        accuracy_val.append(val_acc)
        accuracy_tra.append(tra_acc)
        cost_final.append(cost)
        print ('Validation accuracy: ', val_acc)
        print ('Training accuracy: ', tra_acc)
    return (w,b,cost_final,grad,accuracy_val,accuracy_tra)
[w,b,cost_final,grad,accuracy_val,accuracy_tra]=model(xtr,ytr,[784,200,10],50,'tanh',0.65,0.01,yval,xval,500)
# =============================================================================
# print(cost_final)



plt.figure()
pylab.plot(accuracy_val,'-r',label='Validation Accuracy')
pylab.plot(accuracy_tra,'-b',label='Training Accuracy')
pylab.ylabel('Accuracy')
pylab.xlabel('Epochs')
pylab.title("eta=0.65 Activation=TanH")
pylab.legend(loc='upper left')
pylab.show()

# =============================================================================

DATA_FNAME = 'saloni_mlp_0_65_tanh.hdf5'
with h5py.File(DATA_FNAME, 'w') as hf:
	hf.attrs['act'] = np.string_("tanh")
	hf.create_dataset('w1',data=w["W1"])
	hf.create_dataset('b1',data=b["b1"])
	hf.create_dataset('w2',data=w["W2"])
	hf.create_dataset('b2',data=b["b2"])
# =============================================================================
# 	hf.create_dataset('w3',data=w["W3"])
# 	hf.create_dataset('b3',data=b["b3"])
# =============================================================================
	hf.close()
