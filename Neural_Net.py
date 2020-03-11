import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets


def sigmoid(x):
    return (1/(1+np.exp(-x)))


def sigmoid_derv(x):
    return (sigmoid(x))*(1-sigmoid(x))


np.random.seed(15)
feature_set, labels = datasets.make_moons(100, noise=0.10)
plt.figure(figsize=(10,7))
plt.scatter(feature_set[:,0], feature_set[:,1], c=labels, cmap=plt.cm.winter)
labels = labels.reshape(100,1)



'''
    Foram definidos os valores básicos para que a iteração possa começar
'''

lr_rate = 0.45
bias = np.random.rand(1)
graph_error_cost = []
graph_error_counter = []

hidden_layers = 1
nodes_num = 4



hidden_neurons_dot = np.zeros([hidden_layers,  len(feature_set), nodes_num])
hidden_neurons_sig = np.zeros([hidden_layers,  len(feature_set), nodes_num])


input_weight = np.random.rand(nodes_num, len(feature_set[0]))
hidden_weight = np.random.rand(hidden_layers-1, nodes_num, nodes_num)


output_weight = np.random.rand(nodes_num, len(labels[0]))
output_neurons_dot = np.zeros([len(labels)])
output_neurons_sig = np.zeros([len(labels)])

'''
    FeedForward
'''
    
for k in range(20000):

    hidden_neurons_dot[0] = np.dot(feature_set, input_weight.T) + bias
    
    hidden_neurons_sig[0] = sigmoid(hidden_neurons_dot[0])
    
    
    
    for i in range(1, hidden_layers):
        
        hidden_neurons_dot[i] = np.dot(hidden_neurons_sig[i-1], hidden_weight[i-1]) + bias
        
        hidden_neurons_sig[i] = sigmoid(hidden_neurons_dot[i]) 


        
    
    output_neurons_dot = np.dot(hidden_neurons_sig[-1], output_weight) + bias
    
    output_neurons_sig = sigmoid(output_neurons_dot) 
    
    error_cost = (1/(len(output_neurons_sig)))*sum(((output_neurons_sig-labels)**2))
    
    graph_error_cost.append(error_cost)
    
    graph_error_counter.append(k)
    
    print(error_cost)
    
    '''
        
        BackForward
    
    '''
    
    dC_da_output = (output_neurons_sig-labels)
    
    da_dz_output = sigmoid_derv(output_neurons_dot)
    
    dz_dw_output = hidden_neurons_sig[-1]
    
    dC_dw_output = np.dot((dC_da_output*da_dz_output).T, dz_dw_output).T
    
    output_weight -= dC_dw_output * lr_rate 

    
    for i in range( -1, -hidden_layers, -1):
        
        dC_da_hidden = dC_da_output * da_dz_output * output_weight.T
        
        da_dz_hidden = sigmoid_derv(hidden_neurons_dot[i])
        
        dz_dw_hidden = hidden_neurons_sig[i]
        
        dC_dw_hidden = np.dot((dC_da_hidden*da_dz_hidden).T, dz_dw_hidden).T
    
        hidden_weight[i] -= dC_dw_hidden * lr_rate
    
    
    dC_da_input = dC_da_output * da_dz_output * output_weight.T
    
    da_dz_input = sigmoid_derv(hidden_neurons_dot[0])
    
    dz_dw_input = feature_set
    
    dC_dW_input = np.dot((dC_da_input*da_dz_input).T, dz_dw_input)
    
    input_weight -= dC_dW_input * lr_rate
    




plt.figure()
plt.plot(graph_error_counter, graph_error_cost, c='r')



#[0.28516335]

