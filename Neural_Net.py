import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

def sigmoid(x):
    return (1/(1+np.exp(-x)))


def sigmoid_derv(x):
    return (sigmoid(x))*(1-sigmoid(x))



np.random.seed(0)
feature_set, labels = datasets.make_moons(100, noise=0.10)


plt.figure(figsize=(10,7))
plt.scatter(feature_set[:,0], feature_set[:,1], c=labels, cmap=plt.cm.winter)


weights_layers_input = []
labels = labels.reshape(100,1)


weights_layers_hidden = []

bias = np.random.rand(1)
lrn_rate = 0.2


hidden_layers = 3 # Refere-se a quantidade de camadas desejáveis
layer_counter = 0
nodes = 4
nodes_arrays = np.array([])
nodes_layers_arrays = [] # Armazena a array de cada node no hidden layer em questão

weights_layers_output = np.random.rand(nodes)
weights_layers_output = weights_layers_output.reshape(nodes, 1)

output_layer = np.array([])

derv_layers_arrays = [] #Armazena as derivadas referentes a cada peso

'''
A sequência abaixo gera os valores corrrespondentes aos pesos que serão
usados entre o input e a primeira camada hidden
'''


weight = np.random.rand(len(feature_set[0]), nodes) #Aqui é gerado os weights entre o input e a primeira camada hidden
weights_layers_input.append(weight)

nodes_arrays = np.dot(feature_set,weight)+bias


nodes_arrays = sigmoid(nodes_arrays)


nodes_layers_arrays.append(nodes_arrays)




'''
A sequência abaixo gera os valores correspondentes ao cálculo entre os pesos
das hidden layers em questão.
'''



for i in range(1, hidden_layers):
    weight = np.random.rand(nodes, nodes)
    weights_layers_hidden.append(weight)



for i in range(1, len(weights_layers_hidden)):
    
    nodes_arrays = np.dot(nodes_layers_arrays[-1+i],weights_layers_hidden[i])+bias
    
    
    nodes_arrays = sigmoid(nodes_arrays)
    

    nodes_layers_arrays.append(nodes_arrays)

    
output_layer_dot = np.dot(nodes_layers_arrays[-1], weights_layers_output) + bias

output_layer_sig = sigmoid(output_layer_dot)


error_cost = (1/len(nodes_layers_arrays[-1]))* sum(((output_layer_sig - labels)**2))
print(error_cost)


for i in range(len(nodes_layers_arrays)):
    layer_counter-=1
    
    dcost_da = (2/len(nodes_layers_arrays[layer_counter])) * sum((output_layer_sig - labels))
    da_dz = sigmoid_derv(output_layer_dot)
    dz_dw = nodes_layers_arrays[layer_counter].T
    
    dcost_w = np.dot(dz_dw, dcost_da*da_dz)
    
    weights_layers_hidden[layer_counter] = weights_layers_hidden[layer_counter] - (dcost_w*error_cost*lrn_rate)



























