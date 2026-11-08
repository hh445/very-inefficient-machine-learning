import numpy as np
rng = np.random.default_rng()

layers = [784,128,64,10]

#since this neural net is using relu, so the outputs are numbers which arent contained by any activation function. softmax compresses it into probabilities that sum up to 1, depending on their difference. More positive closer to 1, basically more likely to be that choice
def softmax(z):
    #its kinda st andard to do z - np.max to decrease all the exponents by the maximum number. It helps keeps number bounded without them blowing up because of the huge exponents and does not change the outcome.
    ex = np.exp(z - np.max(z,axis = 0,keepdims= True))
    return ex/np.sum(ex)

#
def softmax_cel():
    pass
#im using kaiming initialization, since this neural network uses relu half of signals are dropped.
#intializing the weights with more variance prevents vanishing gradients (where some weights are never trained)
#x is number of neurons in the previous layer which kaiming intialization uses
#the random generation is a normal distribution around 0, with a variance of the kaiming thingy
weights = [(np.sqrt(2/x))*(np.random.randn(y,x)) for x,y in zip(layers[:-1],layers[1:])] #weight matrix for every layer, where the columns correspond to the amoutn of input vectors and rows correspond to amount of amount of output vectors (it maps high dimensional input vectors into a lower dimension output vector in my case.)
#biases are started off with 0 because the weights are already random. however, the weights cannot be 0 because if they are all of the outputs of the first layer would be zero because the input vectors are just multiplying by a zero matrix, so zero output.
biases = [np.zeros((x,1)) for x in layers[1:]]

#this function is the kind of the core of the neural net it is the function that takes in teh input and passes it through the network generating an output
#every layer the input vector is multiplied by the weight matrix, giving an output vector that is then added by the bias.
def forwardFeed(input):
    #the for loop iterate through all the layers, but it runs 1 less than the amoutn of layers because only n-1 calculations are needed, (the amoutn of gaps between layers)
    for i in range(len(layers)-1):
        if i < len(layers)-2:
            input = np.maximum(0,(np.dot(weights[i],input) + biases[i]))
        else:
            input = softmax(np.dot(weights[i],input) + biases[i])

    return input

print(forwardFeed( rng.random((784,1))))



 

