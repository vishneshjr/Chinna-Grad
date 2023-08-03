import random
from chinna_grad.engine import Node


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
        
    def parameters(self):
        return []

class Neuron(Module):
    
    def __init__(self, inputSize):
        self.weights = [Node(random.uniform(-1, 1)) for i in range(inputSize)]
        self.bias = Node(random.uniform(-1, 1))
    
    def __call__(self, inputVector):
        outputNode = sum((weight * xVal for weight, xVal in zip(self.weights, inputVector)), self.bias)
        outputNode = outputNode.tanh()
        return outputNode

    def parameters(self):
        return self.weights + [self.bias]
    
class Layer(Module):
    def __init__(self, inputSize, outputSize):
        self.neurons = [Neuron(inputSize) for neuronNo in range(outputSize)]


    def __call__(self, inputVector):
        outputs = [neuron(inputVector) for neuron in self.neurons]
        if (len(outputs) == 1):
            return (outputs[0])
        return outputs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params
    
class MLP(Module):
    def __init__(self, inputSize, shape):
        finalShape = [inputSize] + shape
        self.layers = [Layer(finalShape[i], finalShape[i+1]) for i in range(len(shape))]
    
    def __call__(self, inputVector):
        for layer in self.layers:
            inputVector = layer(inputVector)
        outputVector = inputVector
        return outputVector
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params