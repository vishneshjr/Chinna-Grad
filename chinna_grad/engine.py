import math
# Defining the Node Class that stores a value
class Node:
    nameLetterVal = 65  # Stores ASCII Value of name for new Node
    def __init__(self, data = 0.0, name = None, _children = (), _op = None):
        name = chr(Node.nameLetterVal)
        Node.nameLetterVal += 1                 # Update Name
        self.name = name
        self.data = data
        self._backward = lambda : None
        self._prev = set(_children)
        self._op = _op
        self.grad = 0.0
    
    def __repr__(self):
        return f"{self.name} : (Value = {self.data})"
    
    def __neg__(self):              # -self
        return self * -1
    
    def __sub__(self, other):       # self - other
        return self + (-other)
    
    def __add__(self, other):       # self + other
        other = other if isinstance(other, Node) else Node(other)           # Handle edge case when you try to add with integers
        outputData = self.data + other.data
        outputNode = Node(outputData, _children = (self, other), _op = "+")
        def _backward():
            self.grad += 1.0 * outputNode.grad
            other.grad += 1.0 * outputNode.grad
        outputNode._backward = _backward
        return outputNode
    

    
    def __mul__(self, other):       # self * other
        other = other if isinstance(other, Node) else Node(other)
        outputData = self.data * other.data
        outputNode = Node(outputData, _children = (self, other), _op = "*")
        def _backward():
            self.grad += float(other.data) * outputNode.grad
            other.grad += float(self.data) * outputNode.grad
        outputNode._backward = _backward
        return outputNode
    
    # Fallback method which is implicitly called after swapping the order of the operands if multiplication fails
    def __rmul__(self, other):      # other * self
        return self * other
    

    def __truediv__(self, other):   # self / other
        return self * (other ** -1)
    
    def __pow__(self, other):       # self ** (other)
        assert isinstance(other, (int, float)) # Only supports when the power is an int/float
        outputData = self.data ** other
        outputNode = Node(outputData, _children=(self, ), _op = "**")

        def _backward():
            self.grad += (other) * (self.data ** (other - 1)) * (outputNode.grad)
        outputNode._backward = _backward

        return outputNode
    
    def exp(self):      # e ** (self)
        outputData = math.exp(self.data)
        outputNode = Node(outputData, _children = (self, ), _op = "exp")

        def _backward():
            self.grad += (outputData) * outputNode.grad
        outputNode._backward = _backward
        return outputNode
    
    
    def tanh(self):
        outputData = (math.exp(2*self.data) - 1.0) / (math.exp(2*self.data) + 1.0)
        outputNode = Node(outputData, _children = (self, ), _op = "tanh")
        def _backward():
            self.grad += (1 - (outputData**2)) * outputNode.grad
        outputNode._backward = _backward
        return outputNode

    def backward(self):
        # Topological sort (To ensure we are backpropagating in the correct order)
        topo = []
        visited = set()
        def build_topological_graph(node):
            if (node not in visited):
                visited.add(node)
            for child in node._prev:
                build_topological_graph(child)
            topo.append(node)
        build_topological_graph(self)
        topo = topo[::-1]

        self.grad = 1
        for node in topo:
            node._backward()