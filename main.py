from Brain.neuron import neuron
from Brain.utils import Activation
from Brain.utils import Loss

neuron1 = neuron(2, Activation.basic, Loss.basic)

for i in range(50):
    neuron1.train([1, 1], 1)
    neuron1.train([1, 0], 1)
    neuron1.train([0, 1], 1)
    neuron1.train([0, 0], 0)

print("Test1 -> [0, 1]", neuron1.test([0, 1]))
print("Test2 -> [1, 0]", neuron1.test([1, 0]))
print("Test3 -> [1, 1]", neuron1.test([1, 1]))
print("Test4 -> [0, 0]", neuron1.test([0, 0]))