from Brain.neuron import neuron
from Brain.utils import Activation
from Brain.utils import Loss
from Brain.network import network

# neuron1 = neuron(2, Activation.basic, Loss.basic)

# for i in range(50):
#     neuron1.train([1, 1], 1)
#     neuron1.train([1, 0], 1)
#     neuron1.train([0, 1], 1)
#     neuron1.train([0, 0], 0)

# print("Test1 -> [0, 1]", neuron1.test([0, 1]))
# print("Test2 -> [1, 0]", neuron1.test([1, 0]))
# print("Test3 -> [1, 1]", neuron1.test([1, 1]))
# print("Test4 -> [0, 0]", neuron1.test([0, 0]))

network1 = network()
network1.addNewLayer(Activation.basic, Loss.basic, 2, 4)
network1.addNewLayer(Activation.basic, Loss.basic, 1)

for i in range(1000):
    network1.train([1, 1, 1, 1], 1)
    network1.train([1, 0, 0, 0], 1)
    network1.train([0, 0, 0, 0], 0)
    network1.train([0, 0, 1, 0], 1)

print("Test1 -> [0, 0, 0, 0]", network1.calc([0, 0, 0, 0]))
print("Test1 -> [0, 1, 0, 0]", network1.calc([0, 1, 0, 0]))
print("Test1 -> [1, 0, 0, 1]", network1.calc([1, 0, 0, 1]))
print("Test1 -> [1, 1, 1, 0]", network1.calc([1, 1, 1, 0]))