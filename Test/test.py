import sys
import os
sys.path.append(os.getcwd())

from Brain.neuron import neuron
from Brain.utils import Activation
from Brain.utils import Gradient
from Brain.network import network

###### NEURON ######
def onlyANeuron(trainingTime:int, toTest:list[dict]):
    neuron1 = neuron(2, Activation.basic, Gradient.basic)
    for i in range(trainingTime):
        neuron1.train([1, 1], 1)
        neuron1.train([1, 0], 1)
        neuron1.train([0, 1], 1)
        neuron1.train([0, 0], 0)

    #neuron1.printWeight()
    x = 0
    for i in toTest:
        if (neuron1.calc(i["input"]) == i["value"]):
            x += 1
    print("Success rate:", x/len(toTest) *100, "%")

def simpleNetwork(trainingTime:int, toTest:list[dict]):
    network1 = network()
    network1.addNewLayer(Activation.basic, Gradient.basic, 2, 4)
    network1.addNewLayer(Activation.basic, Gradient.basic, 1)
    for i in range(trainingTime):
        network1.train([1, 1, 1, 1], 1)
        network1.train([1, 0, 0, 0], 1)
        network1.train([0, 0, 0, 0], 0)
        network1.train([0, 0, 0, 1], 1)
        network1.train([0, 0, 1, 0], 1)

    x = 0
    for i in toTest:
        if (network1.calc(i["input"]) == i["value"]):
            x += 1
    print("Success rate:", x/len(toTest) *100, "%")


print("Only a neuron:", end="\n\t")
onlyANeuron(1000, [{"input":[0, 1], "value": 1}, {"input":[1, 0], "value": 1}, {"input":[1, 1], "value": 1}, {"input":[0, 0], "value": 0}])
print("Simple network:", end="\n\t")
simpleNetwork(1000, [{"input":[0, 0, 0, 0], "value": 0}, {"input":[0, 0, 0, 1], "value": 1}, {"input":[0, 0, 1, 0], "value": 1}, {"input":[1, 1, 1, 1], "value": 1}, {"input":[1, 0, 0, 0], "value": 1}, {"input":[1, 0, 0, 1], "value": 1}])