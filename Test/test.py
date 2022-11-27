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

print("Only a neuron: ", end="")
onlyANeuron(1000, [{"input":[0, 1], "value": 1}, {"input":[1, 0], "value": 1}, {"input":[1, 1], "value": 1}, {"input":[0, 0], "value": 0}])