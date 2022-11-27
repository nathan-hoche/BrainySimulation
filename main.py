from Brain.neuron import neuron
from Brain.utils import Activation
from Brain.utils import Gradient
from Brain.network import network


##### Number Identification ######
import os

def readDir():
    dir = os.listdir("Test/numbers")
    All_Test = {}
    for file in dir:
        if file.find(".txt") != -1:
            with open("Test/numbers/" + file, "r") as f:
                fc = f.read().replace("_", "0").replace("#", "1").replace("\n", '')
                inputList = [*fc]
                for i in range(len(inputList)):
                    inputList[i] = int(inputList[i])
                All_Test[int(file.replace(".txt", ''))] = inputList
    return All_Test

# https://www.frontiersin.org/articles/10.3389/fnins.2021.690418/full

network1 = network()
network1.addNewLayer(Activation.sigmoid, Gradient.basic, 10, 25)
# Doit ajouter une couche de neuron pour les classification
inputList = readDir()

for i in range(100):
    network1.train(inputList[0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    network1.train(inputList[1], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    network1.train(inputList[2], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    network1.train(inputList[3], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    network1.train(inputList[4], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    network1.train(inputList[5], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    network1.train(inputList[6], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    network1.train(inputList[7], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    network1.train(inputList[8], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    network1.train(inputList[9], [0 ,0, 0, 0, 0, 0, 0, 0, 0, 1])

network1.printWeight()
for x in range(10):
    res = network1.calc(inputList[x])
    print("Test", x, ":", res.index(max(res)))
