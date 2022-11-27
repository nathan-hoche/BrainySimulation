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
print("Test 0 ->", network1.calc(inputList[0]))
print("Test 1 ->", network1.calc(inputList[1]))
print("Test 2 ->", network1.calc(inputList[2]))
print("Test 3 ->", network1.calc(inputList[3]))
print("Test 4 ->", network1.calc(inputList[4]))
print("Test 5 ->", network1.calc(inputList[5]))
print("Test 6 ->", network1.calc(inputList[6]))
print("Test 7 ->", network1.calc(inputList[7]))
print("Test 8 ->", network1.calc(inputList[8]))
print("Test 9 ->", network1.calc(inputList[9]))
