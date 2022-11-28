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
    print("Success rate:\t\t", x/len(toTest) *100, "%")

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
        if (network1.calc(i["input"])[0] == i["value"]):
            x += 1
    print("Success rate:\t\t", x/len(toTest) *100, "%")

def patternRecognization(trainingTime:int):

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
                    try:
                        All_Test[int(file.replace(".txt", ''))] = inputList
                    except:
                        All_Test[file.replace(".txt", '')] = inputList
        return All_Test
    
    network1 = network()
    network1.addNewLayer(Activation.sigmoid, Gradient.basic, 10, 25)
    # Doit ajouter une couche de neuron pour les classification
    inputList = readDir()

    for _ in range(trainingTime):
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

    x = 0
    mx = 0
    for x in range(10):
        res = network1.calc(inputList[x])
        if (res.index(max(res)) == x):
            x += 1
        mx += 1
    print("Success rate:\t\t", x/mx * 100, "%", end="\n\t")

    x = 0
    res = network1.calc(inputList["3-broken"])
    if (res.index(max(res)) == 3):
        x += 1
    res = network1.calc(inputList["4-broken"])
    if (res.index(max(res)) == 4):
        x += 1
    res = network1.calc(inputList["6-broken"])
    if (res.index(max(res)) == 6):
        x += 1
    print("Broken Success rate:\t", x/3 * 100, "%")



print("Only a neuron:", end="\n\t")
onlyANeuron(1000, [{"input":[0, 1], "value": 1}, {"input":[1, 0], "value": 1}, {"input":[1, 1], "value": 1}, {"input":[0, 0], "value": 0}])
print("Simple network:", end="\n\t")
simpleNetwork(1000, [{"input":[0, 0, 0, 0], "value": 0}, {"input":[0, 0, 0, 1], "value": 1}, {"input":[0, 0, 1, 0], "value": 1}, {"input":[1, 1, 1, 1], "value": 1}, {"input":[1, 0, 0, 0], "value": 1}, {"input":[1, 0, 0, 1], "value": 1}])
print("Pattern recognization:", end="\n\t")
patternRecognization(100)