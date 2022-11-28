from Brain.utils import Activation
from Brain.utils import Gradient
from Brain.network import network
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
                try:
                    All_Test[int(file.replace(".txt", ''))] = inputList
                except:
                    All_Test[file.replace(".txt", '')] = inputList
    return All_Test

class sample():
    def __init__(self) -> None:
        self.inputList = readDir()
        self.network = network()
        self.network.addNewLayer(Activation.sigmoid, Gradient.basic, 10, 25)

    def train(self, isTest:bool=False):
        if isTest == True:
            return
        print("Training Values: Images from Test/numbers except broken images")
        print("Example: 3 ->", self.inputList[3])
        for _ in range(100):
            self.network.train(self.inputList[0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            self.network.train(self.inputList[1], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
            self.network.train(self.inputList[2], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
            self.network.train(self.inputList[3], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
            self.network.train(self.inputList[4], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
            self.network.train(self.inputList[5], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
            self.network.train(self.inputList[6], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
            self.network.train(self.inputList[7], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
            self.network.train(self.inputList[8], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
            self.network.train(self.inputList[9], [0 ,0, 0, 0, 0, 0, 0, 0, 0, 1])


    def test(self, isTest:bool=False):
        if isTest == True:
            return
        print("Result - Trained value:")
        for x in range(10):
            value = self.network.calc(self.inputList[x])
            print("\tTest:", x, "->", value.index(max(value)))
        print("Result - Untrained value:")
        print("Example of Broken: 3-broken ->", self.inputList["3-broken"])
        value = self.network.calc(self.inputList["3-broken"])
        print("\tTest: 3-broken ->", value.index(max(value)))
        value = self.network.calc(self.inputList["4-broken"])
        print("\tTest: 4-broken ->", value.index(max(value)))
        value = self.network.calc(self.inputList["6-broken"])
        print("\tTest: 6-broken ->", value.index(max(value)))