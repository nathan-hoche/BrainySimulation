from BrainV2.utils import Activation
from BrainV2.utils import Gradient
from BrainV2.network import network
from matplotlib import pyplot as plt
import numpy as np

def plot(func, h=0.01):
        plt.figure(figsize=(20, 20))
        plt.axis('scaled')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        colors = {
            0: "ro",
            1: "go"
        }
        data = [[0, 0], [0, 1], [1, 0], [1, 1]]
        target = [[0], [1], [1], [0]]

        # plotting the four datapoints
        for i in range(len(data)):
            plt.plot([data[i][0]],
                     [data[i][1]],
                     colors[target[i][0]],
                     markersize=20)
        x_range = np.arange(-0.1, 1.1, h)
        y_range = np.arange(-0.1, 1.1, h)

        # creating a mesh to plot decision boundary
        xx, yy = np.meshgrid(x_range, y_range, indexing='ij')
        t = []
        for x in x_range:
            f = []
            for y in y_range:
                f.append(0 if func([x, y]) < 0.5 else 1)
            t.append(f)
        Z = np.array(t)

        # using the contourf function to create the plot
        plt.contourf(xx, yy, Z, colors=['red', 'green', 'green', 'blue'], alpha=0.4)
        plt.show()

class sample():
    def __init__(self) -> None:
        self.XOR = network(learningRate=0.1)
        self.XOR.addNewLayer(Activation.sigmoid, Gradient.basic, 16, 2)
        self.XOR.addNewLayer(Activation.sigmoid, Gradient.basic, 1)
        self.XOR.printLayers()

    def train(self, isTest:bool=False):
        if isTest == True:
            return
        for _ in range(10000):
            self.XOR.train([[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]])
        self.XOR.displayLoss()

    def test(self, isTest:bool=False):
        if isTest == True:
            return
        print("Result - XOR:")
        print("\tTest: [1, 1] -> ", self.XOR.calc([1, 1]))
        print("\tTest: [0, 0] -> ", self.XOR.calc([0, 0]))
        print("\tTest: [1, 0] -> ", self.XOR.calc([1, 0]))
        print("\tTest: [0, 1] -> ", self.XOR.calc([0, 1]))

        plot(self.XOR.calc)

