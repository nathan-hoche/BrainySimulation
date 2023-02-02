from BrainV2.layers.Dense import Dense
import numpy as np
import matplotlib.pyplot as plt

class network():
    def __init__(self, learningRate=1.0) -> None:
        self.layers = []
        self.learningRate = learningRate
        self.isFirst = True
        self.losses = []
        pass

    def addNewLayer(self, actFunc, gradient, nbNeuron, nbInput=None) -> None:
        if nbInput is None:
            nbInput = self.layers[-1].size()
        newLayer = Dense(actFunc, gradient, nbNeuron, nbInput, self.learningRate, self.isFirst)
        self.layers.append(newLayer)
        self.isFirst = False

    def calc(self, inputList: list[float]) -> list[float]:
        output = inputList
        for layer in self.layers:
            output = layer.calc(output)
        return output

    def train(self, inputList: list[float], outputExpected: float|list[float]) -> None:
        output = self.calc(inputList)
        # Calculate the squared error
        loss = 0.5 * (outputExpected - output) ** 2
        # print(loss)
        self.losses.append(np.sum(loss))
        velocity = None
        for layer in reversed(self.layers):
            velocity = layer.updateWeight(outputExpected, velocity)
    
    def printLayers(self):
        print("Network:")
        x = 0
        for layer in self.layers:
            print("Layer", x, ":", layer)
            x += 1
    
    def displayLoss(self):
        plt.plot(self.losses)
        plt.show()

    def plot(self, h=0.01):
        """
        Generate plot of input data and decision boundary.
        """
        # setting plot properties like size, theme and axis limits
        # sns.set_style('darkgrid')
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
                f.append(0 if self.calc([x, y]) < 0.5 else 1)
            t.append(f)
        Z = np.array(t)

        # using the contourf function to create the plot
        plt.contourf(xx, yy, Z, colors=['red', 'green', 'green', 'blue'], alpha=0.4)
        plt.show()