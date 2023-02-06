import numpy as np

class AveragePool2D():
    def __init__(self, nbInput, pool_size=(2, 2), strides=None, padding=False) -> None:
        self.output = None
        self.input = None
        self.nbInput = nbInput
        self.pool_size = pool_size
        self.strides= pool_size if strides is None else strides
        self.padding = padding
        pass
 
    def __str__(self) -> str:
        return "MaxPool2D"

    def size(self) -> int:
        return self.nbInput

    ## Train ##

    def updateWeight(self, _, velocity:float=None) -> None:
        return velocity

    ## Evaluate ##

    def getPoolMax(self, inputList: list[float], x: int, y: int) -> float:
        res = []
        for i in range(self.pool_size[0]):
            for j in range(self.pool_size[1]):
                res.append(inputList[y+i][x+j])
        return sum(res) / len(res)

    def calc(self, inputList: list[float]) -> list[float]|float:
        self.output = []
        x = 0
        y = 0 
        while y + self.pool_size[0] -1 < len(inputList):
            self.output.append([])
            while x + self.pool_size[1] -1 < len(inputList[y]):
                self.output[-1].append(self.getPoolMax(inputList, x, y))
                x += self.strides[1]
            y += self.strides[0]
            x = 0
        self.output = np.array(self.output)
        return self.output

