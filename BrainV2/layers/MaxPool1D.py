import numpy as np

class MaxPool1D():
    def __init__(self, nbInput, pool_size=2, strides=None, padding=False) -> None:
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

    def getPoolMax(self, inputList: list[float], x: int) -> float:
        res = []
        for i in range(self.pool_size):
            res.append(inputList[x+i])
        return max(res)

    def calc(self, inputList: list[float]) -> list[float]|float:
        self.output = []
        x = 0
        while x + self.pool_size -1 < len(inputList):
            self.output.append(self.getPoolMax(inputList, x))
            x += self.strides
        self.output = np.array(self.output)
        return self.output

