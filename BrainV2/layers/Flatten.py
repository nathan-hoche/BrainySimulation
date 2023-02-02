import numpy as np

class Flatten():
    def __init__(self, nbInput) -> None:
        self.output = None
        self.input = None
        self.nbInput = nbInput
        pass

    def __str__(self) -> str:
        return "Flatten Layer"

    def size(self) -> int:
        return self.nbInput

    ## Train ##

    def updateWeight(self, _, velocity:float=None) -> None:
        return velocity

    ## Evaluate ##

    def calc(self, inputList: list[float]) -> list[float]|float:
        self.input = np.array(inputList)
        self.output = self.input.flatten()
        return self.output

