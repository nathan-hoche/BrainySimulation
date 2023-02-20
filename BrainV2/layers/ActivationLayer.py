

class ActivationLayer():
    def __init__(self, nbInput, func) -> None:
        self.output = None
        self.nbInput = nbInput
        self.input = None
        self.func = None
        pass
 
    def __str__(self) -> str:
        return "ActivationLayer:" + self.func.__name__

    def size(self) -> int:
        return self.nbInput

    ## Train ##

    def updateWeight(self, _, velocity:float=None) -> None:
        return velocity

    ## Evaluate ##

    def calc(self, inputList: list[float]) -> list[float]|float:
        self.input = inputList
        self.output = self.func(inputList)
        return self.output

