import numpy as np
import random

class Filter:
    def VertSobel():
        return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    
    def HorizSobel():
        return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    def Laplacian():
        return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    
    def Gaussian():
        return np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    
    def Identity():
        return np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    
    def Sharpen():
        return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    
    def Emboss():
        return np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    
    def BoxBlur():
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
    
    def EdgeDetect():
        return np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    
    def EdgeEnhance():
        return np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
    
    def MeanRemoval():
        return np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
    
    def MotionBlur():
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

def randomChooseFilter(size:int) -> np.array:
    ls = []
    filters = [Filter.VertSobel(), Filter.HorizSobel(), Filter.Laplacian(), Filter.Gaussian(), Filter.Identity(), Filter.Sharpen(), Filter.Emboss(), Filter.BoxBlur(), Filter.EdgeDetect(), Filter.EdgeEnhance(), Filter.MeanRemoval(), Filter.MotionBlur()]
    for _ in range(size):
        ls.append(random.choice(filters))
    return np.array(ls)

class Conv2D():
    def __init__(self, filter:list[object]|int, strides=None, padding=False) -> None:
        self.output = None
        self.input = None
        self.filter = filter if type(filter) == list else randomChooseFilter(filter)
        self.nbInput = len(filter) if type(filter) == list else filter
        self.strides = filter if strides is None else strides
        self.padding = padding
        pass
 
    def __str__(self) -> str:
        return "Conv2D"

    def size(self) -> int:
        return self.nbInput

    ## Train ##

    def updateWeight(self, _, velocity:float=None) -> None:
        return velocity

    ## Evaluate ##

    def getDotProduct(self, inputList: list[float], x: int, y: int, filterPos:int) -> float:
        res = 0
        for i in range(len(self.filter[filterPos])):
            for j in range(len(self.filter[filterPos][i])):
                res += inputList[y+i][x+j] * self.filter[filterPos][i][j]
        return res

    def calc(self, inputList: list[float]) -> list[float]|float:
        self.output = []
        filterPos = 0

        for inpt in inputList:
            self.output.append([])
            x = 0
            y = 0
            while y + len(self.filter[filterPos][0]) -1 < len(inpt):
                self.output[-1].append([])
                while x + len(self.filter[filterPos][1]) -1 < len(inpt[y]):
                    self.output[-1][-1].append(self.getDotProduct(inpt, x, y, filterPos))
                    x += self.strides[1]
                y += self.strides[0]
                x = 0
            filterPos += 1
        self.output = np.array(self.output)
        return self.output

