from BrainV1.neuron import neuron
from BrainV1.utils import Activation
from BrainV1.utils import Gradient

class sample():
    def __init__(self) -> None:
        self.neuron = neuron(2, Activation.basic, Gradient.basic)

    def train(self, isTest:bool=False):
        if isTest == True:
            return
        print("Training Values:")
        print("\t[1, 1] -> 1")
        print("\t[1, 0] -> 1")
        print("\t[0, 1] -> 1")
        print("\t[0, 0] -> 0")
        for _ in range(1000):
            self.neuron.train([1, 1], 1)
            self.neuron.train([1, 0], 1)
            self.neuron.train([0, 1], 1)
            self.neuron.train([0, 0], 0)

    def test(self, isTest:bool=False):
        if isTest == True:
            return
        print("Result:")
        print("\tTest: [1, 1] ->", self.neuron.calc([1, 1]))
        print("\tTest: [1, 0] ->", self.neuron.calc([1, 0]))
        print("\tTest: [0, 1] ->", self.neuron.calc([0, 1]))
        print("\tTest: [0, 0] ->", self.neuron.calc([0, 0]))
