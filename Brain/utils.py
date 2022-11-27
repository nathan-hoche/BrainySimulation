import math
import numpy as np

# Pour en rajouter : https://www.v7labs.com/blog/neural-networks-activation-functions
class Activation:
    def basic(x:float) -> float:
        return 0 if x < 0 else 1

    def reLu(x:float) -> float:
        return max(0, x)

    def sigmoid(x:float) -> float:
        return 1 / (1 + np.exp(-x))
    
    def tanh(x:float) -> float:
        return np.tanh(x)
    
class Gradient:
    def basic(outputExpected:float, output:float, input:float) -> float:
        return (outputExpected - output) * input

    def reLu(outputExpected:float, output:float) -> float:
        return outputExpected - output