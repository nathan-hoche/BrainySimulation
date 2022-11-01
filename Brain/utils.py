import math

# Pour en rajouter : https://www.v7labs.com/blog/neural-networks-activation-functions
class Activation:
    def basic(x:float) -> float:
        return 0 if x < 0 else 1

    def reLu(x:float) -> float:
        return max(0, x)

    def sigmoid(x:float) -> float:
        return 1 / (1 + math.exp(-x))
    
    def tanh(x:float) -> float:
        return math.tanh(x)
    
class Loss:
    def basic(outputExpected:float, output:float) -> float:
        return outputExpected - output

    # Loss function for the sigmoid activation function
    def sigmoid(prediction:float, result:float) -> float:
        return -result * math.log(prediction) - (1 - result) * math.log(1 - prediction)
    
    # Loss function for the tanh activation function
    def tanh(prediction:float, result:float) -> float:
        return -result * math.log(prediction) + (1 - result) * math.log(1 - prediction)

    # Loss function for the ReLu activation function
    def reLu(prediction:float, result:float) -> float:
        return -result * math.log(prediction) + (1 - result) * math.log(1 - prediction)
