import numpy as np

# Pour en rajouter : https://www.v7labs.com/blog/neural-networks-activation-functions
class Activation:
    def basic(x:float) -> float:
        return 0 if x < 0 else 1

    def reLu(x:float) -> float:
        return np.maximum(0, x)
    
    def leakyReLu(x:float) -> float:
        return 0.01 * x if x < 0 else x

    def sigmoid(x:float) -> float:
        return 1 / (1 + np.exp(-x))
    
    def tanh(x:float) -> float:
        return np.tanh(x)

    def softmax(x:float) -> float:
        return np.exp(x) / np.sum(np.exp(x))
    
    def softplus(x:float) -> float:
        return np.log(1 + np.exp(x))
    
    def softsign(x:float) -> float:
        return x / (1 + np.abs(x))
    
    def selu(x:float) -> float:
        return 1.0507 * (1.67326 * x if x < 0 else x)
    
    def elu(x:float) -> float:
        return 1.0 * (np.exp(x) - 1 if x < 0 else x)
    
    def exponential(x:float) -> float:
        return np.exp(x)

class Derivative:
    def basic(x:float) -> float:
        return 0
    
    def reLu(x:float) -> float:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    
    def leakyReLu(x:float) -> float:
        return 0.01 if x < 0 else 1
    
    def sigmoid(x:float) -> float:
        return x * (1 - x)
    
    def tanh(x:float) -> float:
        return 1 - x ** 2
    
    def softmax(x:float) -> float:
        return x * (1 - x)
    
    def softplus(x:float) -> float:
        return 1 / (1 + np.exp(-x))
    
    def softsign(x:float) -> float:
        return 1 / (1 + np.abs(x)) ** 2
    
    def selu(x:float) -> float:
        return 1.0507 * (1.67326 if x < 0 else 1)
    
    def elu(x:float) -> float:
        return 1.0 * (np.exp(x) if x < 0 else 1)
    
    def exponential(x:float) -> float:
        return np.exp(x)

ACTIVATION_DERIVATIVE = {
    Activation.basic: Derivative.basic,
    Activation.reLu: Derivative.reLu,
    Activation.leakyReLu: Derivative.leakyReLu,
    Activation.sigmoid: Derivative.sigmoid,
    Activation.tanh: Derivative.tanh,
    Activation.softmax: Derivative.softmax,
    Activation.softplus: Derivative.softplus,
    Activation.softsign: Derivative.softsign,
    Activation.selu: Derivative.selu,
    Activation.elu: Derivative.elu,
    Activation.exponential: Derivative.exponential
}
    
class Gradient:
    # def basic(outputExpected:float, output:float, input:float, learningRate: float, **kwargs) -> float:
    #     return learningRate * (outputExpected - output) * input, 0

    # def hebbian(outputExpected:float, output:float, input:float, learningRate: float, **kwargs) -> float:
    #     return learningRate * np.dot(outputExpected - output, input) * input, 0
    
    # def oja(outputExpected:float, output:float, input:float, learningRate: float, **kwargs) -> float:
    #     x = np.dot(outputExpected - output, input)
    #     return learningRate * (x * input - x * x * output), 0

    def sgd(velocity: list[float]|float, output:list[float], **kwargs) -> float|list[float]:
        return velocity * ACTIVATION_DERIVATIVE[kwargs["gradient"]](output)

