import numpy as np

class Activation:
    def basic(x:float) -> float:
        return 0 if x < 0 else 1

    def reLu(x:float) -> float:
        return max(0, x)
    
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

class Gradient:
    def basic(outputExpected:float, output:float, input:float) -> float:
        return (outputExpected - output) * input

    def hebbian(outputExpected:float, output:float, input:float) -> float:
        return np.dot(outputExpected - output, input) * input
    
    def oja(outputExpected:float, output:float, input:float) -> float:
        x = np.dot(outputExpected - output, input)
        return x * input - x * x * output

    