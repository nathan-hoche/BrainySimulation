import numpy as np

class Activation:
    def basic(x:float) -> float:
        x[x <= 0.5] = 0
        x[x > 0] = 1
        return x

    def reLu(x:float) -> float:
        return np.maximum(0, x)
    
    def leakyReLu(x:float) -> float:
        return np.where(x > 0, x, 0.01 * x)

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
        x = np.where(x > 0, x, 1.67326 * x)
        return 1.0507 * x
    
    def elu(x:float) -> float:
        x = np.where(x > 0, x, np.exp(x) - 1)
        return 1.0 * x
    
    def exponential(x:float) -> float:
        return np.exp(x)

class Derivative:
    def basic(x:float) -> float:
        x[x <= 0.5] = 0
        x[x > 0] = 1
        return x

    def reLu(x:float) -> float:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    
    def leakyReLu(x:float) -> float:
        x[x <= 0] = 0.01
        x[x > 0] = 1
        return x
    
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
        x[x <= 0] = 1.67326
        x[x > 0] = 1
        return 1.0507 * x
    
    def elu(x:float) -> float:
        x = np.where(x > 0, x, np.exp(x))
        x[x > 0] = 1
        return 1.0 * x
    
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
    def basic(velocity: list[float]|float, output:list[float], **kwargs) -> float|list[float]:
        return velocity * ACTIVATION_DERIVATIVE[kwargs["gradient"]](output)

MINIMAL_ZERO = np.array([1e-9])
ONE = np.array([1])

class Loss:
    def squaredError(output: list[float], outputExpected: float|list[float]) -> float:
        return np.sum(0.5 * (outputExpected - output) ** 2)
    
    def crossEntropy(output: list[float], outputExpected: float|list[float]) -> float:
        return -np.sum(outputExpected * np.log(output + 1e-9))
    
    def meanSquaredError(output: list[float], outputExpected: float|list[float]) -> float:
        return np.mean((outputExpected - output) ** 2)
    
    def meanAbsoluteError(output: list[float], outputExpected: float|list[float]) -> float:
        return np.mean(np.abs(outputExpected - output))

    def meanAbsolutePercentageError(output: list[float], outputExpected: float|list[float]) -> float:
        return np.mean(np.abs((outputExpected + MINIMAL_ZERO - output) / (outputExpected + MINIMAL_ZERO))) * 100
    
    def meanSquaredLogarithmicError(output: list[float], outputExpected: float|list[float]) -> float:
        return np.mean((np.log(output + ONE) - np.log(outputExpected + ONE)) ** 2)
    
    def squaredHinge(output: list[float], outputExpected: float|list[float]) -> float:
        return np.mean((1 - outputExpected * output) ** 2)
    
    def hinge(output: list[float], outputExpected: float|list[float]) -> float:
        return np.mean(1 - outputExpected * output)
    
    def categoricalHinge(output: list[float], outputExpected: float|list[float]) -> float:
        return np.mean(np.maximum(1 - outputExpected * output, 0))
    
    def logCosh(output: list[float], outputExpected: float|list[float]) -> float:
        return np.mean(np.log(np.cosh(output - outputExpected)))
    
    def categoricalCrossEntropy(output: list[float], outputExpected: float|list[float]) -> float:
        return -np.sum(outputExpected * np.log(output + 1e-9))

