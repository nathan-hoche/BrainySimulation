# About this Project
This project is a simple way to make an artificial neural network. My objectif is to learn how big framework like Tensorflow or Pytorch work. I want to make a simple framework that can be used in a lot of different case.

# How to launch a sample:

1. Execute the following line:
```sh
python3 main.py [sample]
```
2. Be happy!!! :)

# List of content:

## BrainV1: perceptron

The first version of the brain is a simple perceptron using Hebbian or Oja learning rule. It cannot be used for more complexe problem (like XOR).
For the implematation, calculus are made without numpy. It's just a simple python list and classes for neurons, the objectif is to understand how the brain work.

> This version is a good start to understand how perceptron works. But it's not a good way to make a neural network. All implementation is made without any library, and the code is well developped to be understandable (but not optimized).

### Table of content:

| Type | Done | TODO |
| ------ | ------ | ------ |
| Learning rule | Hebbian, Oja | |
| Activation function | basic, reLu, leakyReLu, sigmoid, tanh, softmax, softplus, softsign, selu, elu, exponential | |
| Layer | Dense | |

### List of sample:

| Sample | Description |
| ------ | ------ |
| basicNeuron | A simple neuron with a Hebbian learning rule |
| basicNetwork | A simple network with a Hebbian learning rule |
| patternRecognition | A simple network with a Hebbian learning rule and a pattern recognition |

> This version is currently not maintained. Due to the fact the second version is more used.

## BrainV2: neural network

The second version of the brain is a neural network and can be used for deep learning. It can be used for more complexe problem (like XOR). The major difference is the used of gradient descent to train the network. For the implematation, calculus are made with numpy, all layers use numpy array.

### Table of content:

| Type | Done | TODO |
| ------ | ------ | ------ |
| Gradient Descent | Basic | sgd, adam, etc|
| Activation function | reLu, sigmoid | basic, leakyReLu, tanh, softmax, softplus, softsign, selu, elu, exponential |
| Layer | Dense | Flatten, Dropout, Conv2D, MaxPooling, etc |

> This version is currently in development.

# How to use this project:
If you want to use this project, there is two way to do it:
1. You can add in the folder 'sample' a new file with the name of your sample. You can copy/paste the file 'sample/basicNeuron.py' and change the content of the function.
2. You can replace the 'main.py' file by your own file. And just reproduce one of the sample's class method.

> In utils folder, you can find activated/gradient function. You can add your own network.