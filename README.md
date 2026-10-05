# Neural Network

A neural network framework built from scratch in C++, with the underlying matrix operations, forward propagation, backpropagation, activation functions, and training machinery implemented without external machine-learning libraries.

The project started as an attempt to understand what actually happens underneath a neural-network library rather than simply calling one.

## Features

- Custom `Matrix` and `Vector` classes
- Matrix arithmetic and multiplication
- Element-wise transformations
- Configurable feed-forward layers
- Forward propagation
- Backpropagation
- Gradient-descent training
- Multiple activation functions:
    - Sigmoid
    - Tanh
    - ReLU
    - Linear
    - Softmax (under progress)

- Configurable loss functions
- Random parameter initialization
- Custom assertions and debugging utilities
- No PyTorch, TensorFlow, Eigen, or other ML frameworks

## Experiments

The framework has been tested against progressively different learning problems:

| Problem       | Type                             | Result         |
| ------------- | -------------------------------- | -------------- |
| XOR           | Binary classification            | Successful     |
| `sin(x)`      | Periodic regression              | Successful     |
| `cos(x)`      | Periodic regression              | Successful     |
| `x²`          | Nonlinear regression             | Successful     |
| `x³`          | Nonlinear regression             | Successful     |
| `x⁴`          | Nonlinear regression             | Successful     |
| `x` with Tanh | Piecewise-function approximation | Successful     |
| `x` with ReLU | Piecewise-linear representation  | Near-exact fit |
| Iris          | Multi-class classification       | 100% accuracy  |

These experiments were used not only to demonstrate the network, but also to validate the underlying implementation and investigate issues such as gradient saturation, numerical instability, dimensional consistency, and activation-function behavior.

## Why build it from scratch?

High-level machine-learning libraries hide most of the mechanics involved in training a neural network. This project intentionally avoids that abstraction.

The goal is to understand and implement these operations directly rather than delegate them to an existing framework.

## Current status

The core feed-forward network and numerical machinery are functional.

There are improvements to be made and functions to be implemented.

The next stage is using the framework with the multiple open-source datasets and extending the architecture where necessary for multiclass classification.

This project is primarily an exercise in understanding neural-network internals, numerical computation, and C++ systems implementation.

> AI assistance: While AI was used as a technical discussion and debugging aid; the source code and implementation decisions were written by the author.
