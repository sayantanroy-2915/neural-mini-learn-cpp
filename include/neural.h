#pragma once
#ifndef NEURAL_H
#define NEURAL_H

#include<vector>
#include<matrix.h>

typedef enum Activation {LINEAR, RELU, SIGMOID, TANH, SOFTMAX} Activation;
typedef enum Loss {MSE, BCE, CCE} Loss;

class Layer {
	friend class Model;
private:
	// Input length
	size_t nx;
	// Output length
	size_t ny;
	// Weights
	Matrix W;
	// Biases
	Vector b;
	// Input vector
	Vector x;
	// Output vector before activation
	Vector z;
	// Output vector after activation
	Vector y;
	// Activation function
	Activation act;
	Layer(size_t inputNodes, size_t outputNodes, Activation act_);
	void printWeights();
	Vector forward(Vector X);
	// dL/dy
	Vector dCost(Vector y, Loss L);
	// dL_dy (ny x 1) --> dL_dx (nx x 1)
	Vector backprop(Vector dL_dy, float lr);
};

class Model {
private:
	std::vector<Layer> layers;
	Loss L;
	Vector forward(Vector x);
public:
	Model(std::vector<size_t> nodeCounts, std::vector<Activation> acts, Loss L_);
	void printWeights();
	std::vector<std::vector<float>> predict(std::vector<std::vector<float>> x);
	void train(std::vector<std::vector<float>> x, std::vector<std::vector<float>> y, float lr, int epoch);
};

#endif