#pragma once
#ifndef NEURAL_H
#define NEURAL_H

#include<vector>
#include<matrix.h>

typedef enum activation {LINEAR, RELU, SIGMOID, TANH, SOFTMAX} activation;
typedef enum loss {MSE, BCE, CCE} loss;

class Layer {
	friend class Model;
private:
	size_t nx, ny;
	Matrix W;
	Vector b;
	Vector x, z, y;
	activation act;
	Layer(size_t inputNodes, size_t outputNodes, activation act_);
	void printWeights();
	Vector forward(Vector X);
	Vector dCost(Vector y, loss l);
	Vector backprop(Vector dl_do, float lr);
};

class Model {
private:
	std::vector<Layer> layers;
	loss L;
	Vector forward(Vector x);
public:
	Model(std::vector<size_t> nodeCounts, std::vector<activation> acts, loss L_);
	void printWeights();
	std::vector<std::vector<float>> predict(std::vector<std::vector<float>> x);
	void train(std::vector<std::vector<float>> x, std::vector<std::vector<float>> y, float lr, int epoch);
};

#endif