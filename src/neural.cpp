#include<iostream>
#include<array>
#include<random>
#include<cmath>
#include<cassert>
#include<matrix.h>
#include"neural.h"

float RNG() {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	// std::normal_distribution<float> dist(0.0, 0.5);
	std::uniform_real_distribution<float> dist(-1.0, 1.0);
	return dist(gen);
}

float sigmoid(float x) {
	return 1.0 / (1.0 + expf(-x));
}

Vector softmax(Vector v) {
	Vector exp_v = v.apply(expf);
	return exp_v / (exp_v.sum());
}

Vector activate(activation act, Vector v) {
	switch (act) {
		case activation::RELU:
			return v.apply([](float x) -> float { return (x >= 0) ? x : 0.0; });
		case activation::SIGMOID:
			return v.apply(sigmoid);
		case activation::TANH:
			return v.apply(tanhf);
		case activation::SOFTMAX:
			return softmax(v);
		case activation::LINEAR:
			return Vector(v);
		}
	return v;	// For satisfaction
}

Vector dActivate(activation act, Vector v) {
	switch (act) {
		case activation::RELU:
			return v.apply([](float x) -> float { return (x >= 0) ? 1.0 : 0.0; });
		case activation::SIGMOID:
			return v.apply(sigmoid).apply([](float x) -> float { return x * (1.0 - x); });
		case activation::TANH:
			return v.apply([](float x) -> float { return 1.0 - pow(tanhf(x), 2); });
		case activation::LINEAR:
			return Vector(v.len(), []() -> float { return 1.0f; });
		}
	return v;	// For satisfaction
}

// ========================== LAYER ==========================

Layer::Layer(size_t nx_, size_t ny_, activation act_) : nx(nx_), ny(ny_), act(act_), W(ny, nx, RNG), b(ny, RNG) {}

void Layer::printWeights() {
	std::cout << "Dimension: (" << nx << ", " << ny << ")\n";
	W.print();
	b.print();
	std::cout << "\n";
}

Vector Layer::forward(Vector x_) {
	assert(x_.len() == nx);
	x = x_;
	z = W * x + b;
	y = activate(act, z);
	return Vector(y);
}

Vector Layer::dCost(Vector Y, loss l) {
	assert(Y.len() == ny);
	switch (l) {
		case loss::MSE:
			return (2.0 / ny) * (y - Y);
		case loss::BCE:
			auto oneminus = [](float x) -> float { return 1.0 - x; };
			return -(Y / y) - y * Y.apply(oneminus) / y.apply(oneminus);
		}
	return Vector(ny);	// Blank Vector for satisfaction
}

Vector Layer::backprop(Vector dl_do, float lr) {
	assert(dl_do.len() == ny);
	// Error term (nx x 1)
	Vector d = dl_do * dActivate(act, z);
	// Weight gradient (ny x nx)
	Matrix dl_dW = (Matrix)d * (~x);
	// Bias gradient (ny x 1) = Error term
	// Backpropagated signal (nx x 1)
	Vector dl_dx = ~W * d;
	// Update
	W = W - lr * dl_dW;
	b = b - lr * d;
	// Return backpropagated signal
	return Vector(dl_dx);
}

// ========================== MODEL ==========================

Model::Model(std::vector<size_t> nodeCounts, std::vector<activation> acts, loss L_): L(L_) {
	size_t nLayers = acts.size();
	assert(nLayers > 0 && nLayers + 1 == nodeCounts.size());
	for (size_t i = 0; i < nLayers; i++)
		layers.push_back(Layer(nodeCounts.at(i), nodeCounts.at(i + 1), acts.at(i)));
}

void Model::printWeights() {
	for (Layer& layer: layers)
		layer.printWeights();
}

Vector Model::forward(Vector x) {
	Vector v = x;
	for (Layer& layer: layers)	// layer needs to be changed so &
		v = layer.forward(v);
	return v;
}

std::vector<std::vector<float>> Model::predict(std::vector<std::vector<float>> x) {
	std::vector<std::vector<float>> predicted;
	for (std::vector<float> v: x)
		predicted.push_back(forward(Vector(v)).toStdVector());
	return predicted;
}

void Model::train(std::vector<std::vector<float>> x, std::vector<std::vector<float>> y, float lr, int epoch) {
	assert(x.size() == y.size());
	for (int i0 = 0; i0 < epoch; i0++)
		for (int i1 = 0; i1 < x.size(); i1++) {
			forward(x.at(i1));
			auto dydw = layers.back().dCost(y.at(i1), L);
			for (auto layerit = layers.rbegin(); layerit != layers.rend(); layerit++)
				dydw = layerit->backprop(dydw, lr);
		}
}