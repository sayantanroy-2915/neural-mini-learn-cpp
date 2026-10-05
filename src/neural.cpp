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

// 	(i = j) ? (yi (1 - yi)) : (- yi * yj)
// Must call `dSoftmax(y)` not `dSoftmax(z)`
Matrix dSoftmax(Vector v) {
	size_t n = v.len();
	Matrix d(n, n);
	for (size_t i0 = 0; i0 < n; i0++) {
		float y = v.get(i0);
		d.set(i0, i0, y * (1 - y));
		for (size_t i1 = 0; i1 < i0; i1++) {
			float y = -(v.get(i0) * v.get(i1));
			d.set(i0, i1, y);
			d.set(i1, i0, y);
		}
	}
	return d;
}

// y = f(z)
Vector activate(Activation act, Vector v) {
	switch (act) {
		case Activation::RELU:
			return v.apply([](float x) -> float { return (x >= 0) ? x : 0.0; });
		case Activation::SIGMOID:
			return v.apply(sigmoid);
		case Activation::TANH:
			return v.apply(tanhf);
		case Activation::SOFTMAX:
			return softmax(v);
		case Activation::LINEAR:
			return Vector(v);
		}
	return v;	// For satisfaction
}

// f'(z) = dz/dy
Vector dActivate(Activation act, Vector v) {
	switch (act) {
		case Activation::RELU:
			return v.apply([](float x) -> float { return (x >= 0) ? 1.0 : 0.0; });
		case Activation::SIGMOID:
			return v.apply(sigmoid).apply([](float x) -> float { return x * (1.0 - x); });
		case Activation::TANH:
			return v.apply([](float x) -> float { return 1.0 - pow(tanhf(x), 2); });
		case Activation::LINEAR:
			return Vector(v.len(), []() -> float { return 1.0f; });
		}
	return v;	// For satisfaction
}

// ========================== LAYER ==========================

Layer::Layer(size_t nx_, size_t ny_, Activation act_) : nx(nx_), ny(ny_), act(act_), W(ny, nx, RNG), b(ny, RNG) {}

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

Vector Layer::dCost(Vector Y, Loss L) {
	assert(Y.len() == ny);
	auto oneminus = [](float x) -> float { return 1.0 - x; };
	switch (L) {
		case Loss::MSE:
			return (2.0 / ny) * (y - Y);
		case Loss::BCE:
			return -(Y / y) - y * Y.apply(oneminus) / y.apply(oneminus);
		case Loss::CCE:
			return -(Y / y);
	}
	return Vector(ny); // Blank Vector for satisfaction
}

Vector Layer::backprop(Vector dL_dy, float lr) {
	assert(dL_dy.len() == ny);
	// Error term (ny x 1)
	Vector d;
	if (act == Activation::SOFTMAX)
		d = dSoftmax(y) * dL_dy;	// (ny x ny) x (ny x 1)
	else
		d = dL_dy ^ dActivate(act, z);	// (ny x 1) xelem (ny x 1)
	// Weight gradient (ny x nx) = (ny x 1) x (1 x nx)
	Matrix dL_dW = (Matrix)d * (~x);
	// Bias gradient (ny x 1) = Error term
	// Backpropagated signal (nx x 1) = (nx x ny) x (ny x 1)
	Vector dL_dx = ~W * d;
	// Update
	W = W - lr * dL_dW;
	b = b - lr * d;
	// Return backpropagated signal
	return dL_dx;
}

// ========================== MODEL ==========================

Model::Model(std::vector<size_t> nodeCounts, std::vector<Activation> acts, Loss L_): L(L_) {
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