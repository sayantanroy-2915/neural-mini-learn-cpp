#include<iostream>
#include<math.h>
#include"neural.h"

/**
* !Warning: This file is an experimantal sample. It is a guideline on how to make
* use of the model. Experimentation requires careful refactoring of this file.
*/
int main() {
	// float pi = 3.14159265f;
	std::vector<std::vector<float>> X; //= {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	std::vector<std::vector<float>> Y; //= {{0}, {1}, {1}, {0}};
	for (float x = -2.0f; x <= 2.0f; x+=0.25f) {
		float y = fabs(x);
		// std::cout << x << "\t" << y << "\n";
		X.push_back(std::vector<float>{x});
		Y.push_back(std::vector<float>{y});
	}
	// std::cout << "\n";
	float lr = 0.01f;
	std::vector<std::vector<float>> X1 = X; // {{3.15}, {-1.64}, {5.73}, {-10}};
	for (int i0 = 0; i0 < 1; i0++) {
		Model model = Model(std::vector<size_t>{X.at(0).size(), 8, 1}, std::vector<activation>{activation::RELU, activation::LINEAR}, loss::MSE);
		// model.printWeights();
		model.train(X, Y, lr, 20000);
		auto predicted = model.predict(X1);
		for (int i1 = 0; i1 < X1.size(); i1++)
			std::cout << "" << X1.at(i1).at(0) << "\t" << fabs(X1.at(i1).at(0)) << "\t" << predicted.at(i1).at(0) << "\n";
		// std::cout << "\n";
	}
}

// int main() {
// 	Matrix A({{1, 2, 3}, {2, -1, 0}});
// 	A.print();
// 	(~A).print();
// 	Vector B({1, 1});
// 	Matrix C = ~B * A;
// 	C.print();
// 	Vector D = ~(~B * A);
// 	D.print();
// 	Matrix E({{1, 3, 1}, {-2, -5, 3}});
// 	Matrix F = A / E;
// 	F.print();
// 	return 0;
// }