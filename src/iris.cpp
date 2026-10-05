#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>
#include"neural.h"

class Iris {
	private:
		std::vector<std::vector<float>> X, Y;
		Model model;

	public:
	Iris() : model(Model(
			std::vector<size_t>{4, 8, 3},
			std::vector<Activation>{Activation::TANH, Activation::SOFTMAX},
			Loss::CCE
		)) {
		std::ifstream f("assets/iris/iris.data");
		if (!f.is_open()) {
			std::cerr << "Error opening file\n";
		}
		std::string line;
		std::vector<std::vector<float>> setosa;
		std::vector<std::vector<float>> versicolor;
		std::vector<std::vector<float>> virginica;
		while(std::getline(f, line)) {
			if (line.length() == 0)
				continue;
			std::stringstream ss(line);
			std::string value;
			float fvalue;
			std::vector<float> features;
			for (int i = 0; i < 4; i++) {
				std::getline(ss, value, ',');
				fvalue = atof(value.c_str());
				features.push_back(fvalue);
			}
			std::getline(ss, value);
			// std::cout << value << " ";
			if (value.compare("Iris-setosa") == 0)
				setosa.push_back(features);
			else if (value.compare("Iris-versicolor") == 0)
				versicolor.push_back(features);
			else if (value.compare("Iris-virginica") == 0)
				virginica.push_back(features);
			features.clear();
		}
		for (int i = 49; i >= 0; i--) {
			X.push_back(setosa.at(i));
			Y.push_back(std::vector<float>{1, 0, 0});
			X.push_back(versicolor.at(i));
			Y.push_back(std::vector<float>{0, 1, 0});
			X.push_back(virginica.at(i));
			Y.push_back(std::vector<float>{0, 0, 1});
		}
	}

	void evaluateModel() {
		int slice = (int)(0.70 * 150);
		auto trainingX = std::vector<std::vector<float>>(X.begin(), X.begin() + slice);
		auto testingX = std::vector<std::vector<float>>(X.begin() + slice, X.end());
		auto trainingY = std::vector<std::vector<float>>(Y.begin(), Y.begin() + slice);
		auto testingY = std::vector<std::vector<float>>(Y.begin() + slice, Y.end());
		model.train(trainingX, trainingY, 0.01, 10000);
		auto predictedY = model.predict(testingX);
		std::cout << "Actual\tPredicted\n";
		for (int i = 0; i < testingY.size(); i++) {
			std::cout << testingY.at(i).at(0) << " " << testingY.at(i).at(1) << " " << testingY.at(i).at(2) << "\t";
			std::cout << predictedY.at(i).at(0) << " " << predictedY.at(i).at(1) << " " << predictedY.at(i).at(2) << "\n";
		}
	}
};
