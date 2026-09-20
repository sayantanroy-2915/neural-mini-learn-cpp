#pragma once
#ifndef MATRIX_H
#define MATRIX_H

#include<iostream>
#include<functional>
#include<vector>

class Matrix {
	friend class Vector;

private:
	size_t rows, cols, elements;
	float *mat;
	size_t len() const;
	float get(size_t i) const;
	void set(size_t i, float value);
	// Swap helper
	friend void swap(Matrix& a, Matrix& b) noexcept;
public:
	Matrix();
	Matrix(size_t row_, size_t col_);
	// Copy constructor for deep copy
	Matrix(const Matrix &m);
	// Move constructor
	Matrix(Matrix &&m) noexcept;
	// Direct initialization
	Matrix(std::initializer_list<std::initializer_list<float>> array);
	// Convert std::vector
	Matrix(std::vector<std::vector<float>> vec);
	// Generate values via an RNG passed through parameters
	Matrix(size_t row_, size_t col_, std::function<float()> generator);
	// Convert to std::vector
	std::vector<std::vector<float>> toStdVectorRows();
	// Convert to std::vector
	std::vector<std::vector<float>> toStdVectorColumns();
	// Shallow copy
	Matrix& operator=(Matrix m);
	// Destructor
	~Matrix();
	// Returns an array[2]{rows, columns}
	size_t *dim() const;
	// Get element at index
	float get(size_t i, size_t j) const;
	// Insert/Assign element at index
	void set(size_t i, size_t j, float value);
	// Display the matrix in std::cout
	void print() const;
	// Matrix addition
	Matrix operator+(const Matrix& m) const;
	/* Element wise multiplication
	* Dot product*/
	Matrix operator^(const Matrix& m) const;
	// Matrix and scalar multiplication
	Matrix operator*(const float s) const;
	// Matrix and scalar multiplication
	friend Matrix operator*(const float s, const Matrix& m);
	// Negation
	Matrix operator-() const;
	// Matrix subtraction
	Matrix operator-(const Matrix& m) const;
	// Matrix multiplication
	Matrix operator*(const Matrix& m) const;
	// Element wise division
	Matrix operator/(const Matrix& m) const;
	// Element wise scalar division
	Matrix operator/(const float s);
	// Element wise scalar division
	friend Matrix operator/(const float s, const Matrix& m);
	// Power
	Matrix operator^(const int s) const;
	// Transpose
	Matrix operator~() const;
	// Apply a function on all the elements and return new matrix
	Matrix apply(std::function<float(float)> func) const;
};

class Vector: public Matrix {
public:
	Vector();
	Vector(size_t size);
	Vector(const Matrix &m);
	// Direct initialization
	Vector(std::initializer_list<float> array);
	// Convert std::vector
	Vector(std::vector<float> vec);
	// Generate values via an RNG passed through parameters
	Vector(size_t size, std::function<float()> generator);
	// Convert to std::vector
	std::vector<float> toStdVector();
	// Number of elements
	size_t len() const;
	// Get element at index
	float get(size_t i) const;
	// Insert/Assign element at index
	void set(size_t i, float value);
	// Display the vector in std::cout
	void print() const;
	// Dot product
	Vector operator*(const Vector &v) const;
	// Transpose
	Matrix operator~() const;
	// Sum
	float sum() const;
};

#endif