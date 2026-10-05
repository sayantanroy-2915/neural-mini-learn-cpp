#include<iostream>
#include<cassert>
#include"matrix.h"

#ifndef ASSERT
#define ASSERT(cond, action) do { if (!(cond)) { action; assert(cond); } } while (false);
#endif

Matrix::Matrix() : rows(0), cols(0), elements(0), mat(nullptr) {};

Matrix::Matrix(size_t row_, size_t col_) :
	rows(row_), cols(col_), elements(rows * cols),
	mat(new float[elements]) {}

Matrix::Matrix(const Matrix &m) :
	rows(m.rows), cols(m.cols), elements(rows * cols),	mat(new float[elements]) {
	for (int i = 0; i < elements; i++)
		set(i, m.get(i));
}

Matrix::Matrix(Matrix &&m) noexcept :
	rows(m.rows), cols(m.cols), elements(m.elements), mat(m.mat) {
	m.rows = m.cols = m.elements = 0;
	m.mat = nullptr;
}

Matrix::Matrix(std::initializer_list<std::initializer_list<float>> array) :
	rows(array.size()),
	cols(array.begin()->size()),
	elements(rows * cols),
	mat(new float[elements]) {
	size_t i = 0;
	for (const auto& row: array)
		for (float value: row)
			set(i++, value);
}

Matrix::Matrix(std::vector<std::vector<float>> vec) :
	rows(vec.size()),
	cols(vec.at(0).size()), 
	elements(rows * cols),
	mat(new float[elements]) {
	size_t i = 0;
	for (auto row : vec)
		for (float value : row)
			set(i++, value);
}

Matrix::Matrix(size_t row_, size_t col_, std::function<float()> generator)  :
	rows(row_), cols(col_),
	elements(rows * cols),
	mat(new float[elements]) {
		for (size_t i = 0; i < elements; i++)
			set(i, generator());
}

std::vector<std::vector<float>> Matrix::toStdVectorRows() {
	std::vector<std::vector<float>> vec;
	for (int i0 = 0; i0 < rows; i0++) {
		std::vector<float> row;
		for (int i1 = 0; i1 < cols; i1++)
			row.push_back(get(i0, i1));
		vec.push_back(row);
	}
	return vec;
}

std::vector<std::vector<float>> Matrix::toStdVectorColumns() {
	std::vector<std::vector<float>> vec;
	for (int i0 = 0; i0 < cols; i0++) {
		std::vector<float> col;
		for (int i1 = 0; i1 < rows; i1++)
			col.push_back(get(i1, i0));
		vec.push_back(col);
	}
	return vec;
}

void swap(Matrix &a, Matrix &b) noexcept {
	std::swap(a.rows, b.rows);
	std::swap(a.cols, b.cols);
	std::swap(a.elements, b.elements);
	std::swap(a.mat, b.mat);
}

Matrix& Matrix::operator=(Matrix m) {
	swap(*this, m);
	return *this;
}

Matrix::~Matrix() {
	delete[] mat;
}

size_t Matrix::len() const {
	return elements;
}

float Matrix::get(size_t i) const {
	return mat[i];
}

void Matrix::set(size_t i, float value) {
	mat[i] = value;
}

size_t* Matrix::dim() const {
	return new size_t[]{rows, cols};
}

float Matrix::get(size_t i, size_t j) const {
	ASSERT((i < rows && j < cols), std::cerr << "(" << i << ", " << j << ") (" << rows << ", " << cols << ")\n");
	return mat[i * cols + j];
}

void Matrix::set(size_t i, size_t j, float value) {
	ASSERT((i < rows && j < cols), std::cerr << "(" << i << ", " << j << ") (" << rows << ", " << cols << ")\n");
	mat[i * cols + j] = value;
}

void Matrix::print() const {
	for (size_t i0 = 0; i0 < rows; i0++) {
		for (size_t i1 = 0; i1 < cols; i1++)
			std::cout << "\t" << get(i0, i1);
		std::cout << "\n";
	}
}

Matrix Matrix::operator+(const Matrix &m) const {
	size_t* m_dim = m.dim();
	ASSERT((rows == m_dim[0] && cols == m_dim[1]), std::cerr << "(" << rows << ", " << cols << ") (" << m_dim[0] << ", " << m_dim[1] << ")\n");
	Matrix c = Matrix(rows, cols);
	for (size_t i = 0; i < elements; i++)
			c.set(i, get(i) + m.get(i));
	delete m_dim;
	return c;
}

Matrix Matrix::operator^(const Matrix &m) const {
	size_t* m_dim = m.dim();
	ASSERT((rows == m_dim[0] && cols == m_dim[1]), std::cerr << "(" << rows << ", " << cols << ") (" << m_dim[0] << ", " << m_dim[1] << ")\n");
	Matrix c = Matrix(rows, cols);
	for (size_t i = 0; i < elements; i++)
		c.set(i, get(i) * m.get(i));
	delete m_dim;
	return c;
}

Matrix Matrix::operator*(const float s) const {
	Matrix b = Matrix(rows, cols);
	for (size_t i = 0; i < elements; i++)
		b.set(i, s * get(i));
	return b;
}

Matrix operator*(const float s, const Matrix &m) {
	return m * s;
}

Matrix Matrix::operator-() const {
	return -1 * (*this);
}

Matrix Matrix::operator-(const Matrix &m) const {
	return (*this) + -m;
}

Matrix Matrix::operator*(const Matrix &m) const {
	size_t* m_dim = m.dim();
	ASSERT((cols == m_dim[0]), std::cerr << "(" << rows << ", " << cols << ") (" << m_dim[0] << ", " << m_dim[1] << ")\n");
	Matrix c = Matrix(rows, m_dim[1]);
	for (size_t i0 = 0; i0 < rows; i0++)
		for (size_t i1 = 0; i1 < m_dim[1]; i1++) {
			c.set(i0, i1, 0);
			for (size_t i2 = 0; i2 < cols; i2++)
				c.set(i0, i1, c.get(i0, i1) + get(i0, i2) * m.get(i2, i1));
		}
	delete m_dim;
	return c;
}

Matrix Matrix::operator/(const Matrix &m) const {
	size_t* m_dim = m.dim();
	ASSERT((rows == m_dim[0] && cols == m_dim[1]), std::cerr << "(" << rows << ", " << cols << ") (" << m_dim[0] << ", " << m_dim[1] << ")\n");
	Matrix c = Matrix(rows, cols);
	for (size_t i = 0; i < elements; i++) {
		float d = m.get(i);
		assert(d != 0.0);
		c.set(i, get(i) / d);
	}
	delete m_dim;
	return c;
}

Matrix Matrix::operator/(const float s) {
	assert(s != 0.0);
	Matrix b = Matrix(rows, cols);
	for (size_t i = 0; i < elements; i++)
		b.set(i, get(i) / s);
	return b;
}

Matrix operator/(const float s, const Matrix &m) {
	Matrix b = Matrix(m.rows, m.cols);
	for (size_t i = 0; i < m.elements; i++) {
		float d = m.get(i);
		assert(d != 0.0);
		b.set(i, s / d);
	}
	return b;
}

Matrix Matrix::operator^(const int s) const {
	ASSERT((rows == cols), std::cerr << "(" << rows << ", " << cols << ")\n");
	Matrix c = Matrix(rows, cols);
	for (int i = 0; i <= s; i++)
		c = c * (*this);
	return c;
}

Matrix Matrix::operator~() const {
	Matrix m = Matrix(cols, rows);
	for (size_t i0 = 0; i0 < rows; i0++)
		for (size_t i1 = 0; i1 < cols; i1++)
			m.set(i1, i0, get(i0, i1));
	return m;
}

Matrix Matrix::apply(std::function<float(float)> func) const {
	Matrix m = Matrix(rows, cols);
	for (size_t i = 0; i < elements; i++)
		m.set(i, func(get(i)));
	return m;
}

// ========================== VECTOR ==========================

Vector::Vector() : Matrix() {}

Vector::Vector(size_t size) : Matrix(size, 1) {};

Vector::Vector(const Matrix& m) {
	size_t* m_dim = m.dim();
	ASSERT((m_dim[1] == 1), std::cerr << "(" << m_dim[0] << ", " << m_dim[1] << ")\n");
	rows = elements = m_dim[0];
	cols = 1;
	mat = new float[elements];
	for (size_t i = 0; i < elements; i++)
		set(i, m.get(i, 0));
	delete m_dim;
};

Vector::Vector(std::initializer_list<float> array) {
	rows = elements = array.size();
	cols = 1;
	mat = new float[elements];
	size_t i = 0;
	for (const auto& value: array)
		set(i++, value);
}

Vector::Vector(std::vector<float> vec) {
	rows = elements = vec.size();
	cols = 1;
	mat = new float[elements];
	size_t i = 0;
	for (float value: vec)
		set(i++, value);
}

Vector::Vector(size_t size, std::function<float()> generator) : Matrix(size, 1, generator) {}

std::vector<float> Vector::toStdVector() {
	std::vector<float> vec;
	for (int i = 0; i < elements; i++)
		vec.push_back(get(i));
	return vec;
}

size_t Vector::len() const {
	return elements;
}

float Vector::get(size_t i) const {
	ASSERT((i < elements), std::cerr << i << " " << elements << "\n");
	return Matrix::get(i);
}

void Vector::set(size_t i, float value) {
	Matrix::set(i, value);
}

void Vector::print() const {
	for (size_t i = 0; i < elements; i++)
		std::cout << "\t" << get(i);
	std::cout << "\n";
}

float Vector::sum() const {
	float s = 0.0f;
	for (size_t i = 0; i < elements; i++)
		s += get(i);
	return s;
}

float Vector::operator*(const Vector &v) const {
	return ((Vector)((*this) ^ v)).sum();
}

Matrix Vector::operator~() const {
	Matrix m(1, elements);
	for (int i = 0; i < elements; i++)
		m.set(i, get(i));
	return m;
}
