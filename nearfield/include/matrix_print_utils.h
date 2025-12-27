#ifndef MATRIX_PRINT_UTILS_H_
#define MATRIX_PRINT_UTILS_H_

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "complex_matrix_ops.cuh"

inline std::string FormatComplex(const Complex& value, double eps)
{
    double real = static_cast<double>(complex_real(value));
    double imag = static_cast<double>(complex_imag(value));

    if (std::abs(real) <= eps)
    {
        real = 0.0;
    }
    if (std::abs(imag) <= eps)
    {
        imag = 0.0;
    }

    std::ostringstream oss;
    oss << PRINT_PRECISION;

    if (imag == 0.0)
    {
        oss << real;
        return oss.str();
    }

    const double abs_imag = std::abs(imag);
    const bool imag_is_one = std::abs(abs_imag - 1.0) <= eps;

    if (real == 0.0)
    {
        if (imag < 0.0)
        {
            oss << "-";
        }
        if (imag_is_one)
        {
            oss << "i";
        }
        else
        {
            oss << abs_imag << "i";
        }
        return oss.str();
    }

    oss << real;
    if (imag > 0.0)
    {
        oss << "+";
    }
    else
    {
        oss << "-";
    }

    if (imag_is_one)
    {
        oss << "i";
    }
    else
    {
        oss << abs_imag << "i";
    }

    return oss.str();
}

inline void PrintMatrix(const std::vector<Complex>& matrix, int n, int k, const char* name)
{
    std::cout << name << " =\n";
    const int limit = k > 0 ? std::min(n, k) : n;
    const double eps = static_cast<double>(std::numeric_limits<Real>::epsilon()) * 10.0;
    std::vector<std::string> formatted(static_cast<size_t>(limit) * limit);
    std::vector<size_t> widths(static_cast<size_t>(limit), 0);

    for (int row = 0; row < limit; ++row)
    {
        for (int col = 0; col < limit; ++col)
        {
            const Complex value = matrix[col * n + row];
            std::string text = FormatComplex(value, eps);
            const size_t idx = static_cast<size_t>(col) * limit + row;
            formatted[idx] = std::move(text);
            widths[static_cast<size_t>(col)] =
                std::max(widths[static_cast<size_t>(col)], formatted[idx].size());
        }
    }

    std::cout << std::right;
    for (int row = 0; row < limit; ++row)
    {
        for (int col = 0; col < limit; ++col)
        {
            const size_t idx = static_cast<size_t>(col) * limit + row;
            std::cout << std::setw(static_cast<int>(widths[static_cast<size_t>(col)]))
                      << formatted[idx];
            if (col + 1 < limit)
            {
                std::cout << "  ";
            }
        }
        std::cout << "\n";
    }
}

inline void PrintMatrix(const std::vector<Complex>& matrix, int n, const char* name)
{
    PrintMatrix(matrix, n, 0, name);
}

#endif  // MATRIX_PRINT_UTILS_H_
