
#include <cmath>
#include <iostream>
// Define M_PI and M_E if they're not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

// Approximate the gamma function using Stirling's approximation
// Not accurate for small values
double gammaFunction(double n) {
    return sqrt(2 * M_PI / n) * pow((n / M_E), n);
}

// Numerical integration using the trapezoidal rule
double integrate(double a, double b, double(*func)(double, double, double), double shape, double scale, int steps = 10000) {
    double h = (b - a) / steps;
    double sum = 0.5 * (func(a, shape, scale) + func(b, shape, scale));
    for (int i = 1; i < steps; i++) {
        double x = a + i * h;
        sum += func(x, shape, scale);
    }
    return sum * h;
}

// Gamma PDF
double gammaPDF(double x, double shape, double scale) {
    if (x < 0) return 0.0;
    return pow(x, shape - 1) * exp(-x / scale) / (gammaFunction(shape) * pow(scale, shape));
}

// Gamma CDF approximation
double gammaCDF(double x, double shape, double scale, bool lower_tail = true, bool log_p = false) {
    double cdfValue = integrate(0, x, gammaPDF, shape, scale);
    
    if (!lower_tail) {
        cdfValue = 1.0 - cdfValue;
    }
    
    if (log_p) {
        cdfValue = log(cdfValue);
    }
    
    return cdfValue;
}