#include <../include/neuralNetwork.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>

double neuralNetwork::calcFastSigmoid(int x) {
    return ((double)x / (1 + abs(x)));
}

double neuralNetwork::calcSigmoid(int x) { return 1.0 / (1 + exp(-x)); }

double neuralNetwork::calcRelu(int x) { return std::max(0, x); }

double neuralNetwork::calcRelu6(int x) { return std::min(std::max(0, x), 6); }