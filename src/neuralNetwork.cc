#include <layer.h>
#include <neuralNetwork.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <eigen3/Eigen/Dense>

/*---------------------------------------------------------------------*/
/*                        Private                                      */
/*---------------------------------------------------------------------*/

/**
 * Calculates activation using fast sigmoid which is an (easy
 * to calculate) approximation of the actual sigmoid function.
 *
 * @param x f(x).
 * @return The activation y = f(x).
 */
double NeuralNetwork::calcFastSigmoid(double x) {
    return ((double)x / (1 + abs(x)));
}

/**
 * Calculates activation using the classic sigmoid function.
 *
 * @param x f(x).
 * @return The activation y = f(x).
 */
double NeuralNetwork::calcSigmoid(double x) { return 1.0 / (1 + exp(-x)); }

/**
 * Calculates activation using ReLu.
 *
 * @param x f(x).
 * @return The activation y = f(x).
 */
double NeuralNetwork::calcReLu(double x) { return std::max(0.0, x); }

/**
 * Calculates activation using Relu6 which is Relu but for
 * all x > 6 the function value is 6.
 *
 * @param x f(x).
 * @return The activation y = f(x).
 */
double NeuralNetwork::calcReLu6(double x) {
    return std::min((double)std::max(0.0, x), 6.0);
}

/**
 * Initializes the network.
 * Gets called by all constructors.
 */
void NeuralNetwork::initialize() {
    // initialize the layers
    inputLayer = new Layer(inNeurons);
    hiddenLayer = new Layer(hiddenNeurons);
    outputLayer = new Layer(outNeurons);

    // initialize the weight matrices
    inToHidden = new Connection((int)inNeurons, (int)hiddenNeurons);
    hiddenToOut = new Connection((int)hiddenNeurons, (int)outNeurons);

    learend = false;
};

/*---------------------------------------------------------------------*/
/*                        Public                                       */
/*---------------------------------------------------------------------*/

/**
 * Calculates the activation for an input x using the networks activation
 * function which can be configured using setCurrentActivationFunction.
 *
 * @param x The input value for the activation functions in a fully connected NN
 *          is the sum of the values of all previous neurons times the
 *          connections weight.
 * @return The value of the activation function.
 */
double NeuralNetwork::calcActivation(double x) {
    switch (currentActivationFunction) {
        case sigmoid:
            return calcSigmoid(x);
            break;
        case fastSigmoid:
            return calcFastSigmoid(x);
            break;
        case ReLu:
            return calcReLu(x);
            break;
        case ReLu6:
            return calcReLu6(x);
            break;
        default:
            throw std::invalid_argument(
                "Was not one of Sigmoid, fast Sigmoid, ReLu or ReLu6.");
            break;
    }
};

/**
 * Calculates the energy using the set of ground truth vs actual output values.
 * Notice: groundTruth and netOutput should have the same length!
 *
 * @param groundTruth Vector containing ground truth examples.
 * @param netOutput Vector containing actual network output.
 * @return The energy.
 */
double NeuralNetwork::calcEnergy(Eigen::VectorXd groundTruth,
                                 Eigen::VectorXd netOutput) {
    double energy = 0;
    for (int i = 0; i < (int)netOutput.size(); i++) {
        energy +=
            (groundTruth(i) - netOutput(i)) * (groundTruth(i) - netOutput(i));
    }

    return energy /= 2.0;
}

/**
 * Propagates the current network input trough the network.
 */
void NeuralNetwork::propagate() {
    inputLayer->setThreshold(1);

    for (int i = 0; i < (int)hiddenNeurons; i++) {
        double net = 0;
        for (int j = 0; j < (int)inNeurons; j++) {
            net += inToHidden->getWeights(j, i) * inputLayer->getData(j);
            hiddenLayer->setData(i, calcActivation(net));
        }
    }

    for (int i = 0; i < (int)outNeurons; i++) {
        double net = 0;
        for (int j = 0; j < (int)hiddenNeurons; j++) {
            net += hiddenToOut->getWeights(j, i) * hiddenLayer->getData(j);
            outputLayer->setData((int)i, calcActivation(net));
        }
    }
}

/**
 * Recalculates all weights by comparing the network data with an example.
 * This resembles one learning step!
 *
 * @param teach A vector containing the example. Must be the same length as the
 *              output layer.
 */
void NeuralNetwork::backpropagate(Eigen::VectorXd teach) {
    Eigen::VectorXd deltaH(hiddenLayer->getData().size());
    double e = calcEnergy(outputLayer->getData(), teach);

    double delta = 0;

    if (epsilon < e) {
        for (int i = 0; i < (int)outNeurons; i++) {
            double y = outputLayer->getData(i);
            delta = (teach(i) - y) * y * (1 - y);

            for (int j = 0; j < (int)hiddenNeurons; j++) {
                deltaH(j) += delta * hiddenToOut->getWeights(j, i);
                hiddenToOut->addWeight(
                    j, i, (learningrate * delta * hiddenLayer->getData(j)));
            }
        }

        for (int i = 0; i < (int)hiddenNeurons; i++) {
            delta = deltaH(i) * hiddenLayer->getData(i) *
                    (1 - hiddenLayer->getData(i));

            for (int j = 0; j < (int)inNeurons; j++) {
                inToHidden->addWeight(
                    j, i, (learningrate * delta * inputLayer->getData(j)));
            }
        }
    }
}

/**
 * A wrapper around doing one learning iteration, it:
 * 1. Sets the input
 * 2. Propagates the input through the net
 * 3. Decides wether to backpropagate or if it is finished
 * 4. Might backpropagate
 *
 * @param input A vector resembling network input.
 * @param teach A vector resembling an network example.
 */
void NeuralNetwork::step(Eigen::VectorXd input, Eigen::VectorXd teach) {
    inputLayer->setData(input);
    propagate();
    Eigen::VectorXd output = outputLayer->getData();

    double error = calcEnergy(teach, output);

    printf("Current error: %f\n", error);

    if (error > epsilon) {
        backpropagate(teach);
    } else {
        learend = true;
    }
}

/**
 * For a given input prints the networks output.
 *
 * @param A vector representing the input.
 */
void NeuralNetwork::printInterference(Eigen::VectorXd input) {
    inputLayer->setData(input);
    propagate();
    Eigen::VectorXd output = outputLayer->getData();

    printf("\nFor Input: ");
    for (int i = 0; i < inputLayer->getData().size(); i++) {
        printf("%f ", inputLayer->getData(i));
    }
    printf("\nThe network output is: ");
    for (int i = 0; i < outputLayer->getData().size(); i++) {
        printf("%f ", outputLayer->getData(i));
    }
    printf("\n");
}