#include <layer.h>
#include <math.h>
#include <neuralNetwork.h>
#include <stdlib.h>
#include <algorithm>
#include <eigen3/Eigen/Dense>

/*---------------------------------------------------------------------*/
/*                        Private                                      */
/*---------------------------------------------------------------------*/

/**
 * Activation functions implementation
 */

/**
 *
 */
double NeuralNetwork::calcFastSigmoid(int x) {
    return ((double)x / (1 + abs(x)));
}

/**
 *
 */
double NeuralNetwork::calcSigmoid(int x) { return 1.0 / (1 + exp(-x)); }

/**
 *
 */
double NeuralNetwork::calcRelu(int x) { return std::max(0, x); }

/**
 *
 */
double NeuralNetwork::calcRelu6(int x) { return std::min(std::max(0, x), 6); }

/*---------------------------------------------------------------------*/
/*                        Public                                       */
/*---------------------------------------------------------------------*/

/**
 *
 */
double NeuralNetwork::calcActivation(int x) {
    switch (currentActivationFunction) {
        case sigmoid:
            return calcSigmoid(x);
            break;
        case fastSigmoid:
            return calcFastSigmoid(x);
            break;
        case ReLu:
            return calcRelu(x);
            break;
        case ReLu6:
            return calcRelu6(x);
            break;
        default:
            throw new std::invalid_argument(
                "Was not one of sigmoid, fast sigmoid, relu or relu6.");
            break;
    }
};

/**
 * Network housekeeping
 */

/**
 * Gets called by all constructors.
 */
void NeuralNetwork::initialize() {
    // initialize the layers
    inputLayer = new Layer(inNeurons);
    hiddenLayer = new Layer(hiddenNeurons);
    outputLayer = new Layer(outNeurons);

    // initialize the weight matricies
    inToHidden = new Connection(inNeurons, hiddenNeurons);
    hiddenToOut = new Connection(hiddenNeurons, outNeurons);

    learend = false;
};

/**
 * Learning
 */

/**
 * Network internals
 */

/**
 *
 */
double NeuralNetwork::calcEnergy(Eigen::VectorXd groundTruth,
                                 Eigen::VectorXd netOutput) {
    double energy = 0;
    for (int i = 0; i < netOutput.size(); i++) {
        energy +=
            (groundTruth(i) - netOutput(i)) * (groundTruth(i) - netOutput(i));
    }

    return (energy /= 2.0);
}

/**
 *
 */
void NeuralNetwork::propagate() {
    inputLayer->setThreshold(1);

    for (int i = 0; i < hiddenNeurons; i++) {
        double net = 0;
        for (int j = 0; j < inNeurons; j++) {
            net += inToHidden->getWeights(j, i) * inputLayer->getData(j);
            hiddenLayer->setData(i, calcActivation(net));
        }
    }

    for (int i = 0; i < outNeurons; i++) {
        double net = 0;
        for (int j = 0; j < hiddenNeurons; j++) {
            net += hiddenToOut->getWeights(j, i) * hiddenLayer->getData(j);
            outputLayer->setData(i, calcActivation(net));
        }
    }
}

/**
 *
 */
void NeuralNetwork::backpropagate(Eigen::VectorXd teach) {
    Eigen::VectorXd deltaH(hiddenLayer->getData().size());
    double e = calcEnergy(outputLayer->getData(), teach);

    double delta = 0;

    if (epsilon < e) {
        for (int i = 0; i < outNeurons; i++) {
            double y = outputLayer->getData(i);
            delta = (teach(i) - y) * y * (1 - y);

            for (int j = 0; j < hiddenNeurons; j++) {
                deltaH(j) += delta * hiddenToOut->getWeights(j, i);
                hiddenToOut->addWeight(
                    j, i, (learningrate * delta * hiddenLayer->getData(j)));
            }
        }

        for (int i = 0; i < hiddenNeurons; i++) {
            delta = deltaH(i) * hiddenLayer->getData(i) *
                    (1 - hiddenLayer->getData(i));

            for (int j = 0; j < inNeurons; j++) {
                inToHidden->addWeight(
                    j, i, (learningrate * delta * inputLayer->getData(j)));
            }
        }
    }
}

/**
 *
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
 *
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