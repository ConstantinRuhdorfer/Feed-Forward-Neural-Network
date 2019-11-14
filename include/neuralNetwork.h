#pragma once

#include <connection.h>
#include <layer.h>

#define MAX_INPUT_LAYER_SIZE 20
#define MAX_HIDDEN_LAYER_SIZE 40
#define MAX_OUTPUT_LAYER_SIZE 20

// Mainly for easy testing.
#define DEFAULT_IN_NEURONS 5
#define DEFAULT_HIDDEN_NEURONS 5
#define DEFAULT_OUT_NEURONS 5
#define DEFAULT_EPSILON 1.0
#define DEFAULT_LEARNINGRATE 0.5

enum CurrentActivationFunction { sigmoid, fastSigmoid, ReLu, ReLu6 };

/**
 * Neural network with backpropagation.
 */
class NeuralNetwork {
   private:
    // Network structure:
    const unsigned int inNeurons = 0;
    const unsigned int hiddenNeurons = 0;
    const unsigned int outNeurons = 0;

    // Current activation function; Default is sigmoid
    CurrentActivationFunction currentActivationFunction = sigmoid;

    // Learning
    double epsilon = 1.0;
    double learningrate = 0.5;

    Layer* inputLayer;
    Layer* hiddenLayer;
    Layer* outputLayer;

    Connection* inToHidden;
    Connection* hiddenToOut;

    // Wether training is finished...
    bool learend = false;

    /**
     * Network housekeeping
     */
    void initialize();

    /**
     * Various activation functions
     */
    double calcSigmoid(int x);
    double calcFastSigmoid(int x);
    double calcRelu(int x);
    double calcRelu6(int x);

   public:
    /**
     * Constructors
     */
    NeuralNetwork()
        : inNeurons(DEFAULT_IN_NEURONS),
          hiddenNeurons(DEFAULT_HIDDEN_NEURONS),
          outNeurons(DEFAULT_OUT_NEURONS),
          epsilon(DEFAULT_EPSILON),
          learningrate(DEFAULT_LEARNINGRATE) {
        initialize();
    };
    NeuralNetwork(int inNeurons, int hiddenNeurons, int outNeurons)
        : inNeurons(inNeurons),
          hiddenNeurons(hiddenNeurons),
          outNeurons(outNeurons),
          epsilon(DEFAULT_EPSILON),
          learningrate(DEFAULT_LEARNINGRATE) {
        initialize();
    };
    NeuralNetwork(int inNeurons, int hiddenNeurons, int outNeurons,
                  double epsilon, double learningrate)
        : inNeurons(inNeurons),
          hiddenNeurons(hiddenNeurons),
          outNeurons(outNeurons),
          epsilon(epsilon),
          learningrate(learningrate) {
        initialize();
    };
    /**
     * Destructor
     */
    ~NeuralNetwork() {
        delete (inputLayer);
        delete (hiddenLayer);
        delete (outputLayer);
        delete (inToHidden);
        delete (hiddenToOut);
    };
    /**
     * Network Learning
     */
    void step(Eigen::VectorXd input, Eigen::VectorXd teach);
    void printInterference(Eigen::VectorXd input);
    void propagate();
    void backpropagate(Eigen::VectorXd teach);
    double calcActivation(int x);
    /**
     * Network internals
     */
    double calcEnergy(Eigen::VectorXd groundTruth, Eigen::VectorXd netOutput);
    /**
     * Getters
     */
    int getInNeurons() { return inNeurons; };
    int getHiddenNeurons() { return hiddenNeurons; };
    int getOutNeurons() { return outNeurons; };
    double getEpsilon() { return epsilon; };
    double getLearningrate() { return learningrate; };
    CurrentActivationFunction getCurrentActivationFunction() {
        return currentActivationFunction;
    };
    bool getHasLearned() { return learend; };
    /**
     * Setters
     */
    void setCurrentActivationFunction(CurrentActivationFunction curr) {
        currentActivationFunction = curr;
    };
};
