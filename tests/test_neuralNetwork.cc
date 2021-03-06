#include <../extern/catch/catch.h>
#include <neuralNetwork.h>

TEST_CASE("1: Basic constructor NN", "[multi-file:1]") {
    NeuralNetwork NN;
    REQUIRE(NN.getInNeurons() == 5);
    REQUIRE(NN.getHiddenNeurons() == 5);
    REQUIRE(NN.getOutNeurons() == 5);
    REQUIRE(NN.getEpsilon() == 1.0);
    REQUIRE(NN.getLearningrate() == 0.5);
}

TEST_CASE("2: Set Activation function", "[multi-file:1]") {
    NeuralNetwork NN;
    NN.setCurrentActivationFunction(ReLu);
    REQUIRE(NN.getCurrentActivationFunction() == ReLu);
}

TEST_CASE("3: Sigmoid", "[multi-file:1]") {
    NeuralNetwork NN;
    NN.setCurrentActivationFunction(sigmoid);
    REQUIRE(NN.calcActivation(0) == 0.5);
    REQUIRE(NN.calcActivation(8) == Approx(0.9996646499));
    REQUIRE(NN.calcActivation(-8) == Approx(0.0003353501));
}

TEST_CASE("4: Fast Sigmoid", "[multi-file:1]") {
    NeuralNetwork NN;
    NN.setCurrentActivationFunction(fastSigmoid);
    REQUIRE(NN.calcActivation(0) == 0);
    REQUIRE(NN.calcActivation(1) == 0.5);
    REQUIRE(NN.calcActivation(-1) == -0.5);
}

TEST_CASE("5: ReLu", "[multi-file:1]") {
    NeuralNetwork NN;
    NN.setCurrentActivationFunction(ReLu);
    REQUIRE(NN.calcActivation(0) == 0);
    REQUIRE(NN.calcActivation(10) == 10);
    REQUIRE(NN.calcActivation(-10) == 0);
}

TEST_CASE("6: ReLu6", "[multi-file:1]") {
    NeuralNetwork NN;
    NN.setCurrentActivationFunction(ReLu6);
    REQUIRE(NN.calcActivation(0) == 0);
    REQUIRE(NN.calcActivation(10) == 6);
    REQUIRE(NN.calcActivation(-10) == 0);
    REQUIRE(NN.calcActivation(5) == 5);
}

TEST_CASE("7: Simple Training", "[multi-file:1]") {
    NeuralNetwork testNN(2, 1, 2, 0.01, 0.7);

    Eigen::MatrixXd inToHidden(2, 1);
    Eigen::MatrixXd hiddenToOut(1, 2);

    inToHidden << 0, 0;
    hiddenToOut << 0, 0;

    testNN.getInToHidden()->overrideWeights(inToHidden);
    testNN.getHiddenToOut()->overrideWeights(hiddenToOut);

    testNN.setCurrentActivationFunction(sigmoid);

    Eigen::MatrixXd testIn(10, 2);
    Eigen::MatrixXd teachIn(10, 2);

    /**
     * e.g.:
     *
     * train    teach
     * 1, 0,    1, 0
     * 1, 0,    1, 0
     * 1, 0,    1, 0
     * 1, 0,    1, 0
     * 1, 0,    1, 0
     * 0, 1,    0, 1,
     * 0, 1,    0, 1,
     * 0, 1,    0, 1,
     * 0, 1,    0, 1,
     * 0, 1     0, 1
     */
    testIn << 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1;
    teachIn << 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1;

    int j = 0;

    bool finished = false;

    while (!finished) {
        j++;
        for (int i = 0; i < 10; i++) {
            testNN.step(testIn.row(i), teachIn.row(i));
            finished = testNN.getHasLearned();
        }
    }

    printf("Needed %d sweeps over the data", j);

    Eigen::VectorXd interference(2);
    interference << 1, 0;
    testNN.printInterference(interference);

    interference << 0, 1;
    testNN.printInterference(interference);

    REQUIRE(testNN.getHasLearned());
}