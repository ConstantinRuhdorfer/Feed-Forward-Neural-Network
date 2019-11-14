#include <../extern/catch/catch.h>
#include <neuralNetwork.h>

neuralNetwork NN;

TEST_CASE("1: Basic constructor NN", "[multi-file:1]") {
    REQUIRE(NN.getInNeurons() == 5);
    REQUIRE(NN.getHiddenNeurons() == 5);
    REQUIRE(NN.getOutNeurons() == 5);
    REQUIRE(NN.getEpsilon() == 1.0);
    REQUIRE(NN.getLearningrate() == 0.5);
}

TEST_CASE("2: Set Activation function", "[multi-file:1]") {
    NN.setCurrentActivationFunction(ReLu);
    REQUIRE(NN.getCurrentActivationFunction() == ReLu);
}

TEST_CASE("3: Sigmoid", "[multi-file:1]") {
    NN.setCurrentActivationFunction(sigmoid);
    REQUIRE(NN.calcActivation(0) == 0.5);
    REQUIRE(NN.calcActivation(8) == Approx(0.9996646499));
    REQUIRE(NN.calcActivation(-8) == Approx(0.0003353501));
}

TEST_CASE("4: Fast Sigmoid", "[multi-file:1]") {
    NN.setCurrentActivationFunction(fastSigmoid);
    REQUIRE(NN.calcActivation(0) == 0);
    REQUIRE(NN.calcActivation(1) == 0.5);
    REQUIRE(NN.calcActivation(-1) == -0.5);
}

TEST_CASE("5: ReLu", "[multi-file:1]") {
    NN.setCurrentActivationFunction(ReLu);
    REQUIRE(NN.calcActivation(0) == 0);
    REQUIRE(NN.calcActivation(10) == 10);
    REQUIRE(NN.calcActivation(-10) == 0);
}

TEST_CASE("6: ReLu6", "[multi-file:1]") {
    NN.setCurrentActivationFunction(ReLu6);
    REQUIRE(NN.calcActivation(0) == 0);
    REQUIRE(NN.calcActivation(10) == 6);
    REQUIRE(NN.calcActivation(-10) == 0);
    REQUIRE(NN.calcActivation(5) == 5);
}

TEST_CASE("7: Simple Training", "[multi-file:1]") {
    neuralNetwork testNN(2, 1, 2, 0.01, 0.7);
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

    while (!testNN.getHasLearned()) {
        printf("\nNew Train sweep!\n");

        for (int i = 0; i < 10; i++) {
            testNN.step(testIn.row(i), teachIn.row(i));
        }
    }

    Eigen::VectorXd interference(2);
    interference << 1, 0;
    testNN.printInterference(interference);

    interference << 0, 1;
    testNN.printInterference(interference);

    REQUIRE(testNN.getHasLearned());
}