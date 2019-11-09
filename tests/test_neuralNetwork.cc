#include <../extern/catch/catch.h>
#include <neuralNetwork.h>

neuralNetwork NN;

TEST_CASE("1: Basic constructor", "[multi-file:1]") {
    REQUIRE(NN.getInNeurons() == 5);
    REQUIRE(NN.getHiddenNeurons() == 5);
    REQUIRE(NN.getOutNeurons() == 5);
    REQUIRE(NN.getEpsilon() == 1.0);
    REQUIRE(NN.getLearningrate() == 0.5);
}

TEST_CASE("2: Sigmoid", "[multi-file:1]") {
    REQUIRE(NN.calcSigmoid(0) == 0.5);
    REQUIRE(NN.calcSigmoid(8) == Approx(0.9996646499));
    REQUIRE(NN.calcSigmoid(-8) == Approx(0.0003353501));
}

TEST_CASE("3: Fast Sigmoid", "[multi-file:1]") {
    REQUIRE(NN.calcFastSigmoid(0) == 0);
    REQUIRE(NN.calcFastSigmoid(1) == 0.5);
    REQUIRE(NN.calcFastSigmoid(-1) == -0.5);
}

TEST_CASE("4: ReLu", "[multi-file:1]") {
    REQUIRE(NN.calcRelu(0) == 0);
    REQUIRE(NN.calcRelu(10) == 10);
    REQUIRE(NN.calcRelu(-10) == 0);
}

TEST_CASE("5: ReLu6", "[multi-file:1]") {
    REQUIRE(NN.calcRelu6(0) == 0);
    REQUIRE(NN.calcRelu6(10) == 6);
    REQUIRE(NN.calcRelu6(-10) == 0);
    REQUIRE(NN.calcRelu6(5) == 5);
}