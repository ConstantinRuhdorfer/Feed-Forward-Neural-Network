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