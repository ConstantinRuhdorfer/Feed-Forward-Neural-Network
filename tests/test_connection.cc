#include <catch.h>
#include <connection.h>



TEST_CASE("1: Basic constructor connection", "[multi-file:3]") {

    Connection c(10, 20);

    REQUIRE(c.getWeights().size() == 200);
    /**
     * Weights are initialited randomly in the intervall [-0.5, 0.5].
     */
    REQUIRE(c.getWeights().sum() >= c.getWeights().size() * -0.5);
    REQUIRE(c.getWeights().sum() <= c.getWeights().size() * 0.5);
}