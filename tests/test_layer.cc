#include <catch.h>
#include <layer.h>

layer l(10);

TEST_CASE("1: Basic constructor layer", "[multi-file:2]") {
    REQUIRE(l.getData().size() == 10);
    for (int i = 0; i < l.getData().size(); i++) {
        REQUIRE(l.getData(i) == 0.0);
    }
    REQUIRE(l.getNumNeurons() == 10);
    REQUIRE(l.getThreshold() == 1);
}