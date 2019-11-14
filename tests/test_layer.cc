#include <catch.h>
#include <layer.h>

TEST_CASE("1: Basic constructor layer", "[multi-file:2]") {
    Layer l(10);

    REQUIRE(l.getData().size() == 10);
    for (int i = 0; i < (int)l.getData().size(); i++) {
        REQUIRE((unsigned)l.getData(i) == 0.0);
    }
    REQUIRE(l.getNumNeurons() == 10);
    REQUIRE(l.getThreshold() == 1);
}