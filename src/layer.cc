#include <layer.h>

Layer::Layer(unsigned int numNeurons)
    : numNeurons(numNeurons), data(numNeurons) {
    threshold = 1;
}