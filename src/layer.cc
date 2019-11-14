#include <layer.h>

layer::layer(unsigned int numNeurons)
    : numNeurons(numNeurons), data(numNeurons) {
    threshold = 1;
}