#include <layer.h>

/*---------------------------------------------------------------------*/
/*                        Private                                      */
/*---------------------------------------------------------------------*/

/*---------------------------------------------------------------------*/
/*                        Public                                       */
/*---------------------------------------------------------------------*/

/**
 * Constructor for a layer holding a vector of
 * neurons represented by their value.
 *
 * @param numNeurons The number of neurons to use.
 */
Layer::Layer(unsigned int numNeurons)
    : numNeurons(numNeurons), data(numNeurons) {
    threshold = 1;
}