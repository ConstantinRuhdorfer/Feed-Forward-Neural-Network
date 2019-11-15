#include <connection.h>
#include <eigen3/Eigen/Dense>

/*---------------------------------------------------------------------*/
/*                        Private                                      */
/*---------------------------------------------------------------------*/

/*---------------------------------------------------------------------*/
/*                        Public                                       */
/*---------------------------------------------------------------------*/

/**
 * Constructor for a connection with a weight matrix of nRow X nCol.
 *
 * @param nRow Number of rows.
 * @param nCol Number of cols.
 */
Connection::Connection(int nRow, int nCol) {
    weights = Eigen::MatrixXd::Random(nRow, nCol);
    weights = (weights + Eigen::MatrixXd::Constant(nRow, nCol, 1.)) * 1.0 / 2.0;
    weights = (weights + Eigen::MatrixXd::Constant(nRow, nCol, -0.5));
}

/**
 * Overrides the existing weights.
 * Mostly used to enforce consitency in testing.
 * Use with caution
 * (e.g. not in training or if you do think hard if thats your best option).
 *
 * @param newWeights The matrix containing the new weights.
 */
void Connection::overrideWeights(Eigen::MatrixXd newWeights) {
    weights = newWeights;
}

/**
 * Changes the weight by adding a new value onto it.
 *
 * @param rowIndex rowIndex in the weights matrix.
 * @param colIndex colIndex in the weights matrix.
 * @param data The change to be applied.
 */
void Connection::addWeight(int rowIndex, int colIndex, double data) {
    weights(rowIndex, colIndex) += data;
}