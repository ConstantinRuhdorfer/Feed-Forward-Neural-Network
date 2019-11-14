#include <connection.h>
#include <eigen3/Eigen/Dense>

/**
 *
 */
Connection::Connection(int nRow, int nCol) {
    weights = Eigen::MatrixXd::Random(nRow, nCol);
    weights = (weights + Eigen::MatrixXd::Constant(nRow, nCol, 1.)) * 1.0 / 2.0;
    weights = (weights + Eigen::MatrixXd::Constant(nRow, nCol, -0.5));
}

/**
 * Use with caution or consistent testing.
 */
void Connection::overrideWeights(Eigen::MatrixXd newWeights) {
    weights = newWeights;
}

/**
 *
 */
void Connection::addWeight(int rowIndex, int colIndex, double data) {
    weights(rowIndex, colIndex) += data;
}