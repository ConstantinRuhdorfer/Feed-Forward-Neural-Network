#include <connection.h>
#include <eigen3/Eigen/Dense>

/**
 *
 */
Connection::Connection(int nRow, int nCol) : weights(0, 0) {
    weights = Eigen::MatrixXd::Random(nRow, nCol);
    weights = (weights + Eigen::MatrixXd::Constant(nRow, nCol, 1.)) * 1.0 / 2.0;
    weights = (weights + Eigen::MatrixXd::Constant(nRow, nCol, -0.5));
}

/**
 *
 */
void Connection::addWeight(int rowIndex, int colIndex, double data) {
    weights(rowIndex, colIndex) += data;
}