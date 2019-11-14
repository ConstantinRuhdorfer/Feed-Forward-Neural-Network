#pragma once

#include <eigen3/Eigen/Dense>

class Connection {
   private:
    /**
     * In a fully connected neural network every neuron of
     * every layer is connected with every neuron of the following layer.
     * This is best represented as a matrix.
     *
     * E.g.:
     *
     *  layer1  layer2
     *
     *  l1      k1
     *  l2      k2
     *  l3      k3
     *
     * is best reepresented by the weights:
     *
     *        k1   k2   k3
     *  l1: [[w11, w12, w13],
     *  l1:  [w21, w22, w23],
     *  l1:  [w31, w32, w33]]
     */
    Eigen::MatrixXd weights;

   public:
    Connection(int nRow, int nCol);
    ~Connection(){};
    void overideWeights(Eigen::MatrixXd newWeights);
    /**
     * Weight calculation
     */
    void addWeight(int rowIndex, int colIndex, double data);
    /**
     * Getter
     */
    Eigen::MatrixXd getWeights() { return weights; };
    double getWeights(int nRow, int nCol) { return weights(nRow, nCol); };
};