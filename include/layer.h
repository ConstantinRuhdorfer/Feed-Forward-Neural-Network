#pragma once

#include <eigen3/Eigen/Dense>

class layer {
   private:
    const int numNeurons = 0;
    Eigen::VectorXd data;
    int threshold;

   public:
    layer(unsigned int numNeurons);
    /**
     * Getter
     */
    Eigen::VectorXd getData() { return data; };
    double getData(int index) { return data(index); };
    int getNumNeurons() { return numNeurons; };
    int getThreshold() { return threshold; };
    /**
     * Setter
     */
    void setThreshold(int threshold) { threshold = threshold; };
    void setData(Eigen::VectorXd data) {
        if (layer::data.size() == data.size()) {
            layer::data = data;
        }
    }
    void setData(int index, double data) {
        if (index < layer::data.size()) {
            layer::data(index) = data;
        }
    }
};
