#pragma once

#include <eigen3/Eigen/Dense>

class Layer {
   private:
    const int numNeurons = 0;
    Eigen::VectorXd data;
    int threshold;

   public:
    Layer(unsigned int numNeurons);
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
        if (Layer::data.size() == data.size()) {
            Layer::data = data;
        }
    }
    void setData(int index, double data) {
        if (index < Layer::data.size()) {
            Layer::data(index) = data;
        }
    }
};
