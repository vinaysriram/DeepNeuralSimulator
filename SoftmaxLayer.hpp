
#ifndef SOFTMAX_LAYER_HPP_
#define SOFTMAX_LAYER_HPP_

#include "AbstractLayer.hpp"

class SoftmaxLayer : public AbstractLayer {
  
public:
  
  SoftmaxLayer(int numNeurons, int numInputs, std::default_random_engine &gen);
  ~SoftmaxLayer();
  void activation(Eigen::VectorXd &input, Eigen::VectorXd &output);
  void gradient(Eigen::VectorXd &input, Eigen::VectorXd &output);
  
};

#endif
