
#ifndef SIGMOID_LAYER_HPP_
#define SIGMOID_LAYER_HPP_

#include "AbstractLayer.hpp"

class SigmoidLayer : public AbstractLayer {
  
public:
  
  SigmoidLayer(int numNeurons, int numInputs, std::default_random_engine &gen);
  ~SigmoidLayer();
  void activation(Eigen::VectorXd &input, Eigen::VectorXd &output);
  void gradient(Eigen::VectorXd &input, Eigen::VectorXd &output);
  
};

#endif
