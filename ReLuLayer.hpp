
#ifndef RELU_LAYER_HPP_
#define RELU_LAYER_HPP_

#include "AbstractLayer.hpp"

class ReLuLayer : public AbstractLayer {
  
public:
  
  ReLuLayer(int numNeurons, int numInputs, std::default_random_engine &gen);
  ~ReLuLayer();
  void activation(Eigen::VectorXd &input, Eigen::VectorXd &output);
  void gradient(Eigen::VectorXd &input, Eigen::VectorXd &output);  
  
};

#endif
