
#ifndef LINEAR_LAYER_HPP_
#define LINEAR_LAYER_HPP_

#include "AbstractLayer.hpp"

class LinearLayer : public AbstractLayer {
  
public:
  
  LinearLayer(int numNeurons, int numInputs, std::default_random_engine &gen);
  ~LinearLayer();
  void activation(Eigen::VectorXd &input, Eigen::VectorXd &output);
  void gradient(Eigen::VectorXd &input, Eigen::VectorXd &output);  
  
};

#endif
