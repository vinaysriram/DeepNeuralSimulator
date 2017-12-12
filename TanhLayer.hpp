
#ifndef TANH_LAYER_HPP_
#define TANH_LAYER_HPP_

#include "AbstractLayer.hpp"

class TanhLayer : public AbstractLayer {
  
public:
  
  TanhLayer(int numNeurons, int numInputs, std::default_random_engine &gen);
  ~TanhLayer();
  void activation(Eigen::VectorXd &input, Eigen::VectorXd &output);
  void gradient(Eigen::VectorXd &input, Eigen::VectorXd &output);  
  
};

#endif
