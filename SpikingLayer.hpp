
#ifndef SPIKING_LAYER_HPP_
#define SPIKING_LAYER_HPP_

#include "AbstractLayer.hpp"

class SpikingLayer : public AbstractLayer {

public:

  SpikingLayer(int numNeurons, int numInputs, std::default_random_engine &gen, int type);
  ~SpikingLayer();
  double scaledSpikeRate(double i);
  double rawSpikeRate(double i);
  void activation(Eigen::VectorXd &input, Eigen::VectorXd &output);
  void gradient(Eigen::VectorXd &input, Eigen::VectorXd &output);
  double a, b, c, d, baseCurrent, baseRate;
  
}; 

#endif
