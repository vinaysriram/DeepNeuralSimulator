
#ifndef ABSTRACT_LAYER_HPP_
#define ABSTRACT_LAYER_HPP_

#include <random>
#include <Eigen/Dense>

class AbstractLayer {

public:
  
  AbstractLayer(int numNeurons, int numInputs, std::default_random_engine &gen);
  virtual ~AbstractLayer();
  virtual void activation(Eigen::VectorXd &input, Eigen::VectorXd &output) = 0;
  virtual void gradient(Eigen::VectorXd &input, Eigen::VectorXd &output) = 0;
  void computeScaled(Eigen::VectorXd &input, Eigen::VectorXd &output);
  Eigen::VectorXd b; Eigen::MatrixXd W; int inDim, outDim; bool reg;
  
};

#endif
