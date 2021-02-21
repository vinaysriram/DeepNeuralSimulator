#ifndef POOL_H_
#define POOL_H_

#include <Eigen/Dense>
#include <random>

enum class NeuronType {
  RS,  // Regular Spiking
  IB,  // Intrinsically Bursting
  CH,  // Chattering
  FS,  // Fast-Spiking
  TC,  // Thalamo Cortical
  RZ,  // Resonator
  LT   // Low-Threshold Spiking
};

class Neuron {
 public:
  // Construct a neuron given dimension and type.
  Neuron(int inputDimension, NeuronType type,
         std::default_random_engine &generator);

  // Computes the neural activation on an input.
  double Activation(const Eigen::VectorXd &x);

  // Computes the spike rate a current.
  double SpikeRate(double current);

 private:
  Eigen::VectorXd direction;      // Neuron direction vector
  double baseCurrent, baseRate;   // Tuning curve adjustments
  double a, b, c, d, gain, bias;  // Neuron response parameters
};

class Pool {
 public:
  // Construct neuron pool given pool size, dimension, and type.
  Pool(int numNeurons, int inputDimension, NeuronType type);

  // Use SVD factorization to compute the neuron pool weights.
  void Train(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y);

  // Computes the neural activation of a single sample input.
  Eigen::MatrixXd ActivationVector(const Eigen::VectorXd &x);

  // Computes the neural activations for an entire dataset.
  Eigen::MatrixXd ActivationMatrix(const Eigen::MatrixXd &x);

  // Runs the neuron pool on a single sample input.
  Eigen::VectorXd RunSample(const Eigen::VectorXd &x);

  // Runs the neuron pool on an entire dataset.
  Eigen::MatrixXd RunDataset(const Eigen::MatrixXd &x);

 private:
  std::vector<Neuron> neurons;  // Neurons
  Eigen::MatrixXd weights;      // Weight matrix
};

#endif