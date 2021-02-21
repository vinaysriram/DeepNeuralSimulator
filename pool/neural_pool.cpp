
#include "pool/neural_pool.hpp"

Neuron::Neuron(int inputDimension, NeuronType type,
               std::default_random_engine &generator) {
  std::normal_distribution<double> gaussian(0.0, 1.0);
  std::uniform_real_distribution<double> bias_uniform(-1.0, 1.0);
  std::uniform_real_distribution<double> gain_uniform(1.0, 2.0);
  direction = Eigen::VectorXd(inputDimension);
  for (int i = 0; i < inputDimension; i++) {
    direction(i) = gaussian(generator);
  }
  direction.normalize();
  gain = gain_uniform(generator);
  bias = bias_uniform(generator);

  switch (type) {
    case NeuronType::RS:
      a = 0.02;
      b = 0.20;
      c = -65;
      d = 8.00;
      break;
    case NeuronType::IB:
      a = 0.02;
      b = 0.20;
      c = -55;
      d = 4.00;
      break;
    case NeuronType::CH:
      a = 0.02;
      b = 0.20;
      c = -50;
      d = 2.00;
      break;
    case NeuronType::FS:
      a = 0.10;
      b = 0.20;
      c = -65;
      d = 2.00;
      break;
    case NeuronType::TC:
      a = 0.02;
      b = 0.25;
      c = -65;
      d = 0.05;
      break;
    case NeuronType::RZ:
      a = 0.10;
      b = 0.26;
      c = -65;
      d = 2.00;
      break;
    default:
      a = 0.02;
      b = 0.25;
      c = -65;
      d = 2.00;
      break;
  }

  baseCurrent = 10.0;
  baseRate = SpikeRate(baseCurrent);
}

double Neuron::SpikeRate(double current) {
  double u = d;
  double v = c;
  double T = 0;
  double dt = 0.01;
  while (v < 30) {
    double dv = (0.04 * pow(v, 2) + 5.0 * v + 140 - u + current) * dt;
    double du = (a * (b * v - u)) * dt;
    v = v + dv;
    u = u + du;
    T = T + dt;
  }
  return 1.0 / T;
}

double Neuron::Activation(const Eigen::VectorXd &x) {
  double current = gain * (x.transpose() * direction).value() + bias;
  return (current > 0) ? SpikeRate(current + baseCurrent) - baseRate : 0;
}

Pool::Pool(int numNeurons, int inputDimension, NeuronType type) {
  std::default_random_engine generator;
  for (int i = 0; i < numNeurons; i++) {
    neurons.push_back(Neuron(inputDimension, type, generator));
  }
  weights = Eigen::MatrixXd(neurons.size() + 1, 1);
}

Eigen::MatrixXd Pool::ActivationVector(const Eigen::VectorXd &x) {
  Eigen::MatrixXd activations(1, neurons.size() + 1);
  for (size_t j = 0; j < neurons.size(); j++) {
    activations(0, j) = neurons[j].Activation(x);
  }
  activations(0, neurons.size()) = 1;
  return activations;
}

Eigen::MatrixXd Pool::ActivationMatrix(const Eigen::MatrixXd &x) {
  Eigen::MatrixXd activations(x.cols(), neurons.size() + 1);
  for (int i = 0; i < x.cols(); i++) {
    activations.row(i) = ActivationVector(x.col(i));
  }
  return activations;
}

void Pool::Train(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y) {
  Eigen::MatrixXd activations = ActivationMatrix(x);
  weights = Eigen::MatrixXd(neurons.size() + 1, y.rows());
  auto solveType = Eigen::ComputeThinU | Eigen::ComputeThinV;
  for (int i = 0; i < y.rows(); i++) {
    weights.col(i) = activations.bdcSvd(solveType).solve(y.row(i).transpose());
  }
}

Eigen::VectorXd Pool::RunSample(const Eigen::VectorXd &x) {
  return (ActivationVector(x) * weights).transpose();
}

Eigen::MatrixXd Pool::RunDataset(const Eigen::MatrixXd &x) {
  return (ActivationMatrix(x) * weights).transpose();
}