
#include "neural_network.hpp"

Network::Network(const std::vector<LayerType> &layerTypes,
                 const std::vector<int> &layerDimensions)
    : layerDimensions(layerDimensions), layerTypes(layerTypes) {
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, 1.0);
  for (size_t i = 1; i < layerDimensions.size(); i++) {
    W[i - 1] = Eigen::MatrixXd(layerDimensions[i], layerDimensions[i - 1]);
    b[i - 1] = Eigen::VectorXd(layerDimensions[i]);
    for (int r = 0; r < layerDimensions[i]; r++) {
      for (int c = 0; c < layerDimensions[i - 1]; c++) {
        W[i - 1](r, c) = distribution(generator);
      }
      b[i - 1](r) = distribution(generator);
    }
  }
}

Eigen::VectorXd Network::Activation(const Eigen::VectorXd &z, LayerType type) {
  if (type == LayerType::Linear) {
    return z;
  } else if (type == LayerType::ReLU) {
    return (z.array() < 0).select(0, z);
  } else if (type == LayerType::Logistic) {
    Eigen::VectorXd activation(z.size());
    for (int i = 0; i < z.size(); i++) {
      activation(i) = 1.0 / (1.0 + exp(-z(i)));
    }
    return activation;
  } else if (type == LayerType::Tanh) {
    Eigen::VectorXd activation(z.size());
    for (int i = 0; i < z.size(); i++) {
      activation(i) = tanh(z(i));
    }
    return activation;
  } else {
    double activationSum = 0;
    Eigen::VectorXd activation(z.size());
    for (int i = 0; i < z.size(); i++) {
      activation(i) = exp(z(i));
      activationSum += activation(i);
    }
    for (int i = 0; i < z.size(); i++) {
      activation(i) = activation(i) / activationSum;
    }
    return activation;
  }
}

Eigen::VectorXd Network::Gradient(const Eigen::VectorXd &z, LayerType type) {
  if (type == LayerType::ReLU) {
    Eigen::VectorXd partial = (z.array() < 0).select(0, z);
    return (partial.array() > 0).select(1, partial);
  } else if (type == LayerType::Logistic) {
    Eigen::VectorXd gradient(z.size());
    for (int i = 0; i < z.size(); i++) {
      gradient(i) = exp(z(i)) / pow(1.0 + exp(z(i)), 2.0);
    }
    return gradient;
  } else if (type == LayerType::Tanh) {
    Eigen::VectorXd gradient(z.size());
    for (int i = 0; i < z.size(); i++) {
      gradient(i) = 1.0 - pow(tanh(z(i)), 2.0);
    }
    return gradient;
  } else {
    return Eigen::VectorXd::Constant(z.size(), 1);
  }
}

LayerOutputs Network::ForwardPropagation(const Eigen::VectorXd &x) {
  LayerOutputs outputs;
  outputs.a[0] = x;
  for (size_t i = 1; i <= W.size(); i++) {
    outputs.z[i] = W[i - 1] * outputs.a[i - 1] + b[i - 1];
    outputs.a[i] = Activation(outputs.z[i], layerTypes[i - 1]);
  }
  return outputs;
}

LayerGradients Network::BackwardPropagation(const Eigen::VectorXd &x,
                                            const Eigen::VectorXd &y) {
  LayerGradients gradients;
  std::map<int, Eigen::VectorXd> delta;
  LayerOutputs outputs = ForwardPropagation(x);
  delta[W.size()] = outputs.a[W.size()] - y;
  for (size_t i = W.size() - 1; i >= 1; i--) {
    Eigen::VectorXd gradient = Gradient(outputs.z[i], layerTypes[i - 1]);
    delta[i] = (W[i].transpose() * delta[i + 1]).cwiseProduct(gradient);
  }
  for (size_t i = 0; i < W.size(); i++) {
    gradients.dW[i] = delta[i + 1] * outputs.a[i].transpose();
    gradients.db[i] = delta[i + 1];
  }
  return gradients;
}

Eigen::VectorXd Network::RunSample(const Eigen::VectorXd &x) {
  Eigen::VectorXd inference = x;
  for (size_t i = 0; i < W.size(); i++) {
    inference = Activation(W[i] * inference + b[i], layerTypes[i]);
  }
  return inference;
}

Eigen::MatrixXd Network::RunDataset(const Eigen::MatrixXd &x) {
  Eigen::MatrixXd y(W[W.size() - 1].rows(), x.cols());
  for (int i = 0; i < x.cols(); i++) {
    y.col(i) = RunSample(x.col(i));
  }
  return y;
}

void Network::Train(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y,
                    double learningRate, double regularization, int epochs) {
  for (int e = 0; e < epochs; e++) {
    for (int i = 0; i < x.cols(); i++) {
      LayerGradients gradients = BackwardPropagation(x.col(i), y.col(i));
      for (size_t j = 0; j < W.size(); j++) {
        W[j] -= learningRate * (gradients.dW[j] + regularization * W[j]);
        b[j] -= learningRate * gradients.db[j];
      }
    }
  }
}