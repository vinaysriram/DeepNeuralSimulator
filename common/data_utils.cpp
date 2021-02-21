#include "data_utils.hpp"

#include <math.h>

#include <Eigen/Dense>
#include <fstream>
#include <random>

// Dataset Generators

std::pair<double, double> Curve(data_utils::CategoricalDatasetType type, int i,
                                double t) {
  if (type == data_utils::CategoricalDatasetType::Clustered) {
    if (i == 0) return std::pair<double, double>(0, 0.2);
    if (i == 1) return std::pair<double, double>(-0.3, -0.2);
    return std::pair<double, double>(0.3, -0.2);
  } else if (type == data_utils::CategoricalDatasetType::Concentric) {
    double x0 = cos(2 * M_PI * t), x1 = sin(2 * M_PI * t);
    if (i == 0) return std::pair<double, double>(0.2 * x0, 0.2 * x1);
    if (i == 1) return std::pair<double, double>(0.5 * x0, 0.5 * x1);
    return std::pair<double, double>(0.8 * x0, 0.8 * x1);
  } else if (type == data_utils::CategoricalDatasetType::ZigZagLines) {
    return std::pair<double, double>(2 * t - 1,
                                     abs(2 * t - 1) + double(i) / 2 - 1);
  } else if (type == data_utils::CategoricalDatasetType::Intersection) {
    if (i == 0) return std::pair<double, double>(2 * t - 1, 2 * t - 1);
    if (i == 1) return std::pair<double, double>(2 * t - 1, 1 - 2 * t);
    return std::pair<double, double>(0, 2 * t - 1);
  } else if (type == data_utils::CategoricalDatasetType::SpiralArms) {
    return std::pair<double, double>(
        t * cos(2 * M_PI * t + 2 * M_PI * double(i) / 3),
        t * sin(2 * M_PI * t + 2 * M_PI * double(i) / 3));
  }
  return std::pair<double, double>(0, 0);
}

double Surface(double x0, double x1, data_utils::NumericalDatasetType type) {
  switch (type) {
    case data_utils::NumericalDatasetType::EllipticParaboloid:
      return pow(x0, 2) + pow(x1, 2);
    case data_utils::NumericalDatasetType::HyperbolicParaboloid:
      return pow(x0, 2) - pow(x1, 2);
    case data_utils::NumericalDatasetType::SemiEllipsoid:
      return sqrt(2.0 - (pow(x0, 2) + pow(x1, 2)));
    case data_utils::NumericalDatasetType::SinusoidProduct:
      return sin(M_PI * x0) * cos(M_PI * x1);
    case data_utils::NumericalDatasetType::ConcaveClover:
      return sqrt(abs(x0) * abs(x1));
    default:
      return 0;
  }
}

Eigen::MatrixXd GenerateLabels(int numSamples) {
  Eigen::MatrixXd y = Eigen::MatrixXd::Zero(3, numSamples);
  for (int i = 0; i < numSamples; i++) {
    y(i % 3, i) = 1;
  }
  return y;
}

Eigen::MatrixXd data_utils::UniformSampling(int numSamples) {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  Eigen::MatrixXd x(2, numSamples);
  for (int i = 0; i < numSamples; i++) {
    x(0, i) = dist(generator);
    x(1, i) = dist(generator);
  }
  return x;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> data_utils::MakeCategoricalDataset(
    data_utils::CategoricalDatasetType datasetType, int numSamples) {
  Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(numSamples, 0.0, 1.0);
  Eigen::MatrixXd x(2, numSamples);
  Eigen::MatrixXd y = GenerateLabels(numSamples);
  std::default_random_engine generator;
  std::normal_distribution<double> dist(0.0, 0.05);
  for (int j = 0; j < numSamples; j++) {
    std::pair<double, double> pair = Curve(datasetType, j % 3, t[j]);
    x(0, j) = pair.first + dist(generator);
    x(1, j) = pair.second + dist(generator);
  }
  return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>(x, y);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> data_utils::MakeNumericalDataset(
    data_utils::NumericalDatasetType datasetType, int numSamples) {
  Eigen::MatrixXd x = data_utils::UniformSampling(numSamples);
  Eigen::MatrixXd y(1, numSamples);
  for (int i = 0; i < numSamples; i++) {
    y(0, i) = Surface(x(0, i), x(1, i), datasetType);
  }
  return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>(x, y);
}

// Dataset File I/O

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> data_utils::LoadDataset(
    const std::string &datasetFile) {
  std::string line;
  std::ifstream datasetStream(datasetFile);
  std::getline(datasetStream, line);

  std::istringstream headerStream(line);
  int numSamples, xDim, yDim;
  headerStream >> numSamples;
  headerStream >> xDim;
  headerStream >> yDim;

  Eigen::MatrixXd x(xDim, numSamples);
  Eigen::MatrixXd y(yDim, numSamples);
  for (int i = 0; i < numSamples; i++) {
    std::getline(datasetStream, line);
    std::istringstream exampleStream(line);
    for (int j = 0; j < xDim; j++) {
      exampleStream >> x(j, i);
    }
    for (int j = 0; j < yDim; j++) {
      exampleStream >> y(j, i);
    }
  }
  datasetStream.close();
  return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>(x, y);
}

void data_utils::DumpDataset(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y,
                             const std::string &datasetFile) {
  std::ofstream datasetStream(datasetFile);
  datasetStream << x.cols() << ' ' << x.rows() << ' ' << y.rows() << std::endl;
  for (int i = 0; i < x.cols(); i++) {
    for (int j = 0; j < x.rows(); j++) {
      datasetStream << x(j, i) << ' ';
    }
    for (int j = 0; j < y.rows() - 1; j++) {
      datasetStream << y(j, i) << ' ';
    }
    datasetStream << y(y.rows() - 1, i) << std::endl;
  }
  datasetStream.close();
}

// Model Evaluation Methods

int EncodingToLabel(const Eigen::VectorXd &encoding) {
  int label = 0;
  for (int i = 1; i < encoding.size(); i++) {
    if (encoding(i) > encoding(label)) {
      label = i;
    }
  }
  return label;
}

Eigen::MatrixXi data_utils::ConfusionMatrix(
    const Eigen::MatrixXd &x, const Eigen::MatrixXd &y_truth,
    const Eigen::MatrixXd &y_predicted) {
  int numClasses = y_truth.rows();
  Eigen::MatrixXi confusion = Eigen::MatrixXi::Zero(numClasses, numClasses);
  for (int i = 0; i < x.cols(); i++) {
    int true_label = EncodingToLabel(y_truth.col(i));
    int predicted_label = EncodingToLabel(y_predicted.col(i));
    confusion(true_label, predicted_label) += 1;
  }
  return confusion;
}

double data_utils::DatasetModelLoss(const Eigen::MatrixXd &x,
                                    const Eigen::MatrixXd &y_truth,
                                    const Eigen::MatrixXd &y_predicted) {
  double loss = 0.0;
  for (int i = 0; i < x.cols(); i++) {
    loss += (y_truth.col(i) - y_predicted.col(i)).squaredNorm();
  }
  return 0.5 * (loss / double(x.cols()));
}