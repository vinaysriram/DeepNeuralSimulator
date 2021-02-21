
#ifndef DATA_UTILS_H_
#define DATA_UTILS_H_

#include <Eigen/Dense>

namespace data_utils {

enum class NumericalDatasetType {
  EllipticParaboloid,    // Numerical dataset, used for regression.
  HyperbolicParaboloid,  // Numerical dataset, used for regression.
  SinusoidProduct,       // Numerical dataset, used for regression.
  SemiEllipsoid,         // Numerical dataset, used for regression.
  ConcaveClover          // Numerical dataset, used for regression.
};

enum class CategoricalDatasetType {
  ZigZagLines,   // Categorical dataset, used for classification.
  Intersection,  // Categorical dataset, used for classification.
  Concentric,    // Categorical dataset, used for classification.
  SpiralArms,    // Categorical dataset, used for classification.
  Clustered,     // Categorical dataset, used for classification.
};

// Creates a new dataset of type datasetType and loads it into (x, y).
// Currently, all numerical datasets have 1-dimensional output.
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> MakeNumericalDataset(
    NumericalDatasetType datasetType, int numSamples);

// Creates a new dataset of type datasetType and loads it into (x, y).
// Currently, all categorical datasets have 3-dimensional output, where
// each output is a one-hot vector representing the correct class.
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> MakeCategoricalDataset(
    CategoricalDatasetType datasetType, int numSamples);

// Generates uniformly sampled points in 2D space.
Eigen::MatrixXd UniformSampling(int numSamples);

// Loads an on-disk dataset with filepath datasetFile into (x, y). The dataset
// must be formatted as follows: the first line must contain exactly three
// space-separated positive integers: the first is the number of samples in
// the dataset, the second is the input dimension, and the third is the output
// dimension. Every subsequent line must contain one sample, with the input
// and output vectors space-separated and concatenated. Note that all datasets
// written to a file using DumpDataset method are readable using LoadDataset.
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> LoadDataset(
    const std::string &datasetFile);

// Writes an in-memory dataset (x, y) to a file with filepath datasetFile. The
// output format follows the specification above that LoadDataset accepts.
void DumpDataset(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y,
                 const std::string &datasetFile);

// Given a categorical dataset with inputs x, ground truth outputs y_truth and
// model-predicted outputs y_predicted, this method returns a square confusion
// matrix M of size k x k, where k is the total number of class labels. Entry
// M(i, j) represents the number of samples in the dataset that have true
// label i and predicted label j. Therefore, the sum of the i^(th) row is the
// total number of samples belonging to class i, and the sum of the j^(th)
// column is the total number of samples that have been predicted to be of
// class j. A perfect classifier only has non-zero entries along the diagonal.
Eigen::MatrixXi ConfusionMatrix(const Eigen::MatrixXd &x,
                                const Eigen::MatrixXd &y_truth,
                                const Eigen::MatrixXd &y_predicted);

// Given a dataset with inputs x, ground truth outputs y_truth and
// model-predicted outputs y_predicted, this method returns the loss.
double DatasetModelLoss(const Eigen::MatrixXd &x,
                        const Eigen::MatrixXd &y_truth,
                        const Eigen::MatrixXd &y_predicted);

}  // namespace data_utils

#endif