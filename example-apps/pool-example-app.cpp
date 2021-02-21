#include <Eigen/Dense>
#include <iostream>

#include "common/data_utils.hpp"
#include "pool/neural_pool.hpp"

int main(int argc, char *argv[]) {
  // Generate Training Dataset
  std::cerr << "Creating Dataset..." << std::endl;
  std::pair<Eigen::MatrixXd, Eigen::MatrixXd> dataset = MakeNumericalDataset(
      data_utils::NumericalDatasetType::SinusoidProduct, 1000);
  Eigen::MatrixXd x = dataset.first, y = dataset.second;

  // Create and Train Pool
  std::cerr << "Training Model..." << std::endl;
  Pool pool(25, 2, NeuronType::RS);
  pool.Train(x, y);

  // Generate Evaluation Dataset
  std::cerr << "Evaluating Model..." << std::endl;
  Eigen::MatrixXd t = data_utils::UniformSampling(1000);
  Eigen::MatrixXd z = pool.RunDataset(t);

  // Publish Results from Evaluation Dataset
  std::cerr << "Publishing Results..." << std::endl;
  data_utils::DumpDataset(x, y, "/tmp/true.txt");
  data_utils::DumpDataset(t, z, "/tmp/eval.txt");

  if (argc > 1) {
    // Visualize Results (argv[1] must be path to python script.)
    std::string vis_command = std::string("python3 ") + argv[1];
    std::string vis_flags(" --true /tmp/true.txt --eval /tmp/eval.txt");
    system((vis_command + vis_flags + " --type regression").c_str());
  }
}