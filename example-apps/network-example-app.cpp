#include <Eigen/Dense>

#include "common/data_utils.hpp"
#include "network/neural_network.hpp"

int main(int argc, char *argv[]) {
  // Generate Training Dataset
  std::cerr << "Creating Dataset..." << std::endl;
  std::pair<Eigen::MatrixXd, Eigen::MatrixXd> dataset = MakeCategoricalDataset(
      data_utils::CategoricalDatasetType::Concentric, 1000);
  Eigen::MatrixXd x = dataset.first, y = dataset.second;

  // Train Network on Training Dataset
  std::cerr << "Training Model..." << std::endl;
  std::vector<int> layerDimensions = {2, 25, 3};
  std::vector<LayerType> layerTypes = {LayerType::ReLU, LayerType::Softmax};
  Network network(layerTypes, layerDimensions);
  network.Train(x, y, 0.01, 0.0001, 50);

  // Generate Evaluation Dataset
  std::cerr << "Evaluating Model..." << std::endl;
  Eigen::MatrixXd t = data_utils::UniformSampling(1000);
  Eigen::MatrixXd z = network.RunDataset(t);

  // Publish Results from Evaluation Dataset
  std::cerr << "Publishing Results..." << std::endl;
  data_utils::DumpDataset(x, y, "/tmp/true.txt");
  data_utils::DumpDataset(t, z, "/tmp/eval.txt");

  if (argc > 1) {
    // Visualize Results (argv[1] must be path to python script.)
    std::string vis_command = std::string("python3 ") + argv[1];
    std::string vis_flags(" --true /tmp/true.txt --eval /tmp/eval.txt");
    system((vis_command + vis_flags + " --type classification").c_str());
  }
}