#ifndef NETWORK_HPP_
#define NETWORK_HPP_

#include "AbstractLayer.hpp"
#include <vector>
#include <Eigen/Dense>

class Network {

public:
  
  void raiseError(const char *data, const char *errMssg);
  void raiseError(int idx, const char *data, const char *errMssg);
  void parse(Eigen::MatrixXd &trainInputs, Eigen::MatrixXd &trainOutputs, std::string ln, const char *data, int idx, int p, int q);
  int minEntry(Eigen::VectorXi &vec);
  int maxEntry(Eigen::VectorXi &vec);
  void errorCheck(Eigen::VectorXi &dims, Eigen::VectorXi &types, bool doRegression);
  int trainingExamples(const char *data);
  void loadData(const char *data, Eigen::MatrixXd &trainInputs, Eigen::MatrixXd &trainOutputs);
  Network(Eigen::VectorXi &dims, Eigen::VectorXi &types, const char *data, bool isReg, double alpha, double lambda);
  Network(const char *pubFile);
  ~Network();
  void publishNetwork(const char *pubFile);
  void computeGradients(Eigen::VectorXd &input, Eigen::VectorXd &output);
  void computeValues(Eigen::VectorXd &input, std::vector<Eigen::VectorXd *> &act, std::vector<Eigen::VectorXd *> &grad);
  void runNetwork(Eigen::VectorXd &input, Eigen::VectorXd &output);
  int getClassLabel(Eigen::VectorXd &input);

  std::vector<AbstractLayer *> layers;
  std::vector<Eigen::MatrixXd *> deltaW;
  std::vector<Eigen::VectorXd *> deltaB;
  std::vector<Eigen::MatrixXd *> delW;
  std::vector<Eigen::VectorXd *> delB;
  
  bool reg;
  
};

#endif
