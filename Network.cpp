
#include "Network.hpp"
#include "AbstractLayer.hpp"
#include "LinearLayer.hpp"
#include "ReLuLayer.hpp"
#include "SoftmaxLayer.hpp"
#include "SpikingLayer.hpp"
#include "SigmoidLayer.hpp"
#include "TanhLayer.hpp"

#include <omp.h>
#include <random>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstring>

using namespace Eigen;
using namespace std;

#define RL 1 /* Rectified Linear Activation */
#define SG 2 /* Sigmoid Activation */
#define TH 3 /* Hyperbolic Tangent Activation */

/* File Parsing Errors */
#define LO_ENTR "Runtime Error: Line %d in training file '%s' has insufficient entries."
#define HI_ENTR "Runtime Error: Line %d in training file '%s' has too many entries."
#define BAD_ENT "Runtime Error: Line %d in training file '%s' has a corrupt entry."
#define NOLINES "Runtime Error: Specified training file ('%s') has no training data."
#define CORRUPT "Runtime Error: Specified network file ('%s') is somehow corrupted."
#define DIMS_ER "Runtime Error: One or more of the network spec vectors has incorrect dimension."
#define TYPE_ER "Runtime Error: One or more of the network spec vectors has invalid entries."
#define CLASSER "Runtime Error: Classification problem must have output dimension of at least 2."
#define REG_ERR "Runtime Error: getClass() method called on a network trained for regression."

void Network::raiseError(const char *data, const char *errMssg)
{
  char buffer[strlen(data) + strlen(errMssg) - 1];
  sprintf(buffer, errMssg, data);
  throw runtime_error(buffer);
}

void Network::raiseError(int idx, const char *data, const char *errMssg)
{
  int offset = to_string(idx+1).length() + strlen(data);
  char buffer[offset + strlen(errMssg) - 3];
  sprintf(buffer, errMssg, idx+1, data);
  throw runtime_error(buffer);
}

void Network::parse(MatrixXd &trainInputs, MatrixXd &trainOutputs, string ln, const char *data, int idx, int p, int q)
{
  char *temp; char *ext;
  int i = 0; double token;
  const char *line = ln.c_str();
  temp = strtok(strdup(line), " ");
  while (temp != NULL) {
    token = strtod(temp, &ext);
    if(strlen(ext) != 0) {raiseError(idx, data, BAD_ENT);}
    if(i < p) trainInputs(i, idx) = token;
    else if(reg && i < p+q) trainOutputs(i-p, idx) = token;
    else if(!reg && i == p) trainOutputs((int) token, idx) = 1;
    else raiseError(idx, data, HI_ENTR);
    temp = strtok(NULL, " ");
    i++;
  }
  if(i != p+q) {raiseError(idx, data, LO_ENTR);}
}

int Network::minEntry(VectorXi &vec)
{
  int min = vec(0);
  for(int i = 0; i < vec.size(); i++) {
    if(vec(i) < min) {min = vec(i);}
  }
  return min;
}

int Network::maxEntry(VectorXi &vec)
{
  int max = vec(0);
  for(int i = 0; i < vec.size(); i++) {
    if(vec(i) > max) {max = vec(i);}
  }
  return max;
}

void Network::errorCheck(VectorXi &dims, VectorXi &types, bool doRegression)
{
  bool checka1 = dims.size() == 0;
  bool checka2 = types.size() == 0;
  bool checka3 = dims.size() != types.size() + 2;
  bool checkb1 = minEntry(dims) <= 0;
  bool checkb2 = minEntry(types) <= 0;
  bool checkb3 = maxEntry(types) >= 11;
  if(checka1 || checka2 || checka3) {throw runtime_error(DIMS_ER);}
  if(checkb1 || checkb2 || checkb3) {throw runtime_error(TYPE_ER);}
  if(!doRegression && dims(dims.size()-1) == 1) {throw runtime_error(CLASSER);}
}

int Network::trainingExamples(const char *data)
{
  ifstream input(data);
  string line; int count = 0;
  while(getline(input, line)) {count++;}
  if(count == 0) {raiseError(data, NOLINES);}
  input.close(); return count;
}

void Network::loadData(const char *data, MatrixXd &trainInputs, MatrixXd &trainOutputs)
{
  ifstream input(data);
  int p = trainInputs.rows();
  int q = trainOutputs.rows();
  string line; int count = 0;
  while(getline(input, line)) {
    parse(trainInputs, trainOutputs, line, data, count, p, q); 
    count++;
  }
  input.close();
}

Network::Network(VectorXi &dims, VectorXi &types, const char *data, bool isReg, double alpha, double lambda)
{
  /* Initialize */
  default_random_engine gen;
  errorCheck(dims, types, isReg);
  reg = isReg; int f = types.size();
  
  /* Construct Hidden Layers */
  layers = vector<AbstractLayer *>(types.size()+1);
  deltaW = vector<MatrixXd *>(types.size()+1);
  deltaB = vector<VectorXd *>(types.size()+1);
  delW = vector<MatrixXd *>(types.size()+1);
  delB = vector<VectorXd *>(types.size()+1);
  for(int i = 0; i < types.size(); i++) {
    switch(types(i)) {
    case RL: layers[i] = new ReLuLayer(dims(i+1), dims(i), gen); break;
    case SG: layers[i] = new SigmoidLayer(dims(i+1), dims(i), gen); break;
    case TH: layers[i] = new TanhLayer(dims(i+1), dims(i), gen); break;
    default: layers[i] = new SpikingLayer(dims(i+1), dims(i), gen, types(i));
    }
  }

  /* Constuct Output Layer */
  if(isReg) layers[f] = new LinearLayer(dims(f+1), dims(f), gen);
  else if(dims(f+1) == 2) {dims(f+1) = 1; layers[f] = new SigmoidLayer(1, dims(f), gen);}
  else {layers[f] = new SoftmaxLayer(dims(f), dims(f+1), gen);}

  /* Parse Training Data */
  int dataSize = trainingExamples(data);
  double invm = 1.0 / ((double) dataSize);
  MatrixXd trainInputs = MatrixXd::Zero(dims(0), dataSize);
  MatrixXd trainOutputs = MatrixXd::Zero(dims(f+1), dataSize);
  loadData(data, trainInputs, trainOutputs);
  
  /* Training */
  for(unsigned int i = 0; i < layers.size(); i++) {
    deltaW[i] = new MatrixXd(layers[i]->outDim, layers[i]->inDim);
    delW[i] = new MatrixXd(layers[i]->outDim, layers[i]->inDim);
    deltaB[i] = new VectorXd(layers[i]->outDim);
    delB[i] = new VectorXd(layers[i]->outDim);
  }
  VectorXd input(dims(0));
  VectorXd output(dims(f+1));
  int epoch = 0;
  
  do {
    
    cerr << "Epoch " << epoch << " of training has been started." << endl;
    
    for(unsigned int i = 0; i < layers.size(); i++) {
      *(deltaW[i]) = MatrixXd::Zero(layers[i]->outDim, layers[i]->inDim);
      *(deltaB[i]) = VectorXd::Zero(layers[i]->outDim);
    }

    for(int m = 0; m < dataSize; m++) {
      for(int d = 0; d < dims(0); d++) input(d) = trainInputs(d, m);
      for(int d = 0; d < dims(f+1); d++) output(d) = trainOutputs(d, m);
      computeGradients(input, output);
      for(unsigned int i = 0; i < layers.size(); i++) {
	*(deltaW[i]) += *(delW[i]);
	*(deltaB[i]) += *(delB[i]);
      }
    }

    for(unsigned int i = 0; i < layers.size(); i++) {
      layers[i]->W = layers[i]->W - alpha * (invm * (*(deltaW[i])) + lambda * layers[i]->W);
      layers[i]->b = layers[i]->b - alpha * (invm * (*(deltaB[i])));
    }

    cerr << "Epoch " << epoch << " of training has been completed." << endl;
    epoch++;
    
  } while (epoch <= 10000);

}

Network::Network(const char *pubFile)
{
  /* TODO ENABLE FILE READING */
}

Network::~Network() {
  for(unsigned int i = 0; i < layers.size(); i++) {
    delete layers[i];
    delete deltaW[i];
    delete deltaB[i];
    delete delW[i];
    delete delB[i];
  }
}

void Network::publishNetwork(const char *pubFile)
{
  ofstream ofs(pubFile);
  for(auto layer: layers) {
    ofs << layer->W << endl;
    ofs << layer->b << endl;
  }
  ofs.close();
}

void Network::computeGradients(VectorXd &input, VectorXd &output)
{
  vector<VectorXd *> act(layers.size());
  vector<VectorXd *> grad(layers.size());
  computeValues(input, act, grad);  

  int fIdx = layers.size()-1;
  VectorXd argmt = *(act[fIdx]) - output;
  *(delB[fIdx]) = argmt.cwiseProduct(*(grad[fIdx]));
  for(int i = fIdx - 1; i >= 0; i--) {
    argmt = (layers[i+1]->W).transpose() * (*(delB[i+1]));
    *(delB[i]) = argmt.cwiseProduct(*(grad[i]));
  }
  
  for(int i = fIdx; i > 0; i--) {
    *(delW[i]) = (*(delB[i])) * (*(act[i-1])).transpose();
  }
  *(delW[0]) = (*(delB[0])) * input.transpose();

  for(unsigned int i = 0; i < layers.size(); i++) {
    delete act[i]; delete grad[i];
  }
}
    
void Network::computeValues(VectorXd &input, vector<VectorXd *> &act, vector<VectorXd *> &grad)
{
  Eigen::VectorXd scaled;
  for(unsigned int i = 0; i < layers.size(); i++) {
    act[i] = new VectorXd(layers[i]->outDim);
    grad[i] = new VectorXd(layers[i]->outDim);
  }

  layers[0]->computeScaled(input, scaled);  
  layers[0]->activation(scaled, *(act[0]));
  layers[0]->gradient(scaled, *(grad[0]));
  
  for(unsigned int i = 1; i < layers.size(); i++) {
    layers[i]->computeScaled(*(act[i-1]), scaled);
    layers[i]->activation(scaled, *(act[i]));
    layers[i]->gradient(scaled, *(grad[i]));
  }
}

void Network::runNetwork(VectorXd &input, VectorXd &output)
{
  Eigen::VectorXd scaled;
  Eigen::VectorXd activation = input;
  for(auto layer: layers) {
    layer->computeScaled(activation, scaled);
    activation = VectorXd(scaled.size());
    layer->activation(scaled, activation);
  }
  output = activation;
}

int Network::getClassLabel(VectorXd &input)
{
  if(reg) {throw runtime_error(REG_ERR);}
  VectorXd output; runNetwork(input, output);

  /* Binary Classification */
  if(output.size() == 1) {
    if(output(0) > 0.5) {return 1;}
    else {return 0;}
  }

  /* Multi-Class Case */
  int maxIndex, max = output(0);
  for(int i = 0; i < output.size(); i++) {
    if(max > output(i)) {
      max = output(i);
      maxIndex = i;
    }
  }
  return maxIndex+1;
}
