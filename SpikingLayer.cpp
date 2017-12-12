
#include <cmath>
#include <omp.h>
#include <iostream>
#include "SpikingLayer.hpp"

using namespace Eigen;
using namespace std;

#define RS 4 /* Regular Spiking Neuron */
#define IB 5 /* Intrinsically Bursting Neuron */
#define CH 6 /* Chattering Neuron */
#define FS 7 /* Fast Spiking Neuron */
#define TC 8 /* Thalmo-Cortical Neuron */
#define RZ 9 /* Resonator Neuron */
#define LT 10 /* Low-Threshold Spiking Neuron */

SpikingLayer::SpikingLayer(int numNeurons, int numInputs, default_random_engine &gen, int type)
  : AbstractLayer(numNeurons, numInputs, gen)
{
  switch(type) {
  case RS: a = 0.02; b = 0.2; c = -65.0; d = 8.0; break;
  case IB: a = 0.02; b = 0.2; c = -55.0; d = 4.0; break;
  case CH: a = 0.02; b = 0.2; c = -50.0; d = 2.0; break;
  case FS: a = 0.1; b = 0.2; c = -65.0; d = 2.0; break;
  case TC: a = 0.02; b = 0.25; c = -65.0; d = 0.05; break;
  case RZ: a = 0.1; b = 0.25; c = -65.0; d = 2.0; break;
  case LT: a = 0.02; b = 0.25; c = -65.0; d = 2.0; break;
  default: a = 0.02; b = 0.2; c = -65.0; d = 8.0; break;
  }
  baseCurrent = 4.0; /* Hardcoded for RS Neuron */
  baseRate = rawSpikeRate(baseCurrent);
}

SpikingLayer::~SpikingLayer() {}

double SpikingLayer::scaledSpikeRate(double i)
{
  if(i > 0) return rawSpikeRate(i + baseCurrent) - baseRate;
  return baseRate - rawSpikeRate(-i + baseCurrent);
}

double SpikingLayer::rawSpikeRate(double i)
{
  double v = c;
  double u = 0.0;
  double time = 0.0;
  double dt = 0.01;
  double endTime = 0.0;
  double startTime = 0.0;
  bool firstSpike = true;
  while(true) {
    if(v >= 30) {
      v = c; u = u + d;
      if(firstSpike) {
	startTime = time;
	firstSpike = false;
      } else {
	endTime = time;
	break;
      }
    }
    v += (0.04*v*v + 5*v + 140 - u + i) * dt;
    u += (a * (b*v - u)) * dt;
    time = time + dt;
  }
  return 1.0/(endTime - startTime);
}

void SpikingLayer::activation(VectorXd &input, VectorXd &output)
{
#pragma omp parallel for
  for(int i = 0; i < input.size(); i++) {
    output(i) = scaledSpikeRate(input(i));
  }
}

void SpikingLayer::gradient(VectorXd &input, VectorXd &output)
{
  double sr1, sr2; /* Gradient Approximation */
#pragma omp parallel for 
  for(int i = 0; i < input.size(); i++) {
    sr1 = scaledSpikeRate(input(i) - 0.01);
    sr2 = scaledSpikeRate(input(i) + 0.01);
    output(i) = (sr2 - sr1) / (0.02);
  }
}
