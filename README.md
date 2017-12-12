This repository provides a C++ deep learning utility for training and testing a general feed-forward neural network. The user can specify the architecture of the network (i.e. number of layers, neurons per layer, layer activation functions), the purpose of the network (either regression or classification), and the training data. The utility uses backpropagation and stochastic gradient descent for tuning the layer weight matrices based on the training examples (see http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/). Each element of each weight matrix and bias vector is initialized according to a gaussian. A regularization (weight decay) term is applied to the cost function.

Dependencies
============

* Eigen (http://eigen.tuxfamily.org/)
* OpenMP (http://www.openmp.org/)

Network Training
================

Networks can be specified and trained via:

  `Network::Network(Eigen::VectorXi &dims, Eigen::VectorXi &types, const char *data, bool regClass, double alpha, double lambda);`

The dims vector specifies the number of neurons to place in each layer of the network. Therefore, the first entry of this vector is the input dimension (P) and the last entry is the output dimension (Q). Note that for regression problems, Q is the number of individual outputs to be computed, and for classification problems, Q is the number of possible class labels. The types vector specifies the activation functions to be used for the hidden layers of the network. Each entry of this vector must be an integer in the range [0,10], and the length of this vector must be two less than the length of the dims vector. The activation functions are encoded as follows:

0: Linear Activation
1: Rectified Linear Activation
2: Sigmoid Activation
3: Hyperbolic Tangent Activation
4: Regular Spiking Neuron
5: Intrinsically Bursting Neuron
6: Chattering Neuron
7: Fast Spiking Neuron
8: Thalmo-Cortical Neuron
9: Resonator Neuron
10: Low-Threshold Spiking Neuron

Note that options 5->10 are activation functions which result from biologically inspired neuron models. For each of these activation functions, the output is the firing rate (for a given input) that is exhibited by a certain type of neuron (see https://www.izhikevich.org/publications/spikes.htm).

The constructor also accepts a (const) char *, which contains the relative path to the training data file. The next argument is a boolean which should be true if the network should do regression and false if it should do classification. In regression, a linear activation is used on the output layer. Note that for classification, we must specify Q >= 2. A sigmoid activation is used on the output when Q = 2, and a softmax activation is used on the output when Q > 2. The training data file should be structured as follows. Each line should contain a single training example. The inputs should be followed immediately by the outputs (or class labels), space separated. Therefore, for regression problems, each line should have P+Q entries. For classification problems, each line should have P+1 entries, where the last entry in an integer in the set {0,...,Q-1}. The final two arguments specify alpha and lambda, which are, respectively, the learning rate of the gradient descent algorithm and the regularization parameter.

Network Publishing and Reading
==============================

Once a network has been constructed and trained, it can be pubished via:

  `void publishNetwork(const char *pubFile);`

Here, the const char * that is accepted is simply the relative path to the desired publication file. Networks can be read from files that have been produced using the above method via the secondary constructor: 

  `Network(const char *pubFile);`

Running the Network
===================

A constructed network may be run on some input via:

 `void runNetwork(Eigen::VectorXd &input, Eigen::VectorXd &output);`

This method accepts by reference an Eigen::VectorXd that is of the same dimension as the input layer of the network and another that is of the same dimension as the output layer. Based on the contents of the former, it does forward-propagation over the layers and then populates the contents of the latter.
