OBJS = Abstract.o Softmax.o Sigmoid.o Spiking.o Tanh.o ReLu.o Linear.o
OBJS1 = $(OBJS) layerTestProgram.cpp
OBJS2 = $(OBJS) networkTestProgram.cpp Network.o
CC = g++
DEBUG = -g
CFLAGS = -std=c++11 -fopenmp -Wall -c $(DEBUG)
LFLAGS = -std=c++11 -fopenmp -Wall $(DEBUG)

all : layerTestProgram networkTestProgram

networkTestProgram : $(OBJS2)
	$(CC) $(LFLAGS) $(OBJS2) -o networkTestProgram

layerTestProgram : $(OBJS1)
	$(CC) $(LFLAGS) $(OBJS1) -o layerTestProgram

Network.o : Network.hpp Network.cpp
	$(CC) $(CFLAGS) Network.cpp -o Network.o

Abstract.o : AbstractLayer.hpp AbstractLayer.cpp
	$(CC) $(CFLAGS) AbstractLayer.cpp -o Abstract.o

Softmax.o : SoftmaxLayer.hpp SoftmaxLayer.cpp
	$(CC) $(CFLAGS) SoftmaxLayer.cpp -o Softmax.o

Sigmoid.o : SigmoidLayer.hpp SigmoidLayer.cpp
	$(CC) $(CFLAGS) SigmoidLayer.cpp -o Sigmoid.o

Spiking.o : SpikingLayer.hpp SpikingLayer.cpp
	$(CC) $(CFLAGS) SpikingLayer.cpp -o Spiking.o

Tanh.o : TanhLayer.hpp TanhLayer.cpp
	$(CC) $(CFLAGS) TanhLayer.cpp -o Tanh.o

ReLu.o : ReLuLayer.hpp ReLuLayer.cpp
	$(CC) $(CFLAGS) ReLuLayer.cpp -o ReLu.o

Linear.o : LinearLayer.hpp LinearLayer.cpp
	$(CC) $(CFLAGS) LinearLayer.cpp -o Linear.o

clean:
	\rm *.o *~ layerTestProgram networkTestProgram
