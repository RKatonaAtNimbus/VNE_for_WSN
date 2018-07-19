# VNE_for_WSN
QoS/QoI aware Virtual Network Embedding Algorithm for Wireless Sensor Network

## Input vectors

Data used in the experiments to validate the algorithm was stored in input vectors,
which are made available.

There are input vectors for networks of 50, 100 and 150 nodes. Each network size
was tested with batches of 1 to 8 VNRs. For each (nwk size, number of vnrs) pair,
100 tests were done.

In each test the following were randomized:
* network topology
* sink position
* temperature process
* VNR parameters.

The input vectors are stored as _pickle_ files, each file contains a list of 100
elements, each a dictionary with the following keys:
* nwksize,
* substrate\_coords
* interests,
* temp\_process,
* adjacency\_matrix,
* iteration,
* numvn,
* vnlist.


## Mixed Integer Linear Programming representation

The performance of the heuristics is evaluated relative to exact optimal solutions
obtained by a solving a MILP representation of the problem with the CBC (Coin-Or
Branch and Cut) solver.

### VNR location constraint
VNRs specify their prefered sampling location using an interest point as well as 
the tolerated measurement error. The two parameters are used to define a set of 
candidate nodes for each VNR.

In the manner of Chowdhury _et al_ [TODO], we define meta nodes for each VNR, which
are connected by 100% reliable links to all the candidate nodes of the VNR.

### Constraints
The following constraints are imposed:
* Flow conservation
* Link capacity
* End-to-end path reliability
* Single source rule - only one connection between a meta node and a physical one.

### Objectives
Two objectives are used:
1. Maximize acceptance ratio
2. Minimize cost.

The two objectives are used as separate programs. First the acceptance ratio 
objective is used to determine the highest acceptance ratio for an IV. If the 
highest number of accepted VNRs is k (out of n), the next step is to solve the
problem with the cost minimization objective for all k out of n combinations of the
VNRs.

### Requirements

Solving the MILP requires:
* PyOmo framework: http://www.pyomo.org/; "pip install Pyomo"
  * Pyomo is used to represent the MILP and interface with the solver.
* CBC solver: https://projects.coin-or.org/Cbc
  * CBC is not a Python application, it's a separate binary that will be called
    by Pyomo.
* NetworkX for processing input vectors.

### Usage

The script "test\_script.py" will run the algorithm on an entire input vector,
processing all the iterations in the vector.

Usage:
~~~
test_script.py input_vector.pickle
~~~

The script generates a solution vector corresponding to the input vector.

The MILP is implemented using the PyOmo framework in
* mc\_flow\_lp.py -- maximize acceptance ratio objective
* mc\_flow\_cost.py -- minimize cost objective.

Before running the MILP the input vector must be translated into the PyOmo format.
This is done in test\_vne.py.

Two functions are used from test\_vne.py:
* algorithm -- takes as input the input vector iteration and runs the MILP with
acceptance ratio maximization
* find\_best\_cost\_fixed\_acceptance -- takes as input the input vector iteration
and the number of accepted VNRs, runs MILP with cost minimization.

