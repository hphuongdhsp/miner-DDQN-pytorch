This repository is a python code for training the Double DQN model - Miner game (by Pytorch).

The state of the DDQN is a matrix of 6 dim. The first and second dimensions are for the map, and the last four dimensions are for players. The observation of the CNN is a matrix fixed size 24x(21*5)x(9*5), which is stacked of 4 last states. 
