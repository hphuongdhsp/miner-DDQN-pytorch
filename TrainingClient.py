import sys
from DQNModel import DQN # A class of creating a deep q-learning model
from MinerEnv_P import MinerEnv # A class of creating a communication environment between the DQN model and the GameMiner environment (GAME_SOCKET_DUMMY.py)
from Memory import Memory # A class of creating a batch in order to store experiences for the training process
import random
import pandas as pd
import datetime 
import numpy as np
import json
#     STATUS_PLAYING = 0
    # STATUS_ELIMINATED_WENT_OUT_MAP = 1
    # STATUS_ELIMINATED_OUT_OF_ENERGY = 2
    # STATUS_ELIMINATED_INVALID_ACTION = 3
    # STATUS_STOP_EMPTY_GOLD = 4
    # STATUS_STOP_END_STEP = 5
finish_name=['STATUS_PLAYING','STATUS_ELIMINATED_WENT_OUT_MAP','STATUS_ELIMINATED_OUT_OF_ENERGY',
             'STATUS_ELIMINATED_INVALID_ACTION','STATUS_STOP_EMPTY_GOLD','STATUS_STOP_END_STEP']
action_name=['Go left', 'Go right','Go up','Go down','Rest','Dig for gold']
HOST = "localhost"
PORT = 1111
if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = str(sys.argv[2])

# Create header for saving DQN learning file
now = datetime.datetime.now() #Getting the latest datetime
header = ["Ep", "Step",'energy', "Reward", "Total_reward", "Action", "Epsilon", "Termination_Code"] #Defining header for the save file
filename = "Data/data_" + now.strftime("%Y%m%d-%H%M") + ".csv" 
with open(filename, 'w') as f:
    pd.DataFrame(columns=header).to_csv(f, encoding='utf-8', index=False, header=True)

# Parameters for training a DQN model
N_EPISODE = 50000 #The number of episodes for training
epsilon = 1
MAX_STEP = 100   #The number of steps for each episode
BATCH_SIZE =  32  #The number of experiences for each replay 
MEMORY_SIZE = 1000 #The size of the batch for storing experiences
SAVE_NETWORK = 200  # After this number of episodes, the DQN model is saved for testing later. 
INITIAL_REPLAY_SIZE = 100 #The number of experiences are stored in the memory batch before starting replaying
input_image_dim = (21,9,24)
 #The number of input values for the DQN model
ACTIONNUM = 6  #The number of actions output from the DQN model
MAP_MAX_X = 21 #Width of the Map
MAP_MAX_Y = 9  #Height of the Map
#mapID = np.random.randint(0, 1) 
'''
map0 = [[0,-2,0,-2,0,-1,0,0,0,0,-1,-2,-2,-2,0,0,800,500,0,-2,0],
  [-2,-2,-2,-2,-1,0,-1,-1,-1,-1,-3,650,-2,-2,-2,-2,-3,-3,500,-2,-1],
  [-2,-2,0,-2,0,-2,600,-2,-3,-3,-2,100,-3,-2,-2,0,-3,-3,0,0,0],
  [0,-3,-3,-2,0,0,-1,0,0,-3,-2,0,0,100,-1,0,0,-1,-1,-1,-2],
  [-2,850,1100,0,-1,100,-1,450,1050,-3,-2,0,-3,350,0,0,-1,-3,-3,-2,-1],
  [-1,-3,-1,-3,0,-2,0,0,-2,-1,0,-3,400,-2,0,-3,700,-2,-3,-2,0],
  [-2, -3,-1,-3,-1,0,-1,-3,-2,-1,300,-1,0,-1,200,-1,150,-2,-3,-3,-1],
  [0, -3, -1, -3, 0,-2,-3,-3,0,0,0,0,-2,300,-2,-3,-3,-3,-3,0,-1],
  [0,-3,-1,-3,-1,-1,-2,-2,0,-1,0,-2,0,-2,0,0,-2,-3,-3,0,0]
]
Map = np.array(map0.copy())
index = np.where(Map==0)
listOfCoordinates= list(zip(index[0], index[1]))
'''
# Initialize a DQN model and a memory batch for 
# storing experiences
weight_path='/home/mayleo/Documents/Inreforcement learning/miner/TrainedModels/DQNmodel_MinerLoss_ep600.h5'
DQNAgent = DQN(input_image_dim, ACTIONNUM, gamma = 0.95,epsilon =epsilon, learning_rate = 0.01, load_weights=None)
memory = Memory(MEMORY_SIZE)

# Initialize environment
minerEnv = MinerEnv(HOST, PORT) #Creating a communication environment between the DQN model and the game environment (GAME_SOCKET_DUMMY.py)
minerEnv.start()  # Connect to the game
path = '/home/mayleo/Documents/Inreforcement learning/miner/Maps/'
train = False #The variable is used to indicate that the replay starts, and the epsilon starts decrease.
#Training Process
#the main part of the deep-q learning agorithm 
for episode_i in range(0, N_EPISODE):
    print('*****')
    try:
        # Choosing a map in the list
        mapID = np.random.randint(0, 1) #Choosing a map ID from 5 maps in Maps folder randomly
        maplist = [0,25,50,75,100]
        mapID = random.choice(maplist)   

        with open(path +"map"+str(mapID)) as json_file:
            data = json.load(json_file)
        Map = np.array(data)
        index = np.where(Map==0)
        listOfCoordinates= list(zip(index[0], index[1]))

        list_index = random.choice(listOfCoordinates)
        posID_x = list_index[1] #Choosing a initial position of the DQN agent on X-axes randomly
        posID_y = list_index[0] 

        #posID_x = np.random.randint(MAP_MAX_X) #Choosing a initial position of the DQN agent on X-axes randomly
        #posID_y = np.random.randint(MAP_MAX_Y) #Choosing a initial position of the DQN agent on Y-axes randomly
        #Creating a request for initializing a map, initial position, the initial energy, and the maximum number of steps of the DQN agent
        request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100") 
        #Send the request to the game environment (GAME_SOCKET_DUMMY.py)
        minerEnv.send_map_info(request)

        # Getting the initial state
        minerEnv.reset() #Initialize the game environment
        s = minerEnv.get_state()

        #history = [np.stack((s[0], s[0], s[0], s[0]), axis=2),
        #           np.stack((s[1], s[1], s[1], s[1]), axis=1)]

        history = np.reshape([np.stack((s, s, s, s), axis=2)], (1, 21, 9, 24))
                    



        #Get the state after reseting. 
                                #This function (get_state()) is an example of creating a state for the DQN model 
        loss_eps = 0
        count = 0
        total_reward = 0 #The amount of rewards for the entire episode
        terminate = False #The variable indicates that the episode ends
        maxStep = minerEnv.state.mapInfo.maxStep #Get the maximum number of steps for each episode in training
        #Start an episde for training
        for step in range(0, maxStep):
            count = 0
            # print('****')
            #print(step)
            action = DQNAgent.act(history)  # Getting an action from the DQN model from the state (s)
            # print('step', step,'action',action)
            minerEnv.step(str(action))  # Performing the action in order to obtain the new state
            s_next = minerEnv.get_state()  # Getting a new state
            #print('pass')
            next_history = np.append(np.reshape(s_next, (1, 21,9, 6)), history[:, :, :, :18], axis=3)
                            
            reward = minerEnv.get_reward()  # Getting a reward
            terminate = minerEnv.check_terminate()  # Checking the end status of the episode
            end_code = finish_name[int(minerEnv.state.status)]
            # Add this transition to the memory batch
            memory.push(history, action, reward, terminate, next_history)
            #print(memory.historys[-1])
            # Sample batch memory to train network
            if (memory.length > INITIAL_REPLAY_SIZE):
                #If there are INITIAL_REPLAY_SIZE experiences in the memory batch
                #then start replaying
                batch = memory.sample(BATCH_SIZE) #Get a BATCH_SIZE experiences for replaying
                loss = DQNAgent.replay(batch, BATCH_SIZE)#Do relaying
                train = True #Indicate the training starts
                loss_eps = + loss
                count = +1
            total_reward = total_reward + reward #Plus the reward to the total rewad of the episode
            history = next_history #Assign the next state for the next step.

            # Saving data to file
            save_data = np.hstack(
                [episode_i + 1, step + 1,minerEnv.state.energy, reward, total_reward, 
                 action_name[int(action)], DQNAgent.epsilon, 
                 end_code]).reshape(1, 8)
            with open(filename, 'a') as f:
                pd.DataFrame(save_data).to_csv(f, encoding='utf-8', index=False, header=False)
            
            if terminate == True:
                #If the episode ends, then go to the next episode
                break

        # Iteration to save the network architecture and weights
        if (np.mod(episode_i + 1, SAVE_NETWORK) == 0 and train == True):
            DQNAgent.target_train()  # Replace the learning weights for target model with soft replacement
            #Save the DQN model
            now = datetime.datetime.now() #Get the latest datetime
            DQNAgent.save_model("TrainedModels/",
                                "DQNmodel_MinerLoss_ep" + str(episode_i + 1))

        
        #Print the training information after the episode
        print('Episode %d ends. Number of steps is: %d. Accumulated Reward = %.2f. Loss =  %.2f.  Epsilon = %.2f .Termination code: %s' % (
            episode_i + 1, step + 1, total_reward, loss_eps/(count+ 0.0001), DQNAgent.epsilon, end_code))
        
        #Decreasing the epsilon if the replay starts
        if train == True:
            DQNAgent.update_epsilon()

    except Exception as e:
        import traceback

        traceback.print_exc()
        # print("Finished.")
        break
