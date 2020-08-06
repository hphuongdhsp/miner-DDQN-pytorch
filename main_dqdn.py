
import numpy as np
from ddqn_agent import DDQNAgent
from MinerEnv import plot_learning_curve, MinerEnv
import sys
import random
import pandas as pd
import datetime 
import json
from pathlib import Path

FILE_PATH = str(Path(__file__).parent.resolve())

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
filename = FILE_PATH + "/Data/data_" + now.strftime("%Y%m%d-%H%M") + ".csv" 
with open(filename, 'w') as f:
    pd.DataFrame(columns=header).to_csv(f, encoding='utf-8', index=False, header=True)

# Parameters for training a DQN model
epsilon = 1
  #The number of steps for each episode
BATCH_SIZE =  32  #The number of experiences for each replay 
MEMORY_SIZE = 10000 #The size of the batch for storing experiences
SAVE_NETWORK = 200 # After this number of episodes, the DQN model is saved for testing later. 
image_dim = (24,21*5,9*5)
 #The number of input values for the DQN model
ACTION_NUM = 6  #The number of actions output from the DQN model
MAP_MAX_X = 21 #Width of the Map
MAP_MAX_Y = 9  #Height of the Map


agent = DDQNAgent(gamma=0.99, epsilon= epsilon, lr=0.001,
                     input_dims= image_dim,
                     n_actions=ACTION_NUM, mem_size=MEMORY_SIZE, eps_min=0.05,
                     batch_size=BATCH_SIZE, replace=10000, eps_dec=1e-5,
                     chkpt_dir=FILE_PATH + '/weights/', algo='DDQNAgent',
                     env_name='miner')

load_checkpoint = False
if load_checkpoint:
        agent.load_models()
n_games = 20000
minerEnv = MinerEnv(HOST, PORT) #Creating a communication environment between the DQN model and the game environment (GAME_SOCKET_DUMMY.py)
minerEnv.start()  # Connect to the game
path = FILE_PATH + '/Maps/'

fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'
    
figure_file = FILE_PATH + '/' + fname + '.png'
best_score = -np.inf


n_steps = 0
scores, eps_history, steps_array = [], [], []


for  episode_i in range(n_games):
    done = False
    n_steps = 0
    try: 
        mapID = np.random.randint(0, 1)
        maplist = [0,25,50,75,100]     ##
        mapID = random.choice(maplist)  
        with open(path +"map"+str(mapID)) as json_file:
            data = json.load(json_file)
        Map = np.array(data)
        index = np.where(Map==0)
        listOfCoordinates= list(zip(index[0], index[1]))

        list_index = random.choice(listOfCoordinates)
        posID_x = list_index[1] #Choosing a initial position of the DQN agent on X-axes randomly
        posID_y = list_index[0] 
        request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100") 
        #Send the request to the game environment (GAME_SOCKET_DUMMY.py)
        minerEnv.send_map_info(request)

        # Getting the initial state
        minerEnv.reset() #Initialize the game environment
        s = minerEnv.get_state()
        score = 0       
        observation = np.reshape([np.stack((s, s, s, s), axis=0)],    
                                 (24, image_dim[1], image_dim[2]))# stack state
        
        while not done: 
            action = agent.choose_action(observation)
            minerEnv.step(str(action))
            ### make env.step 
            s_next = minerEnv.get_state()  # next state
            observation_ = np.append(np.reshape(s_next, (6, image_dim[1],image_dim[2])), observation[:18, :, :], axis=0) # new_observation 
            reward = minerEnv.get_reward(action) # get reward
            done = minerEnv.check_terminate() # get terminal
            end_code = finish_name[int(minerEnv.state.status)]
            score += reward
            ######
            #if not load_checkpoint:
            agent.store_transition(observation, action,
                                     reward, observation_, int(done))
            agent.learn()

            observation = observation_
            n_steps += 1

            save_data = np.hstack(
                [episode_i + 1, n_steps + 1, minerEnv.state.energy, reward, score, 
                 action_name[int(action)], 
                 end_code]).reshape(1, 7)
            with open(filename, 'a') as f:
                pd.DataFrame(save_data).to_csv(f, encoding='utf-8', index=False, header=False)

        avg_score = np.mean(scores[-100:])
        print('episode %.1f ' % episode_i ,'score %.2f' % score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps)
        
        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score
        steps_array.append(episode_i)
        scores.append(score)
        eps_history.append(agent.epsilon)

    except :
        import traceback
        traceback.print_exc()
    if np.mod(episode_i + 1, SAVE_NETWORK) == 0:
        agent.save_models()

plot_learning_curve(steps_array, scores, eps_history, figure_file) ## plot the result

