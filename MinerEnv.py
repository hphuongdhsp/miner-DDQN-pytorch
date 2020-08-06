import numpy as np
from GAME_SOCKET_DUMMY import GameSocket #in testing version, please use GameSocket instead of GAME_SOCKET_DUMMY
from MINER_STATE import State
import json
import matplotlib.pyplot as plt

def str_2_json(str):
    return json.loads(str, encoding="utf-8")

TreeID = 1
TrapID = 2
SwampID = 3
class MinerEnv:
    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
        self.state = State()
        self.pre_x = 0
        self.pre_y = 0
        self.pre_energy = 0
        #self.pre_action = ''   
        self.score_pre = self.state.score#Storing the last score for designing the reward function

    def start(self): #connect to server
        self.socket.connect()

    def end(self): #disconnect server
        self.socket.close()

    def send_map_info(self, request):#tell server which map to run
        self.socket.send(request)

    def reset(self): #start new game
        try:
            message = self.socket.receive() #receive game info from server
            self.state.init_state(message) #init state
        except:
            import traceback
            traceback.print_exc()

    def step(self, action): #step process
        #self.pre_action = action
        self.pre_energy = self.state.energy
        self.pre_x, self.pre_y = self.state.x,self.state.y # store the last coordinate
        self.socket.send(action) #send action to server
        try:
            message = self.socket.receive() #receive new state from server
            self.state.update_state(message) #update to local state
        # new_state = str_2_json(message)
        # players = new_state["players"]
        # print('length of players in step', len(players))
        except:
            import traceback
            traceback.print_exc()
        # print(self.state.players)
    # Functions are customized by client
    def get_state(self):
        # Building the map
        #print(self.state.x,self.state.y)
        view = np.zeros((5*(self.state.mapInfo.max_x + 1), 5*(self.state.mapInfo.max_y + 1), 6), dtype=int)
        #view[0:3, :] = -10
        #view[-3:, :] = -10
        #view[:, 0:3] = -10
        #view[:, -3:] = -10
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree     # trap map
                    view[5*i:5*i+5, 5*j:5*j+5,0] = -TreeID
                if self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap     # trap map
                    view[5*i:5*i+5, 5*j:5*j+5,0] = -TrapID
                if self.state.mapInfo.get_obstacle(i, j) == SwampID: # Swamp    # trap map
                    view[5*i:5*i+5, 5*j:5*j+5,0] = -SwampID
                if self.state.mapInfo.gold_amount(i, j) > 0:
                    view[5*i:5*i+5, 5*j:5*j+5,0] = self.state.mapInfo.gold_amount(i, j)/1000  ## gold map
        
        for stt,player in enumerate(self.state.players):
            if player["playerId"] != self.state.id:
                try:
                    if player["status"] not in [1,2,3]:
                        try:
                            view[5*player["posx"]:5*player["posx"]+5,5*player["posy"]:5*player["posy"]+5,stt + 1] = player["energy"]/50
                        except:
                            view[5*player["posx"]:5*player["posx"]+5,5*player["posy"]:5*player["posy"]+5,stt + 1] = 1
                except:
                    view[5*player["posx"]: 5*player["posx"]+5,5*player["posy"]:5*player["posy"]+5,stt]= 1
                    # print(self.state.players)
            else:
                try:
                    view[5*self.state.x:5*self.state.x+5,5*self.state.y:5*self.state.y+5,2]= self.state.energy
                except: 
                    print('out of map')
                
        DQNState = np.array(view)
        return DQNState

    def get_reward(self,action):
        # Calculate reward
        reward = 0
        score_action = self.state.score - self.score_pre
        self.score_pre = self.state.score
        
        pre_x, pre_y =self.pre_x,self.pre_y
        
        if self.state.energy >=45 and self.state.lastAction == 4:
            reward += -0.2
        #plus a small bonus if the agent go to a coordinate that has golds 
        if self.state.mapInfo.gold_amount(self.state.x,self.state.y) >= 50:
            reward += 0.2
        #If the DQN agent crafts golds, then it should obtain a positive reward (equal score_action)
        if score_action > 0:
            reward += score_action/50
        # if still in the map, plus a small bonus
        if self.state.status == State.STATUS_PLAYING:
            reward += 0.1
        # if there is no gold, but the agent still crafts golds, it will be punished
        if self.state.mapInfo.get_obstacle(pre_x,pre_y)<4 and int(self.state.lastAction)==5:
            reward+=-0.2
        if (self.state.mapInfo.gold_amount(pre_x,pre_y) >= 50 and self.pre_energy >15) and (int(self.state.lastAction)!=5):
            reward+=-0.2


        


        # If out of the map, then the DQN agent should be punished by a larger nagative reward.
        #if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
        #    reward = -1
            
        #Run out of energy, then the DQN agent should be punished by a larger nagative reward.
        #if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
        #    reward = -1
        # print ("reward",reward)
        #if self.state.status == State.STATUS_STOP_END_STEP:
        #    reward = +2
        return reward

    def check_terminate(self):
        #Checking the status of the game
        #it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING
    




def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)
