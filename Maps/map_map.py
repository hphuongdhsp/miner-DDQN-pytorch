
import numpy as np 
import matplotlib.pyplot as plt
map1 = [[450,-2,0,-2,150,-1,0,0,0,0,-1,-2,-2,-2,0,0,0,0,150,-2,350],
        [-2,-2,-2,-2,-1,0,-1,-1,-1,-1,-3,50,-2,-2,-2,-2,-3,-3,50,-2,-1],
        [-2,-2,200,-2,0,-2,0,-2,-3,-3,-2,0,-3,-2,-2,150,-3,-3,0,0,50],
    [0,-3,-3,-2,0,0,-1,0,550,-3,-2,0,0,0,-1,0,0,-1,-1,-1,-2],
    [-2,0,0,0,-1,0,-1,50,300,-3,-2,0,-3,0,0,0,-1,-3,-3,-2,-1],
    [-1,-3,-1,-3,0,-2,0,0,-2,-1,100,-3,0,-2,300,-3,0,-2,-3,-2,0],
    [-2,-3,-1,-3,-1,500,-1,-3,-2,-1,0,-1,0,-1,0,-1,0,-2,-3,-3,-1],
    [0,-3,-1,-3,0,-2,-3,-3,0,0,0,0,-2,0,-2,-3,-3,-3,-3,200,-1],
    [1200,-3,-1,-3,-1,-1,-2,-2,0,-1,150,-2,0,-2,0,0,-2,-3,-3,1500,50]]
Map = np.array(map1)
TreeID = 1
TrapID = 2
SwampID = 3
max_x = Map.shape[1]-1
max_y = Map.shape[0]-1
print(max_x)
view = np.zeros([(max_y+1)*5, (max_x+1)*5], dtype=int)
for i in range(max_y):
    for j in range(max_x):
        if Map[i, j] == -TreeID:  # Tree
            view[i*5:i*5+5, j*5:j*5+5] = -TreeID
        if Map[i, j] == -TrapID:  # Trap
            view[i*5:i*5+5, j*5:j*5+5] = -TrapID
        if Map[i, j] == -SwampID: # Swamp
            view[i*5:i*5+5, j*5:j*5+5] = -SwampID
        if Map[i, j] > 0:
            view[i*5:i*5+5, j*5:j*5+5] = Map[i, j]
plt.imshow(view)
p_x =5
p_y =5
view[p_x*5+1:p_x*5+4, p_y*5+1:p_y*5+4] =400

def get_state(self):
        # Building the map
        view = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree
                    view[i, j] = -TreeID
                if self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap
                    view[i, j] = -TrapID
                if self.state.mapInfo.get_obstacle(i, j) == SwampID: # Swamp
                    view[i, j] = -SwampID
                if self.state.mapInfo.gold_amount(i, j) > 0:
                    view[i, j] = self.state.mapInfo.gold_amount(i, j)
        DQNState = []
        DQN_array = []
        DQNState.append(view)
        DQN_array.append(self.state.x)
        DQN_array.append(self.state.y)
        DQN_array.append(self.state.energy)
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                DQN_array.append(player["posx"])
                DQN_array.append(player["posy"])
        DQN_array = np.array(DQN_array)
        DQNState.append(DQN_array)