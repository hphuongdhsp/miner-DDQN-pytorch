import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.history_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.next_history_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, history, action, reward, next_history, done):
        index = self.mem_cntr % self.mem_size
        self.history_memory[index] = history
        self.next_history_memory[index] = next_history
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        histories = self.history_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_histories = self.next_history_memory[batch]
        terminal = self.terminal_memory[batch]

        return histories, actions, rewards, next_histories, terminal