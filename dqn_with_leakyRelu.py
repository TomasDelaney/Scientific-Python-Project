import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.optimizers
from keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
import matplotlib.pyplot as plt
import time


class ReplayBuffer(object):  # agents memory
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size  # max size of memory
        self.input_shape = input_shape  # the input of the environment
        self.discrete = discrete  # the action space discrete or continous
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = action
        else:
            self.action_memory[index] = action
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)  # so you do not read over the end of the array
        batch = np.random.choice(max_mem, batch_size)  # get a random -> we do not select the same over and over again

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = keras.Sequential()
    model.add(keras.layers.Dense(fc1_dims, input_shape=(input_dims,)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(keras.layers.Dense(fc2_dims))
    model.add(LeakyReLU(alpha=0.1))
    model.add(keras.layers.Dense(n_actions, activation=None))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')

    return model


class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dims, file_name, epsilon_dec=0.996,
                 epsilon_end=0.01, mem_size=1000000):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = file_name

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)  # Lunar lander discrete action space

        self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def chose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()

        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):  # do not learn if not filled up a batch size memory
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        # (64,) alaku matrixok lesznek mivel batch=64 tehat minden kepre egy action-> egy reward

        q_eval = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, action[:, 0]] = reward + self.gamma * np.max(q_next, axis=1) * done

        _ = self.q_eval.fit(state, q_target, verbose=0)

        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.eps_min else self.eps_min

    def evaluate(self, state):
        state = state[np.newaxis, :]
        actions = self.q_eval.predict(state)
        action = np.argmax(actions)

        return action

    def plot_training_data(self, training_scores, length_of_training_episodes, n_games, start_time):
        x = [i + 1 for i in range(n_games)]
        plt.figure(1)
        plt.plot(x, training_scores, label="Training accuracy")
        plt.title('Scores of the agent in each episode')
        plt.xlabel('Number of episodes')
        plt.ylabel('Achieved score')
        plt.legend()
        plt.show()

        # plot the length of the episodes
        plt.figure(2)
        plt.plot(x, length_of_training_episodes, label="Length of training episode")
        # plt.plot(x, length_of_test_episodes, label="Length of testing episode")
        plt.title('Episode lengths of the agent')
        plt.xlabel('Number of episodes')
        plt.ylabel('Length of the episodes')
        plt.show()

        print('\n', 'The highest received training award of the agent: {}'.format(np.max(training_scores)))
        print('\n', 'The average episode length of training the agent: {}'.format(np.mean(length_of_training_episodes)))
        print('\n', "--- %s seconds ---" % (time.time() - start_time))

    def save_model(self):
        self.q_eval.save(self.model_file)
        print('...Model saved...')

    def load_model(self):  # no function but in main file to specify the file name
        self.q_eval = load_model(self.model_file)
        print('...Model loaded...')
