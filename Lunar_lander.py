import gym
import matplotlib.pyplot as plt
import numpy as np
import time
from dqn_with_leakyRelu import Agent

if __name__ == '__main__':
    start_time = time.time()
    env = gym.make('LunarLander-v2')
    n_games = 5
    training = False
    evaluate = True
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=8, n_actions=env.action_space.n, mem_size=100000,
                  batch_size=64, epsilon_end=0.01, file_name='dqn_model_with_leakyRelu_v5.h5')

    training_scores = []
    test_scores = []
    length_of_training_episodes = []
    length_of_test_episodes = []
    eps_history = []

    if training:
        for i in range(1, n_games+1):
            done = False
            score = 0
            episode_length = 0
            observation = env.reset()

            while not done:
                action = agent.chose_action(observation)
                observation_, reward, done, info = env.step(action)
                episode_length += 1
                score += reward
                agent.remember(observation, action, reward, observation_, done)
                observation = observation_
                agent.learn()

            # append to the lists which we plot + epsilon list
            eps_history.append(agent.epsilon)
            training_scores.append(score)
            length_of_training_episodes.append(episode_length)

            # give console feedback
            avg_score = np.mean(training_scores)
            print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)
            print('episode length: {}'.format(episode_length))

            # test the model accuracy with evaluate
            done = False
            score = 0
            episode_length = 0
            observation = env.reset()

            while not done:
                action = agent.evaluate(observation)
                observation_, reward, done, info = env.step(action)
                score += reward
                agent.remember(observation, action, reward, observation_, done)
                observation = observation_
                episode_length += 1

            test_scores.append(score)
            length_of_test_episodes.append(episode_length)

            # saving the model
            if i % 10 == 0:
                agent.save_model()
                print("Saved after {} iterations.".format(i))

        # plot the training accuracy and the "test" accuracy
        agent.plot_training_data(training_scores, length_of_training_episodes, n_games, start_time)

    if evaluate:
        agent.load_model()  # make sure the hyper params match in the initializing
        for i in range(n_games):
            done = False
            score = 0
            episode_length = 0
            observation = env.reset()

            while not done:
                env.render()
                action = agent.evaluate(observation)
                observation_, reward, done, info = env.step(action)
                score += reward
                agent.remember(observation, action, reward, observation_, done)
                observation = observation_
                episode_length += 1

            test_scores.append(score)
            length_of_test_episodes.append(episode_length)
            avg_score = np.mean(test_scores)
            print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)
            print('episode length: {}'.format(episode_length))

        # plotting the results
        x = [i + 1 for i in range(n_games)]
        plt.plot(x, test_scores, label='Score of the agent')
        plt.title("Evaluation of the agent")
        plt.xlabel('Episodes')
        plt.ylabel('Scores')
        plt.legend()
        plt.show()
