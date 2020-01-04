import gym
from Logger import writer
from DQN import DQN_discrete
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    agent = DQN_discrete(env.observation_space.shape[0], env.action_space.n)
    for episode in range(5000):
        tot_reward = 0
        state = env.reset()
        while True:
            action = agent.act(state)
            new_state, reward, done, _ = env.step(action)

            agent.append((state, action, reward, new_state, done))
            agent.train()

            state = new_state
            tot_reward += reward

            # env.render()
            plt.show()
            if done:
                break
        if episode % 50 == 0:
            print(episode, tot_reward)
        writer.add_scalar('reward', tot_reward, episode)

    env.close()
