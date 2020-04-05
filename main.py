import gym
from Logger import writer
from DQN_target import DQN

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    agent = DQN(
        env.observation_space.shape[0],
        env.action_space.n,
        writer=writer
    )

    for episode in range(100):
        tot_reward = 0
        state = env.reset()
        while True:
            action = agent.act(state)
            new_state, reward, done, _ = env.step(action)

            # let's use reward shaping
            modified_reward = reward + 300 * (0.99 * abs(new_state[1]) - abs(state[1]))
            agent.append((state, action, modified_reward, new_state, done))
            agent.train()

            state = new_state
            tot_reward += reward

            # env.render()
            if done:
                break
        writer.add_scalar('reward', tot_reward, episode)

    env.close()
