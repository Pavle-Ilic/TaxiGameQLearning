import gym
import random
import numpy as np

# returns observation/state, probability, actions available
env = gym.make("Taxi-v3", render_mode="ascii")
env.reset()
#env.render()


qRows = env.observation_space.n
qColumns = env.action_space.n

qTable = np.zeros((qRows, qColumns))
print(qTable)
print(f"States = {qRows}, Actions = {qColumns}")

# learning rate(How easily should agent accept new info)
learnRate = 0.9
# discount factor(how much should long term rewards be considered over short term)
discRate = 0.8
# prob agent explores
epsi = 0.85
# decay rate(decaying the exploration)
decRate = 0.01

numEpisodes = 1000
numSteps = 100

# Q-Learning Algorithm
for episode in range(numEpisodes):
    truncated = False
    terminated = False
    # print(env.reset())
    state = env.reset()[0]

    for step in range(numSteps):
        if random.uniform(0, 1) < epsi:
            action = env.action_space.sample()
        else:
            action = np.argmax(qTable[state, :])

        # print(env.step(action))
        newState, reward, truncated, terminated, _ = env.step(action)

        qTable[state, action] = qTable[state, action] + learnRate * \
                                (reward + discRate * np.max(qTable[newState, :]) - qTable[state, action])

        print(f"Step: {step} of episode {episode}")
        state = newState
        # print(type(state))

        if truncated or terminated:
            break

    epsi = np.exp(-decRate * episode)
    print(f"Epsilon: {epsi}")


truncated = False
terminated = False
score = 0
step = 0
state = env.reset()[0]

game = input("Do you want to watch the  AI play: ")
if game.lower() == "yes":
    #Actually Playing
    while not(truncated or terminated):
        print(f"Step {step}")
        action = np.argmax(qTable[state, :])
        newState, reward, truncated, terminated, _ = env.step(action)
        score += reward
        env.render()
        print(f"Score = {score}")
        state = newState
        step += 1

    env.close()

#%%

#%%
