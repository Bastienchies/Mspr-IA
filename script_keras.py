from collections import deque
import marlo
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.optimizers import RMSprop, Adam
import numpy as np
from enum import Enum
import QLearn
from QAgent import QAgent
print("testo")
episodes = 1000
client_pool = [('127.0.0.1', 10000)]

join_tokens = marlo.make('MarLo-CliffWalking-v0',
                         params={
                             "client_pool": client_pool
                         })
# As this is a single agent scenario,
# there will just be a single token
assert len(join_tokens) == 1
join_token = join_tokens[0]
env = marlo.init(join_token)
if __name__ == "__main__":
    env = marlo.init(join_token)
    trials = 1000
    trial_len = 200

    updateTargetNework = 100
    DQN_agent = QLearn.DQN(env=env)
    steps = []
    print('nombre d\'essais: ', trials)
    done = False

    for trial in range(trials):
        while not done:
            action = env.action_space.sample()
            obs, r, done, info = env.step(action)
            cur_state = obs
            for step in range(trial_len):
                action = DQN_agent.act(cur_state)
                env.render()
                new_state, reward, done, _ = action
                reward = reward if not done else -20
                print('récompense: ', reward)
                print('step: ', step)
                # new_state = new_state
                DQN_agent.remember(cur_state, action,
                                   reward, new_state, done)

                DQN_agent.replay()
                DQN_agent.target_train()
                cur_state = new_state
                if done:
                    break
            if step >= 199:
                print("Echec")
            else:
                print("Terminé en {} essais".format(trial))
