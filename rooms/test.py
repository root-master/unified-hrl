import time
import gym
from gym_rooms.envs import *
environment = 'Rooms-v0'
env = gym.make(environment)

for i in range(10):
    s = env.reset()
    # env.render()
    for j in range(200):
        a = env.action_space.sample()
        sp,r,done,info = env.step(a)
        # env.render()
        if env.new_pos_before_passing_doorway in env.hallways:
            doorway = env.new_pos_before_passing_doorway
            print('-'*30)
            print('s = ', s)
            print('doorway = ', doorway)
            print('sp =', sp)
        s = sp
