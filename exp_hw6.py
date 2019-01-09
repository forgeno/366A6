#!/usr/bin/env python

import numpy as np
from agent_hw6 import Agent

from rl_glue import RLGlue
from env_hw6 import Environment
from plot import plotGraph

def question_1():
    # Specify hyper-parameters

    agent = Agent()
    environment = Environment()
    rlglue = RLGlue(environment, agent)

    num_episodes = 200
    num_runs = 50
    max_eps_steps = 100000

    steps = np.zeros([num_runs, num_episodes])

    for r in range(num_runs):
        print("run number : ", r)
        rlglue.rl_init()
        for e in range(num_episodes):
            #print("Episode number: "+str(e))
            rlglue.rl_episode(max_eps_steps)
            steps[r, e] = rlglue.num_ep_steps()
            #print("Number of steps: "+str(steps))
            # print(steps[r, e])
    np.save('steps', steps)
    plotGraph()
    
    del agent, environment, rlglue
    agent = Agent()
    environment = Environment()
    rlglue = RLGlue(environment, agent)

    num_episodes = 1000
    num_runs = 1
    max_eps_steps = 100000

    steps = np.zeros([num_runs, num_episodes])

    for r in range(num_runs):
        print("run number : ", r)
        rlglue.rl_init()
        for e in range(num_episodes):
            print("Episode number: "+str(e))
            rlglue.rl_episode(max_eps_steps)
            steps[r, e] = rlglue.num_ep_steps()
            #print("Number of steps: "+str(steps))
            # print(steps[r, e])
    #np.save('steps', steps)
    #plotGraph()
    rlglue.rl_agent_message("plot3DGraph")


if __name__ == "__main__":
    question_1()
    print("Done")
