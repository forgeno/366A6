import numpy as np
from rl_glue import BaseAgent
import random
from tile3 import IHT, tiles
from plot3DGraph import plotGraph3D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
class Agent(BaseAgent): 
    """
    simple random agent, which moves left or right randomly in a 2D world

    Note: inheret from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""
        self.iht = IHT(2048)
        self.numTiling = 8
        self.stepSize = 0.1/self.numTiling 
        self.lambdaValue = 0.9
        self.epsilon = 0
        self.gamma =  1


    def agent_init(self):
        """Initialize agent variables. resets at every run"""
        self.weights = np.random.uniform(-0.001,0,2048)
        self.tdError = None
        self.traces = []
        self.currentAction = None
        self.previousAction = None
        self.currentTiles = np.zeros(0)
        self.previousTiles = np.zeros(0)
        self.currentActionValue = None
        self.previousActionValue = None  
        self.frogs = 69696969

    def getActiveTiles(self, state):
        tileVectors = []
        scaleX = 0.5 + 1.2
        scaleY = 0.07 + 0.07
        tile = self.getTileScale(state)
        for i in range(0,3):
            selectedTiles = tiles(self.iht, 8, tile,[i])
            tileVectors.append(selectedTiles)
        return tileVectors
        
    def getAction(self, state, tileVectors):
        tempQArr = []
        for choice in range (0,3):
            stateVector = np.zeros(2048)
            for element in tileVectors[choice]:
                stateVector[element] = 1
            qValue = np.dot(stateVector, self.weights)
            tempQArr.append(qValue)
        bestQIndex = np.argmax(tempQArr) #Maybe need tiebreaking?
        bestQValue = tempQArr[bestQIndex]
        return bestQIndex, bestQValue

    def agent_start(self, state):
        # Resets at every episode
        self.currentTiles = self.getActiveTiles(state)
        self.currentAction, self.currentActionValue = self.getAction(state, self.currentTiles)
        self.traces = np.zeros(2048)
        action = self.currentAction
        return action

    def agent_step(self, reward, state):    
        ##Take action A and observe R, S'##
        #print()
        self.previousAction = self.currentAction #setting S' -> S
        self.previousTiles = self.currentTiles #saves F(S,A) as previousTiles because we are about to take a new action
        self.previousActionValue = self.currentActionValue #sets prev to Q
        self.currentTiles = self.getActiveTiles(state)#saves F(S',A') for future calculations
        self.currentAction, self.currentActionValue = self.getAction(state, self.currentTiles) #Gets a new S and A
        action = self.currentAction #selecting new S' to replace it
        
        #print("Old stats: {} {} New stats: {} {}".format(self.previousAction,self.previousActionValue,self.currentAction,self.currentActionValue))
        #print("Current state: {}".format(state))
        ## calc tdError ##
        self.tdError = reward + self.gamma*self.currentActionValue - self.previousActionValue
        
        ## Loop for i in F(S,A) ##
        for element in self.previousTiles[self.previousAction]:
            self.tdError -= self.weights[element]
            self.traces[element] = 1
        ## Loop for i in F(S',A') ##
        for element in self.currentTiles[self.currentAction]:
            self.tdError += self.gamma*self.weights[element]
        
        self.weights += self.stepSize*self.tdError*self.traces
        self.traces = self.gamma*self.lambdaValue*self.traces

        return action

    def agent_end(self, reward):
        self.previousAction = self.currentAction #setting S' -> S
        self.previousTiles = self.currentTiles #saves F(S,A) as previousTiles because we are about to take a new action
        self.tdError = reward - self.previousActionValue
        for element in self.previousTiles[self.previousAction]:
            self.tdError -= self.weights[element]
            self.traces[element] = 1

        self.weights += self.stepSize*self.tdError*self.traces

    def getTileScale(self, state):
        return [8*state[0]/(0.5+1.2), 8*state[1]/(0.07+0.07)]

    def plot3DGraph(self):
        step_size = 50
        # scaleX = 0.5 + 1.2
        # scaleY = 0.07 + 0.07
        f = open('plotValues.txt','w')
        for i in range(step_size):
            pos = -1.2 + (i * 1.7/step_size)
            for j in range(step_size):
                vel = -0.07  + (j * 0.14/step_size)
                values = []
                for a in range(0,3):
                    tile = self.getTileScale([pos,vel])
                    stateVector = np.zeros(2048)
                    # [(-1.2+(i*1.7/step_size))*scaleX,(-0.07+(j*0.14/step_size))*scaleY]
                    inds = tiles(self.iht, 8, tile, [a])
                    for element in inds:
                        stateVector[element] = 1
                    values.append(np.dot(stateVector, self.weights))
                height = max(values)
                f.write(repr(-height)+" ")
            f.write("\n")
        f.close()
        plotGraph3D()
    def agent_message(self, message):
        
        if 'plot3DGraph' in message:
            self.plot3DGraph()
        
if __name__ == "__main__": 
    pass
        