import os
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

filename = 'plotValues.txt'

def plotGraph3D():
    x = np.arange(-1.2, 0.5, 1.7/50) #Positon
    y = np.arange(-0.07, 0.07, 0.14/50) #Velocity
    zCoord = np.array([[0.0]*50]*50)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if os.path.exists(filename):
        #data = open(filename,"r")
        with open(filename) as data:
            counter = 0
        
            for line in data:
                listNums = [float(value) for value in line.split()]
                for index in range(50):
                    zCoord[counter][index] = listNums[index]
                counter += 1
        X, Y= np.meshgrid(x, y)
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.plot_surface(X, Y, zCoord, cstride=1,rstride=1)
        plt.show()

if __name__ == "__main__": 
    plotGraph3D()
        