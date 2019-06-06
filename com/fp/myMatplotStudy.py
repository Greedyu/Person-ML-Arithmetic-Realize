import matplotlib.pyplot
import numpy as np
from matplotlib import pyplot

myarray = np.array([[1,2,3],[3,4,5],[5,6,7]])
pyplot.plot(myarray)
pyplot.xlabel('x axis')
pyplot.ylabel('y axis')

pyplot.show()