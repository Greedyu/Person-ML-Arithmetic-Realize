
import numpy as np

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readline()
    numberOFLines = len(arrayOLines)

    returnMat = np.zeros((numberOFLines,3))
    classLabeVector = []
    index = 0
