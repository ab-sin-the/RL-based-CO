import sys
sys.path.append('../utils')
import utils
import numpy as np
import random
from tqdm import tqdm
# To read input
def read(file):
    with open(file, 'r') as input:
        n = int(next(input))
        E = np.zeros(shape=(n,n), dtype=np.int)
        i = 0
        for line in input:
            j = 0
            for x in line.split():
                E[i][j] = int(x)
                j = j + 1
            i = i + 1
    return (n, E)


def main():
    # Main
    file_path = '../data/g4.in'
    (n, E) = read(file_path)
    verticles = [i for i in range(n)]
    pmat1 = np.zeros([1,n])  #for test
    pmat2 = np.zeros([n,n])
    pmat3 = np.zeros([n,n])
    for i in range(n):
        pmat1[0,i] = random.random()
        for j in range(n):
            if (i != j):
                pmat2[i,j] = 0.5
                pmat3[i,j] = 0.5
    print utils.generate_state(pmat1,pmat2,pmat3)
    return


if __name__ == '__main__':
	main()