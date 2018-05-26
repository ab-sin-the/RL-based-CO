import sys
sys.path.append('../utils')
import utils
import numpy as np
import random
import math
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
    file_path = '../data/g30.in'
    (n, E) = read(file_path)
    E = np.array(E)
    verticles = [i for i in range(n)]
    alpha = 0.005
    beta = 0.005
    gamma = 0.01
    theta = 0.8
    batch_size = math.ceil(n / 10)
    batch_size = n - 1
    print(E)
    utils.reinforcement_learning(alpha,beta,gamma,theta,E,batch_size)
    return


if __name__ == '__main__':
	main()