import sys
sys.path.append('../utils')
import utils
import numpy as np

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
	(n, E) = read('../data/g4.in')
	verticles = [i for i in range(n)]  
	print n,E
	print verticles
	return


if __name__ == '__main__':

	main()