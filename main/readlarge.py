import sys
sys.path.append('../utils')
import utils
import numpy as np
import random
import math
# from tqdm import tqdm
# To read input
def readlargetofile(file):
    with open(file, 'r') as input: #up to now you should add n in the first line by hand =-=
        line1 = input.readline()
        # print line1
        line1sp=line1.split()
        # n = next(input)
        n=int(line1sp[2])
        E = np.zeros(shape=(n,n), dtype=np.int)
        i = 0
        i=0
        for line in input:
            linesp=line.split()

            x=int(linesp[1])-1
            y=int(linesp[2])-1
            # if i<10:
            #     i+=1
            #     print linesp
            #     print x,y
            E[x][y] = 1
            E[y][x] = 1
    filenamew = '../data/g450.in'
    # with open(filenamew,'w') as f: #
    #     f.write(str(n)) 
    #     f.write("\n") 
    #     f.write(E)   
    np.savetxt(filenamew, E,fmt="%d") #write the numpy
    # with open('filenamew', 'r+') as f:
    #     content = f.read()        
    #     f.seek(0, 0)
    #     f.write(str(n)+content)
    return (n, E)
def main():
    # Main
    file_path = '../data/g450.mis'
    (n, E) = readlargetofile(file_path)
    E = np.array(E)
    verticles = [i for i in range(n)]
    # print n
    # print E
    # utils.reinforcement_learning(alpha,beta,gamma,theta,E,batch_size)
    return


if __name__ == '__main__':
    main()