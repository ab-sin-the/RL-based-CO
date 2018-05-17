import numpy as np
import random

# random.seed(1)
def reinforcement_learning(alpha,beta,gamma,graph,pmat1,pmat2,pmat3):
	# alpha, beta, gamma are hyperparameters
	# graph stores the graph information with numpy matrix
	# pmat1~3 stores probability info
	# Initialize the probability matrix
	n = graph.shape[0] # node number
	pmat1 = np.zeros([1,n])
	pmat2 = np.zeros([n,n])
	pmat3 = np.zeros([n,n])
	for i in range(n):
		pmat1[1,i] = 0.5
		for j in range(n):
			if (i != j):
				pmat2[i,j] = 0.5
				pmat3[i,j] = 0.5
	# Generate the First State
	state =[]
	for i in range(n):
		if random.random() <= pmat1[i]:
			state.append(i) 
def local_search(graph,state):
def cost_function(state,k):