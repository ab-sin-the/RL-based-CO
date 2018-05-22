import numpy as np
import random
from tqdm import tqdm
import math

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


def local_search(graph,state,batch_size):
    #local search function
	#run local search on batch size elements
	choice = []
	n = state.shape[0]
	for i in range(n):
		choice.append(random.randint(0, n - 1))

	old_state = state.copy()
	for i in range(len(choice)):
		state[choice[i]] = 1 - state[choice[i]]
		if (cost_function(state,k) < cost_function(old_state,k)):
			old_state[choice[i]] = 1 - old_state[choice[i]]
		else:
			state[choice[i]] = 1- state[choice[i]]
	return old_state


def update_function(matrix1, matrix2, matrix3, state, old_conflicts, t, alpha, beta):
	#update three state matrix
	#t is the iteration number
	#alpha is the hyperparameter of matrix2 and matrix 3
	#beta is the hyperparameter of matrix1
	state_01 = state.copy()
	state_11 = state.copy()
	conflicts = old_conflicts.copy()
	state_11 = state_11 * 2 - 1

	conflicts = conflicts + 1
	conflicts = 1 / conflicts

	expanded_state_11 = np.tile(state_11, (np.shape(state_11)[0], 1))

	update_matrix2 = np.outer(state_01, conflicts)
	update_matrix3 = np.outer((1 - state_01), conflicts)

	update_matrix2 = np.multiply(update_matrix2, expanded_state_11)
	update_matrix3 = np.multiply(update_matrix3, expanded_state_11)

	matrix2 += (alpha / math.log(t) ) * update_matrix2
	matrix3 += (alpha / math.log(t) ) * update_matrix3

	conflicts = conflicts * beta

	matrix1 = np.multiply(matrix1, 1 - conflicts)
	matrix1 += np.multiply(state_01, conflicts)
	return [matrix1, matrix2, matrix3]



def cost_function(state,k):
	return

