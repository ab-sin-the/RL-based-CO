import numpy as np
import random
from tqdm import tqdm
import math

# random.seed(1)
def reinforcement_learning(alpha,beta,gamma,theta,graph,batch_size):
	# alpha, beta, gamma are hyperparameters
	# graph stores the graph information with numpy matrix
	# pmat1~3 stores probability info
	# Initialize the probability matrix
	n = graph.shape[0] # node number
	t = 0
	max_iteration = 1000
	temp_best_cost = 99999999
	temp_best_state = None
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
	state = generate_state(pmat1,pmat2,pmat3)


	while(t < max_iteration):
	 	if (t != 0):
	 		if(random.random()<gamma):
	 			if (random.random()<theta):
	 				state = generate_state(pmat1,pmat2,pmat3)
	 			else:
	 				state = generate_random_state(n)
	 		else:
	 			state = local_search(graph,state,batch_size)
 		state_probability = [abs(pmat1[i]-0.5) for i in state if state[i]==1]
 		inverted_state = state[:]
 		indx = state_probability.index(min(state_probability))
 		inverted_state[indx]=1 - inverted_state[indx]
 		sub_optimal_state = local_search(graph,inverted_state,batch_size)
 		temp_cost = cost_function(sub_optimal_state,graph)
 		if(temp_cost < temp_best_cost):
 			temp_best_cost = temp_cost
 			temp_best_state = sub_optimal_state
 		pmat1,pmat2,pmat3 = update_function(pmat1,pmat2,pmat3,sub_optimal_state,calculate_conflict(sub_optimal_state, graph),t,alpha,beta)
	 	t += 1
	print(temp_best_cost)
	print(temp_best_state)

def local_search(graph,state,batch_size):
    #local search function
	#run local search on batch size elements
	choice = []
	n = state.shape[0]
	for i in range(batch_size):
		choice.append(random.randint(0, n - 1))

	old_state = state.copy()
	for i in range(len(choice)):
		state[choice[i]] = 1 - state[choice[i]]
		if (cost_function(state,graph) < cost_function(old_state,graph)):
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

	conflicts = conflicts * beta / math.log(t)

	matrix1 = np.multiply(matrix1, 1 - conflicts)
	matrix1 += np.multiply(state_01, conflicts)
	return [matrix1, matrix2, matrix3]

def calculate_conflict(state,graph):
	# state is a list of 0s and 1s
	n = graph.shape[0]
	state_set = set()
	connection_info = dict()
	conflict_number = 0
	conflict_info = np.zeros([1,n])	
	# set to contain all chosen nodes in independent set
	for i in range(state):
		if (state[i] == 1):
			state_set.add(i)
	k = len(state_set)
	# graph is a sparse matrix, so first represent it in a better way
	for i in range(n):
		temp = set()
		for j in range(n):
			if graph[i,j] == 1:
				temp.add(j)
		connection_info[i] = temp

	# count conflict number
	for i in state_set:
		for node in connection_info[i]:
			if node in state_set:
				conflict_info[i] += 1
	reward = (conflict_number+1)/k
	return conflict_info
def cost_function(state,graph):
	# state is a list of 0s and 1s
	n = graph.shape[0]
	state_set = set()
	connection_info = dict()
	conflict_number = 0
	# set to contain all chosen nodes in independent set
	for i in range(state):
		if (state[i] == 1):
			state_set.add(i)
	k = len(state_set)
	# graph is a sparse matrix, so first represent it in a better way
	for i in range(n):
		temp = set()
		for j in range(n):
			if graph[i,j] == 1:
				temp.add(j)
		connection_info[i] = temp

	# count conflict number
	for i in state_set:
		for node in connection_info[i]:
			if node in state_set:
				conflict_number += 1
	reward = (conflict_number+1)/k
	return reward
	
def flipcoin(p):
	r=random.random()
	return r<p

def generate_random_state(n):
	state=[]
	for i in range(n):
		state.append(random.randint(0,1))
	return np.array(state)

def generate_state(pmat1,pmat2,pmat3):
	allstate=[]
	n=len(pmat1[0])
	for i in range(n):
		allstate.append(-1)
	pma1_list=pmat1[0].tolist()
	first_index=pma1_list.index(max(pma1_list))	
	chosen =[]
	if flipcoin(max(pma1_list)):
		chosen.append(first_index)
		allstate[first_index]=1
	else:
		allstate[first_index]=0	
	for i in range(n-1):
		prob=0.0
		newnode=random.randint(0, n-1)
		while allstate[newnode]!=-1:
			newnode=random.randint(0, n-1)
		for j in range(n):
			if (allstate[j]==1):
				prob+=pmat2[i,j]
			if (allstate[j]==0):
				prob+=pmat3[i,j]
		prob/=(i+1)		
		if random.random() <= prob:
			chosen.append(newnode)
			allstate[newnode]=1
		else:
			allstate[newnode]=0
	return allstate