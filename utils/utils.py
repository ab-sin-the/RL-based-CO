import numpy as np
import random
from tqdm import tqdm
import math
import time
# random.seed(1)

global connection_info

def generate_optimal(n):
	optimal = [0, 3, 7, 8, 14, 18, 25, 26, 28, 30]
	state = np.zeros(n)
	for i in range(n):
		if i in optimal:
			state[i] = 1
		else:
			state[i] = 0
	return state

def reinforcement_learning(alpha,beta,gamma,theta,graph,batch_size):
	# alpha, beta, gamma are hyperparameters
	# graph stores the graph information with numpy matrix
	# pmat1~3 stores probability info
	# Initialize the probability matrix
	n = graph.shape[0] # node number
	t = 0
	max_iteration = 500
	temp_best_cost = 99999999
	temp_best_state = None
	pmat1 = np.zeros([1,n])
	pmat2 = np.zeros([n,n])
	pmat3 = np.zeros([n,n])
	for i in range(n):
		pmat1[0,i] = 0.5
		for j in range(n):
			if (i != j):
				pmat2[i,j] = 0.5
				pmat3[i,j] = 0.5
	global connection_info
	connection_info = dict()
	# graph is a sparse matrix, so first represent it in a better way
	for i in range(n):
		temp = set()
		for j in range(n):
			if graph[i,j] == 1:
				temp.add(j)
		connection_info[i] = temp

	# Generate the First State
	state = generate_state(pmat1,pmat2,pmat3)
	for t in tqdm(range(max_iteration)):
	 	if (random.random()<theta):
	 		state = generate_state(pmat1,pmat2,pmat3)

	 	else:
	 		state = generate_random_state(n)

	 	old_state_1 = state.copy()
	 	old_state = state.copy()
	 	state = local_search(graph,np.array(state),batch_size)
	 	while(sum(abs(np.array(old_state) - np.array(state))) != 0):
	 		old_state = state.copy()
	 		state = local_search(graph,np.array(state),batch_size)
 			update_function(pmat1,pmat2,pmat3,old_state_1,state,calculate_conflict(state, graph),t,alpha,beta)
 		temp_cost = cost_function(state,graph)
 		if(temp_cost < temp_best_cost):
 			temp_best_cost = temp_cost
 			temp_best_state = state

	temp_best_state = local_search(graph,temp_best_state,len(state))
	temp_best_cost = cost_function(temp_best_state,graph)
	print(temp_best_cost)
	print(sum(temp_best_state))

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


def update_function(matrix1, matrix2, matrix3, old_state, state, old_conflicts, t, alpha, beta):
	#update three state matrix
	#t is the iteration number
	#alpha is the hyperparameter of matrix2 and matrix 3
	#beta is the hyperparameter of matrix1
	t = t + 2.7
	difference = state - old_state
	difference = np.minimum(0, difference)
	state_01 = state.copy()
	if sum(difference) != 0:
		matrix1 -= alpha * difference  / (sum(difference))
	matrix1 += alpha * state / sum(state)

	if sum(difference) != 0:
		matrix2 -= alpha * np.outer(state_01,difference) / sum(difference)
	matrix2 += alpha * np.outer(state_01, state_01) / sum(state_01)

	state_10 = 1 - state_01
	matrix3 += alpha * np.outer(state_10, state_01) / sum(state_01)
	matrix3 -= alpha * np.outer(state_10, state_10) / sum(state_10)

	matrix1 = np.maximum(0, matrix1)
	matrix1 = np.minimum(1, matrix1)
	matrix2 = np.maximum(0, matrix2)
	matrix2 = np.minimum(1, matrix2)
	matrix3 = np.maximum(0, matrix3)
	matrix3 = np.minimum(1, matrix3)

def update_function_old(matrix1, matrix2, matrix3, old_state, state, old_conflicts, t, alpha, beta):
	#update three state matrix
	#t is the iteration number
	#alpha is the hyperparameter of matrix2 and matrix 3
	#beta is the hyperparameter of matrix1
	t = t + 2.7
	difference = state - old_state
	#print(difference)
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
	matrix2 = np.maximum(0, matrix2)
	matrix3 = np.maximum(0, matrix3)

	beta = beta / math.log(t)
	conflicts = conflicts * beta
	old_matrix = matrix1.copy()
	matrix1 = np.multiply(matrix1, 1 - conflicts)
	matrix1 += np.multiply(state_01, conflicts)
	return [matrix1, matrix2, matrix3]

def calculate_conflict(state,graph):
	# state is a list of 0s and 1s
	n = graph.shape[0]
	state_set = set()
	global connection_info 
	conflict_number = 0
	conflict_info = np.zeros([1,n])	
	# set to contain all chosen nodes in independent set
	for i in range(len(state)):
		if (state[i] == 1):
			state_set.add(i)
	k = len(state_set)
	# count conflict number
	for i in state_set:
		for node in connection_info[i]:
			if node in state_set:
				conflict_info[0,i] += 1
	reward = (conflict_number+1)/k
	return conflict_info


def cost_function(state,graph):
	# state is a list of 0s and 1s
	n = graph.shape[0]
	state_set = set()
	global connection_info
	conflict_number = 0
	# set to contain all chosen nodes in independent set
	for i in range(len(state)):
		if (state[i] == 1):
			state_set.add(i)

	k = len(state_set)
	k = k + 1

	# count conflict number
	for i in state_set:
		for node in connection_info[i]:
			if node in state_set:
				conflict_number += 1
	reward = 1 / k + 0.1 * conflict_number 
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
	#generate state by the three probability matrix
	allstate=[]
	choice = []
	n=len(pmat1[0])
	for i in range(n):
		allstate.append(-1)
	pma1_list=pmat1[0].tolist()
	#first choose highest prob node 
	first_index=pma1_list.index(max(pma1_list))	 
	chosen =[]
	if flipcoin(max(pma1_list)):
		chosen.append(first_index)
		allstate[first_index]=1
	else:
		allstate[first_index]=0	
	nodelist=[i for i in range(n)]
	nodelist.remove(first_index)
	#choose the other n-1 nodes	
	for i in range(n-1):
		prob=0.0
		newnode=nodelist[random.randint(0, len(nodelist)-1)]
		nodelist.remove(newnode)
		# newnode=random.randint(0, n-1)
		# while allstate[newnode]!=-1:
		# 	newnode=random.randint(0, n-1)

		#determine node by chosen nodes' prob in matrix 2,3 
		for j in range(n):
			if (allstate[j]==1):
				prob+=pmat2[j,newnode]
			if (allstate[j]==0):
				prob+=pmat3[j,newnode]
		prob/=(i+1)		
		if random.random() <= prob:
			chosen.append(newnode)
			allstate[newnode]=1
		else:
			allstate[newnode]=0
	return allstate

