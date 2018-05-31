import numpy as np
import random
from tqdm import tqdm
import math
import time
# random.seed(1)

global connection_info
def reinforcement_learning(alpha,beta,gamma,theta,graph,batch_size):
	# alpha, beta, gamma are hyperparameters
	# graph stores the graph information with numpy matrix
	# pmat1~3 stores probability info
	# Initialize the probability matrix
	n = graph.shape[0] # node number
	t = 0
	max_iteration = 10000
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
	 	#if (t != 0):
	 		#if(random.random()<gamma):
	 	#		if (random.random()<theta):
	 	#			state = generate_state(pmat1,pmat2,pmat3)
	 	#		else:
	 	#			state = generate_random_state(n)
	 	#	else:
	 	#		state = local_search(graph,np.array(state),batch_size)
	 	if (random.random()<theta):
	 		state = generate_state(pmat1,pmat2,pmat3)
	 	else:
	 		state = generate_random_state(n)
	 	old_state1 = generate_state(pmat1,pmat2,pmat3)
	 	old_state = state[:]
	 	state = local_search(graph,np.array(state),batch_size)
	 	while(sum(abs(np.array(old_state) - np.array(state))) != 0):
	 		old_state = state.copy()
	 		state = local_search(graph,np.array(state),batch_size)
 			#pmat1,pmat2,pmat3 = update_function(pmat1,pmat2,pmat3,state,calculate_conflict(state, graph),t,alpha,beta)
 		pmat1,pmat2,pmat3 = update_function(pmat1,pmat2,pmat3,np.array(state),calculate_conflict(state, graph),t,alpha,beta)
 		#state_probability = [abs(pmat1[0,i]-0.5) for i in range(len(state)) if state[i]==1]
 		#inverted_state = state[:]
 		#if state_probability == []:
 	#		indx = 0
 	#	else:
 	#		indx = state_probability.index(min(state_probability))
 	#	inverted_state[indx]=1 - inverted_state[indx]
 	#	sub_optimal_state = local_search(graph,np.array(inverted_state),batch_size)
 		temp_cost = cost_function(state,graph)
 		if(temp_cost < temp_best_cost):
 			temp_best_cost = temp_cost
 			temp_best_state = state
 		if ((t + 1)% 400 == 0):
 			new_state = generate_state(pmat1,pmat2,pmat3)
 			print(sum(abs(np.array(new_state) - np.array(old_state1))))
 			print(temp_best_cost)
 			print(sum(temp_best_state))
	
	temp_best_state = local_search(graph,temp_best_state,len(state))
	print(temp_best_state)
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

def update_function(matrix1, matrix2, matrix3, state, old_conflicts, t, alpha, beta):
	#update three state matrix
	#t is the iteration number
	#alpha is the hyperparameter of matrix2 and matrix 3
	#beta is the hyperparameter of matrix1
	t = t + 2.7
	state_01 = state.copy()
	state_11 = state.copy()
	conflicts = old_conflicts.copy()
	state_11 = state_11 * 2 - 1
	conflicts = conflicts + 1
	conflicts = 1 / conflicts
	expanded_state_11 = np.tile(state_11, (np.shape(state_11)[0], 1))

	update_matrix2 = np.outer(state_01, conflicts)
	update_matrix3 = np.outer((1 - state_01), conflicts)
	if (math.floor(t) % 400 == 0):
		print(alpha / math.log(t))
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
	reward = ((conflict_number+1))/k
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

