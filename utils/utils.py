import numpy as np
import random
from tqdm import tqdm

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
	for i in range(n):u
		choice.append(random.randint(0, n - 1))

	old_state = state.copy()
	for i in range(len(choice)):
		state[choice[i]] = 1 - state[choice[i]]
		if (cost_function(state,k) < cost_function(old_state,k)):
			old_state[choice[i]] = 1 - old_state[choice[i]]
		else:
			state[choice[i]] = 1- state[choice[i]]
	return old_state


def reward_function(state,graph):
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
				conflict_number += 1
				conflict_info[i] += 1
	reward = (conflict_number+1)/k
	return reward

def generate_state(pmat1,pmat2,pmat3):
	allstate=[]
	n=len(pmat1[0])
	for i in range(n):
		allstate.append(-1)

	chosen =[]
	if flipcoin(max(pma1_list)):
		chosen.append(first_index)
		allstate[first_index]=1
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