#!/usr/bin/python

# Independent set algos

# E is represented by a adjacency matrix
# V is represented by an array

import csv
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

def lineValue(l, V, limit):
    c = 0
    i = 0
    for k in l:
        if (k):
            # We check if i is not a removed node
            if (i in V):
                c += 1
        if (c > limit):
            return -1
        i += 1
    return c

def removeNeighbors(E, V, u):
    for i in range(len(E[u])):
        if (E[u][i]):
            if (i in V):
                V.remove(i)
    V.remove(u)

def seekMaximumDegree(E, V):
    max = 0
    uMax = 0
    for i in V:
        c = 0
        n = 0
        for j in E[i]:
            # We check if i is not a removed node
            if (j and (n in V)):
                c = c + 1
            n += 1
        if (c > max):
            uMax = i
            max = c
    return uMax

def seekSmallVertex(E, V, limit):
    for i in V:
        d = lineValue(E[i], V, limit)
        if (d == 0):
            return (i, 0)
        if (d == 1):
            return (i, 1)
        if (d == 2):
            return (i, 2)
    return (-1, -1)

# ==========================
            
# Algo R0
def algoR0(E, V):
    global calls
    # If the graph is empty
    if (not V):
        return 0
    # If the graph has an isolated vertex
    v, d = seekSmallVertex(E, V, 0)
    if (d == 0):
        V.remove(v);
        calls += 1
        return 1 + algoR0(E, V)
    # Otherwise
    # Find maximum degree
    u = seekMaximumDegree(E, V)
    V2 = list(V)
    V2.remove(u)
    removeNeighbors(E, V, u)
    calls += 2
    return max(1 + algoR0(E, V), algoR0(E, V2))

# Algo R1
def algoR1(E, V):
    global calls
    # If the graph is empty
    if (not V):
        return 0
    # If the graph has an isolated vertex or degree equals 1
    v, d = seekSmallVertex(E, V, 1);
    if (d == 0):
        V.remove(v);
        calls += 1
        return 1 + algoR1(E, V)
    if (d == 1):
        removeNeighbors(E, V, v)
        calls += 1
        return 1 + algoR1(E, V)
    # Otherwise
    # Find maximum degree
    u = seekMaximumDegree(E, V)
    V2 = list(V)
    V2.remove(u)
    removeNeighbors(E, V, u)
    calls += 2
    return max(1 + algoR1(E, V), algoR1(E, V2))

# AlgoR2
def algoR2(E, V):
    global calls
    # If the graph is empty
    if (not V):
        return 0
    v, d = seekSmallVertex(E, V, 2);
    if (d == 0):
        V.remove(v);
        calls += 1
        return 1 + algoR2(E, V)
    if (d == 1):
        removeNeighbors(E, V, v)
        calls += 1
        return 1 + algoR2(E, V)
    if (d == 2):
        u = -1
        w = -1
        for i in range(len(E[v])):
            # We check if i is not a removed node
            if (E[v][i] == 1):
                if (i in V):
                    if (u == -1):
                        u = i
                    else:
                        w = i
        # If u and w neighbors
        if (E[u][w] == 1):
            V.remove(u)
            V.remove(w)
            V.remove(v)
            calls += 1
            return 1 + algoR2(E, V)
        else:
            E2 = np.copy(E)
            n = len(E[0])
            z = np.zeros(shape=(1,n), dtype=np.int)
            E2 = np.append(E2, z, axis=0)
            z = np.append(z, 0)
            z.resize(n+1,1)
            E2 = np.append(E2, z, axis=1)
            for i in range(n):
                if (E2[u][i] or E2[w][i]):
                    E2[n][i] = 1
                    E2[i][n] = 1
            #removeNeighbors(E, V, v)
            V.remove(u)
            V.remove(w)
            V.remove(v)
            V.append(n)
            calls += 1
            return 1 + algoR2(E2, V)
    # Otherwise
    # Find maximum degree
    u = seekMaximumDegree(E, V)
    V2 = list(V)
    V2.remove(u)
    removeNeighbors(E, V, u)
    calls += 2
    return max(1 + algoR2(E, V), algoR2(E, V2))

# Main
calls = 0
(n, E) = read('data/g100.in')
verticles = [i for i in range(n)]
print algoR2(E, verticles)
print calls
