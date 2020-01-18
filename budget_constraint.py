#!/usr/bin/env python
# coding: utf-8

# In[4]:


"""
PROJECT TOPIC:Relay Node Placement Problem with Minimum Number of Connected Components.

TEAM MEMBERS:

Rohith Kumar Punithavel   1215339827

INPUT: 1. Text file connections.txt which gives the connection between the nodes.
       2. Text file nodeposition.txt which gives the position of the nodes as x,y coordinates.
       3. Range: range_budget denoting the range of Sensor and Budget nodes.
       4. Budget: budget denoting the budget for the task.
       
OUTPUT: Forest tree denoting the relay node placement between sensor nodes.
"""

import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt

eucdist=[]
weights=[]
tot_weight=0
G=nx.Graph()
a=[]
G1=nx.Graph()

#taking coordinates for nodes from text file
input1 = np.loadtxt("nodeposition.txt", dtype= 'i', delimiter=',')
total_size=np.size(input1,0)
print('Coordinates for sensors in order of sensor number:')
print(input1)

#taking connections between nodes from text file
input2 = np.loadtxt("connections.txt", dtype= 'i', delimiter=',')
total_connection_size=np.size(input2,0)
print('Represents connection between sensors in terms of sensor number:')
print(input2)

#input range
range_budget=float(input('Enter Range: '))

#input budget
budget=int(input('Enter Budget: '))

#calculating euclidean distance
for i in range(total_connection_size):
    i1=input2[i][0]-1
    i2=input2[i][1]-1
    eucdist.append(math.sqrt((input1[i2][0]-input1[i1][0])**2 +(input1[i2][1]-input1[i1][1])**2))
print(' Euclidean distance between sensors')
for i in range(total_connection_size):
    print(input2[i],eucdist[i])

#plotting nodes and connections
plt1=plt.figure(1)
for i in range(total_connection_size):
    k1=input2[i][0]-1
    k2=input2[i][1]-1
    plt.plot([input1[k1][0],input1[k2][0]],[input1[k1][1],input1[k2][1]],'r-')
    plt.plot(input1[k1][0],input1[k1][1],'bo')
    plt.plot(input1[k2][0],input1[k2][1],'bo')
plt.xlabel('x - axis')
plt.ylabel('y - axis')  
plt.title('Graph')
plt.axis('equal')
plt.plot(input1[k1][0],input1[k1][1],'bo',label='Nodes')
plt.plot([input1[k1-1][0],input1[k2-1][0]],[input1[k1-1][1],input1[k2-1][1]],'r-',label='Connection between nodes')
plt.legend(loc='upper left')
for i in range(total_size):
    plt.annotate(i+1, (input1[i][0], input1[i][1]),fontsize='xx-large')
plt.savefig('Graph.png')
plt1.show()

#calculate weights and form minimum spanning tree
for i in range(total_connection_size):
    if eucdist[i]>range_budget:
        weight1=math.ceil(eucdist[i]/range_budget)-1
        G.add_edge((input2[i][0]),(input2[i][1]),weight = weight1)
T=nx.minimum_spanning_tree(G)
print('Minimum Spanning Tree created for nodes that are out of range: ')
print(T.edges(data=True))
a= list(T.edges())

#calculating total weight for the Minimum spanning tree
for i in range(len(a)):
    a1=a[i][0]
    a2=a[i][1]
    tot_weight=tot_weight+T[a1][a2]['weight']
E=sorted(T.edges())
n=len(E)
flag=0

#removing the highest weighted nodes in minimum spanning tree
while tot_weight>budget:
    flag=1
    G1.clear()
    for i in range(n):
        if (E[n-1][0]==E[i][0]) and (E[n-1][1]==E[i][1]):
            continue
        else:
            G1.add_edge(E[i][0],E[i][1],weight=T[E[i][0]][E[i][1]]['weight'])
    tot_weight=tot_weight-T[E[n-1][0]][E[n-1][1]]['weight']   
    n=n-1
if flag==1:
    a1=list(G1.edges(data=True))
else:
    a1=list(T.edges(data=True))
print(' OUTPUT: ')
print('The obtained forest is (Weight denotes the number of required relay sodes betweent the two sensor nodes): ')
print(a1)

#plots the result
plt2=plt.figure(2)
i1=0
j=a1[i1][0]-1
k=a1[i1][1]-1
for i in range(total_connection_size):
    v1=input2[i][0]-1
    v2=input2[i][1]-1
    if(((v1+1)!=input2[j][0])and((v2+1)!=input2[k][1])):
        if i1<len(a1):
            j=a1[i1][0]-1
            k=a1[i1][1]-1
            l=a1[i1][2]['weight']
            plt.plot([input1[j][0],input1[k][0]],[input1[j][1],input1[k][1]],'y-')
            plt.plot(((input1[j][0]+input1[k][0])/2),((input1[j][1]+input1[k][1])/2),'bo')
            i1=i1+1
    plt.plot(input1[v1][0],input1[v1][1],'mo')
    plt.plot(input1[v2][0],input1[v2][1],'mo')
plt.xlabel('x - axis')
plt.ylabel('y - axis')  
plt.title('Plot after finding Forest')
plt.axis('equal')
plt.plot(((input1[j][0]+input1[k][0])/2),((input1[j][1]+input1[k][1])/2),'bo',label='Relay Nodes')
plt.plot(input1[0][0],input1[0][1],'mo',label='Sensor Noes')
plt.plot([input1[j][0],input1[k][0]],[input1[j][1],input1[k][1]],'y-',label='Forest Connection')
plt.text(2,2.5, '** doesnot represent the number of relay nodes but just placement')
plt.legend(loc='upper left')

for i in range(total_size):
    plt.annotate(i+1, (input1[i][0], input1[i][1]),fontsize='xx-large')
plt.savefig('Forest_nodes.png')
plt2.show()


# In[ ]:




