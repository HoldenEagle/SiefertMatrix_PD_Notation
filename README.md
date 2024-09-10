# SiefertMatrix_PD_Notation

The goal of this project is to calculate the Siefert Matrix of a knot using a given 
PD Notation of the knot. This file will contain the steps I took to calculate the Siefert Matrix 
and the knot invarient determinent, as well as subsections of code for these steps.

Given: Knot in PD_Notation  , Output: Siefert Matrix and the determinent of the knot.

First step: Find the incoming and outgoing edges at each crossing. The PD_notation contains 
the edges of the knot component numbered in a certain direction. By going to each crossing and
respecting the order of the path, we can find the incoming and outgoing edges at each of the given 
crossings. The path will start from 1 and go to the length of the number of edges, which will be the max value
of the PD Notation. The incoming and outgoing edges will be stored in a hashmap for each given crossing. A picture 
is given below of what is going on, as well as the code for this part.

![image](https://github.com/user-attachments/assets/12289547-182a-4438-b1b5-9fdd4c1acffa)
```
def find_incoming_outgoing(pd_notation):
    max_value = max(max(row) for row in pd_notation)
    incoming_edges = {}
    outcoming_edges = {}
    #add all of the nodes as keys for outcoming and incoming keys
    for node in range(len(pd_notation):
        incoming_edges[node] = []
        outcoming_edges[node]= []
    for crossing in range(len(pd_notation)):
        vertice = pd_notation[crossing]
        incoming_edges[crossing].append(vertice[0])
        outcoming_edges[crossing].append(vertice[2])
        if(vertice[0] > vertice[2]):
            if(vertice[1] > vertice[3]):
                incoming_edges[crossing].append(vertice[3])
                outcoming_edges[crossing].append(vertice[1])
            else:
                incoming_edges[crossing].append(vertice[1])
                outcoming_edges[crossing].append(vertice[3])
        else:
            if(vertice[1] < vertice[3]):
                if(vertice[3] == max_value and vertice[1] == 1):
                    incoming_edges[crossing].append(vertice[3])
                    outcoming_edges[crossing].append(vertice[1])
                else:
                    incoming_edges[crossing].append(vertice[1])
                    outcoming_edges[crossing].append(vertice[3])
            else:
                if(vertice[1] == max_value and vertice[3] == 1):
                    incoming_edges[crossing].append(vertice[1])
                    outcoming_edges[crossing].append(vertice[3])
                else:
                    incoming_edges[crossing].append(vertice[3])
                    outcoming_edges[crossing].append(vertice[1])
                    
    return incoming_edges, outcoming_edges
```






