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

Second Step: Using our knowledge of the incoming and outgoing edges and each crossing, we can create 
a graph data strcuture that will resemble the knot. This is useful in future calculations such as cycle detection,
etc. To simplify this process, I imported the Networkx library. The rest of this code section is pretty 
straightforward. I am using the outgoing edges and matching them with the incoming edges of another crossing. In this 
graph, the crossings are the nodes. We are also storing the information of whether or not the edge is going over or not.
The code in the previous section allowed us to store the edges in (under, over) fashion.

![image](https://github.com/user-attachments/assets/1f404d9a-5252-4f27-949f-c81be6f1755c)

```
import networkx as nx
def create_knot_graph(incoming_edges , outcoming_edges):
    knot_graph = nx.MultiDiGraph()
    knot_graph.add_nodes_from([crossing for crossing in range(num_crossings)])

    for crossing in range(len(pd_notation)):
        out_edge1 , out_edge2 = outcoming_edges[crossing]
        for key in incoming_edges:
            if(out_edge1 in incoming_edges[key]):
                if(incoming_edges[key].index(out_edge1) %2 == 0):
                    knot_graph.add_edge(crossing , key , over = False , visited = False , index = out_edge1)
                else:
                    knot_graph.add_edge(crossing, key , over = True , visited = False  , index = out_edge1)
            
            if(out_edge2 in incoming_edges[key]):
                if(incoming_edges[key].index(out_edge2) %2 == 0):
                    knot_graph.add_edge(crossing , key , over = False , visited = False , index = out_edge2)
                else:
                    knot_graph.add_edge(crossing , key , over = True , visited = False, index = out_edge2)
    return knot_graph
```

Step 3: Create the Siefert Surface for the knot. For the next steps, we must have the Siefert Surface.
For each crossing, we already have the information for the incoming and outgoing crossings, as well as the information
on whether the edge goes under or over on a certain spot. With this information, we can each edge in a hashmap called 
NewBounds, where it will hold the next edge in there. From there, what we can do is we can find these new isolated regions
via cycle detection, and then store these components. A diagram of this process and the code is shown below.

![image](https://github.com/user-attachments/assets/1286b009-4a8b-41a1-b8c9-9b57410eb564)

```
def get_components(incoming_edges , outcoming_edges):
    newBounds = {}
    for crossing in range(num_crossings):
        incoming_under = incoming_edges[crossing][0]
        incoming_over = incoming_edges[crossing][1]
        outgoing_1 = outcoming_edges[crossing][0]
        outgoing_2 = outcoming_edges[crossing][1]
        newBounds[incoming_under] = outgoing_2
        newBounds[incoming_over] = outgoing_1


    newComponents = []
    def cycleDetection(key , surfaces):
        if key in surfaces:
            newComponents.append(surfaces)
            for key in surfaces:
                newBounds[key] = "visited"
            return
        else:
            surfaces.append(key)
            cycleDetection(newBounds[key] , surfaces)
        

    for key in newBounds:
        surfaces = []
        if(newBounds[key] == "visited"):
            continue
        surfaces.append(key)
        cycleDetection(newBounds[key] , surfaces)

    return newComponents
```








