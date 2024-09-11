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

Step 4: Find all the band crossings on the Siefert Surface. For each crossing in the PD notation matrix,
we find all of the incoming and outgoing edges again. We find the two surfaces involved and then label that crossing.
Whichever surface is over the other is the first surface mentioned in the crossing tuple. These tuples are all stored in the
all_crossing list. We also try to figure out whether a crossing is a left or right crossing in this stage, and store an array
of +1's and -1's to hold the oreintation of each crossing. I believe right crossings are represented with a +1.

![image](https://github.com/user-attachments/assets/93b1408b-e433-4a56-a315-e33d8e9363fe)

```
def find_crossings(incoming_edges , outcoming_edges):
    all_crossings = []
    left_right_crossings = []
    for crossing in range(num_crossings):
        vertice = pd_notation[crossing]
        incoming_under = incoming_edges[crossing][0]
        incoming_over = incoming_edges[crossing][1]
        outgoing_1 = outcoming_edges[crossing][0]
        outgoing_2 = outcoming_edges[crossing][1]
    
        for i in range(len(newComponents)):
            if incoming_under in newComponents[i] and   outgoing_2 in newComponents[i]:
                underStrand = i
            if incoming_over in newComponents[i] and   outgoing_1 in newComponents[i]:
                overStrand = i
        all_crossings.append((overStrand , underStrand))
        left_right_crossings.append(1 if (incoming_over == vertice[3]) else -1)
    return all_crossings , left_right_crossings
```
Step 5: Find the Surface Orientations for each surface on the knot. This is the step that will
aid the process of creating the surface rolidex later on. The goal of this step is to find the oreintation of each surface.
We start with a hashmap that will store the neighboring surfaces. This will allow us to find the neighbor count of each surface.
In the next step we have to find a surface to start the traversal on. I decided to try 
to start on a surface with only one neighbor, and if this wasn't possible, then start at the first surface in the components list.
After this step, I reconstructed a graph of the surfaces using the crossings tuples, with each surface as a node and the edges as 
the crossings between them. I did this using a custom built class Crossing_Graph(), which I will explain later on, but from here, I
performed a depth first traversal to get to all of the surfaces and the oreintation of each. I started the beginning surface as 
True, resembling that we are starting clockwise. I start from the beginning surface and branch out, and since each surface is connected in this component
we will reach all of them. If the surface is over another and is already clockwise, the surface on the  new under side will be counterclockwise, or if the surface
that is over is counterclockwise, the under side will be clockwise. If the under surface is clockwise, the new over surface will be clockwise, and vice versa.

```
def get_orientation(all_crossings):
    neighbors = {}
    for crossing in all_crossings:
        over , under = crossing
        if over not in neighbors:
            neighbors[over] = [under]
        if under not in neighbors:
            neighbors[under] = [over]
    
        if over not in neighbors[under]:
            neighbors[under].append(over)
        if under not in neighbors[over]:
            neighbors[over].append(under)


    starting_node = None
    for node in neighbors:
        if len(neighbors[node]) < 2 and not starting_node:
            starting_node = node
    #Surface front/back orientation
    surfaceGraph = Crossing_Graph()
    surfaceGraph.create_graph_from_crossings(all_crossings)
    #start at starting node
    surface_orientation = {}
    visited = [False for _ in range(len(surfaceGraph.nodes))]
    surface_orientation[starting_node] = True
    visited[starting_node] = True

    def findOrientation(over, under , original):
        if visited[over] and visited[under]:
            return
        elif visited[over]:
            surface_orientation[under] = False if surface_orientation[over] else True
            visited[under] = True
        elif visited[under]:
            surface_orientation[over] = True if surface_orientation[under] else False
            visited[over] = True
        nextSurf = over if over != original else under
        for neighbor in surfaceGraph.node_edge_indices[nextSurf]:
            crossing_ = all_crossings[neighbor]
            findOrientation(crossing_[0] , crossing_[1] , nextSurf)
    for neighbor in surfaceGraph.node_edge_indices[starting_node]:
        crossing_ = all_crossings[neighbor]
        findOrientation(crossing_[0] , crossing_[1] , starting_node)
        
    return surface_orientation
```
Step 6: Using the surface orientation for each surface, we create the surface rolidex for each surface.
Basically we go through each surface and append its crossings associated with the surface. If the surface 
orientation is counterclockwise, we reverse the order of the crossings for the array of crossings for that surface.
I believe this is a section I have to switch up to make the Siefert Matrix calculation.
```
edges_with_data = knot_graph.edges(data= True)
index = [index for index in edges_with_data if index[2]['index'] == 1]
surface_rolidex = []
#print(surface_orientation)
print(all_crossings)
for components_index in range(len(newComponents)):
    crossing_positions = []
    for edge in newComponents[components_index]:
        edge_dest = [index[1] for index in edges_with_data if index[2]['index'] == edge]
        crossing_positions.append(edge_dest[0])
    if not surface_orientation[components_index]:
        crossing_positions = crossing_positions[::-1]
    surface_rolidex.append(crossing_positions)
```
```
For the next step we must find the linearly independent loops that make up the homology group
of the knot. To do this, we use the functions in my custom class that I created, Crossing Graph,
such as the create graph given the crossings and a helper function that can detect cycles in a graph.
For this section I represent each crossing as a linear independent vector, and each loop we create
cannot be in the existing vector span. Below is the code for the linear independent loop function, as well
as the create graph function and the detect cycle function. This code displays the entire class for the
Crossing Graph, which includes the linear independent loops function (find_linear_independent_loops). The loops
will be returned in a matrix where each element in the list is a sublist of crossings that the loop will
traverse.

```
```
class Crossing_Graph():
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_edge_indices = {}
        self.indices = 0
        self.visited_surfaces = None
        self.loops =[]
        self.all_crossings = None
```
```    
    
    def add_node(self, value):
        if value not in self.nodes:
            self.nodes.append(value)
            if value not in self.node_edge_indices:
                self.node_edge_indices[value] = []
        
    
    def add_edge(self, src_node , dest_node):
        if src_node in self.nodes and dest_node in self.nodes:
            self.edges.append([src_node , dest_node])
            self.node_edge_indices[src_node].append(self.indices)
            self.node_edge_indices[dest_node].append(self.indices)
            self.indices += 1
            
    def create_graph_from_crossings(self , all_crossings):
        #add all the nodes first
        self.all_crossings = all_crossings
        for crossings in all_crossings:
            over , under = crossings
            self.add_node(over)
            self.add_node(under)
        
        for crossings2 in all_crossings:
            over , under = crossings2
            self.add_edge(over, under)
    def display(self):
        print(self.nodes)
        print(self.edges)
        print(self.node_edge_indices)
        print("-----------------")
        
    def dfs_cycles(self, surface_dest , surface , path):
        path.append(surface)
        #get its neighbors
        neighbors = self.node_edge_indices[surface]
        for neighbor in neighbors:
            if self.visited_surfaces[neighbor]:
                continue
            #get next crossing that isn't equal to surface
            next_surface = self.edges[neighbor][0] if self.edges[neighbor][0] != surface else self.edges[neighbor][1]
            if next_surface == surface_dest:
                self.loops.append(path[:])
                continue
            else:
                self.visited_surfaces[neighbor] = True
                next_surface = self.edges[neighbor][0] if self.edges[neighbor][0] != surface else self.edges[neighbor][1]
                self.dfs_cycles(surface_dest , next_surface ,path)
        path.pop()
    
    
    def detect_cycles(self , loops):
        unique_loops_pre = []
        for loop in loops:
            start_surface = loop[0]
            for crossing in self.all_crossings:
                if crossing[0] == start_surface:
                    unique_loops_pre.append(loop)
        unique_loops = []
        for loop_ind in range(len(unique_loops_pre)):
            loop = unique_loops_pre[loop_ind]
            can_keep_it = True
            starting_surface= loop[0]
            for next_loop_ind in range(loop_ind +1 , len(unique_loops_pre)):
                next_loop = unique_loops_pre[next_loop_ind]
                if len(next_loop) != len(loop) or starting_surface not in next_loop:
                    continue
                else:
                    ind = next_loop.index(starting_surface)
                    cycle = next_loop[ind:] + next_loop[:ind]
                    if cycle == loop:
                        can_keep_it = False
            if loop not in unique_loops and loop[::-1] not in unique_loops and can_keep_it:
                unique_loops.append(loop)
        return unique_loops
    
    def find_cycles(self):
        for surface in range(len(self.nodes)):
            path = []
            self.visited_surfaces = [False for i in range(len(self.edges))]
            self.dfs_cycles(surface , surface , path)
        kept_loops = self.detect_cycles(self.loops)
        print(kept_loops)
        return kept_loops
    
    def find_linearly_independent_loops(self , all_crossings):
        loops = self.find_cycles()
        for loop in loops:
            loop.append(loop[0])
        #print(f"loops: {loops}")
        loops = sorted(loops , key=len)
        crossing_vectors = {}
        for crossing_index in range(len(all_crossings)):
            vector = np.zeros(len(all_crossings))
            vector[crossing_index] = 1
            crossing_vectors[crossing_index] = vector
        #print(crossing_vectors)

        loops_we_have = []
        vector_span = None

        for crossing_index in range(len(all_crossings)):
            crossing = all_crossings[crossing_index]
            next_occurence = -1
            for crossing_index2 in range(crossing_index+1 , len(all_crossings)):
                if all_crossings[crossing_index2] == crossing or all_crossings[crossing_index2] == crossing[::-1]:
                    next_occurence = crossing_index2
                    break
            if next_occurence == -1:
                continue
    
            loops_we_have.append([crossing_index , next_occurence])
            vector1 , vector2 = crossing_vectors[crossing_index].reshape(-1, 1) , crossing_vectors[next_occurence].reshape(-1 , 1)
            if(vector_span is None):
                if(crossing[0] > crossing[1]):
                    vector_span = np.add(-1 * vector1 , vector2)
                else:
                    vector_span = np.add(vector1 , -1 * vector2)
            else:
                if(crossing[0] > crossing[1]):
                    vector_span = np.hstack((vector_span , np.add(-1 *vector1 , vector2)))
                else:
                    vector_span = np.hstack((vector_span , np.add(vector1 , -1 * vector2)))

        for cycles in loops:
            if(len(cycles) == 3):
                continue
            else:
                new_vect = np.zeros(len(all_crossings))
                crossing = None
                crossing_index = []
                for index in range(1 , len(cycles)):
                    if((cycles[index] , cycles[index-1]) in all_crossings):
                        cross_occurence_vect = crossing_vectors[all_crossings.index((cycles[index] , cycles[index-1]))].reshape(-1,1)
                        crossing = (cycles[index] , cycles[index-1])
                        crossing_index.append(all_crossings.index((cycles[index] , cycles[index-1])))
                    else:
                        cross_occurence_vect = crossing_vectors[all_crossings.index((cycles[index-1] , cycles[index]))].reshape(-1,1)
                        crossing = (cycles[index-1] , cycles[index])
                        crossing_index.append(all_crossings.index((cycles[index-1] , cycles[index])))
                    if(cycles[index] > cycles[index-1]):
                        new_vect = np.add(cross_occurence_vect , new_vect.reshape(-1 ,1))
                    else:
                        new_vect = np.subtract(new_vect.reshape(-1 ,1) ,cross_occurence_vect)
                if vector_span is None:
                    loops_we_have.append(crossing_index)
                    vector_span = new_vect
                else:
                    test_array = np.hstack((vector_span , new_vect))
                    num_columns = len(test_array[0])
                    rref_matrix, pivots = sympy.Matrix(test_array).rref()
                    if num_columns-1 in pivots:
                        loops_we_have.append(crossing_index)
                        vector_span = np.hstack((vector_span , new_vect))

        return loops_we_have
```
After this step, we only need to calculate two more lists before starting to create the matrix. I decided to calculate
when each loop goes over and under at each crossing they traverse, as well as the previous crossing for each position on
the linearly independent loops. The goal of this is to make the calculations for the matrix entries easier by knowing whether
the loop is going under or over at each crossing, and if we have that information for both loops in the intersection, then it
becomes easy to calculate the value to be placed in the Siefert Matrix. The previous crossing matrix will allow us to detect
the invisible crossings more efficiently. Knowing where the loop is exactly moving on the surface will make it easier to
detect such crossings. Both of these calculations are simple, the previous crossings is stored as a cycle of the previous
crossings in the loop, and the over under calculation just involves starting on an over crossing, and then going through the
crossings it involves and checking for the new suface. If its the first element of the crossing, the loop is going over on
this crossing.



```
def find_over_unders(loops_we_have , all_crossings):
    over_unders = []
    #get the over/ under information , over = True, under = False
    for loop in loops_we_have:
        prev_surf = all_crossings[loop[0]][1]
        ov_und = []
        ov_und.append(True)
        for crossing_index in range(1 , len(loop)):
            crossing = loop[crossing_index]
            if all_crossings[crossing][0] == prev_surf:
                ov_und.append(True)
            else:
                ov_und.append(False)
            prev_surf = all_crossings[crossing][1]
            
        over_unders.append(ov_und)
    return over_unders
```
```
def find_previous_crossings(loops_we_have):
    prev_crossing = []
    for loop in loops_we_have:
        prev = []
        prev.append(loop[-1])
        for crossing_index in range(1 , len(loop)):
            prev.append(loop[crossing_index-1])
        
        prev_crossing.append(prev) 
    return prev_crossing
```
Now that we have all of the information we need, it is time to create the Siefert Matrix. I created my own custom class called Matrix_test where
I put multiple helper functions for add entries into the Siefert Matrix, along with the function to compute the Matrix. Some of these helper function
include computing the similar surfaces that two loops travel on. I originally intialize the matrix class with these two lines, and it creates a 
Numpy Array of all zeroes to the dimensions of the loops we have. Then we can begin to add entries to the matrix

```
SM = Matrix_test(loops_we_have , left_right_crossings , surface_rolidex , over_unders , prev_crossing , all_crossings , surface_orientation)
SM.add_entries()
```

Then we begin to add entries to the matrix. I loop through the dimensions of the columns and the rows, and each index corrsponds to the
index for the loop in the loops_we_have array. The loop represented by the column index will always be the loop that goes other the other
corresponding loop. First we check if the row loop is equal to the column loop, meaning they have the same index, if they do, we can 
iterate through each crossing they go through, check the direction of the crossing, and add it on, then divide that linking number by 
2 and adding it to the matrix. If not, we begin by finding the similar crossings between the two loops using set intersection. If they 
have common crossings, then they will not have any invisible case crossings, but we go to the crossings that they share, and we check
if each loop is over and under, and based on that information we can compute the linking number for these loop intersections.
If they do not share similar crossings, we check for invisible cases. First, we check if the loops contain similar surfaces, and they do not,
we can just add a zero for the Siefert Matrix entry. On this shared surface, we check whether the loops cross eachother. Using the previous crossing
information from earlier, we can map out on the surface where each loop travels, and we check if the previous crossing or the next crossing is 
in the range of the other loops, using the XOR method to make sure not both crossings are in the range, or else there
will be no intersection. After this, we check to make sure one of the loops is traveling above the other. If they are both above or 
both horizontal, then no invisible crossing will occur. If this condition is the case, we then construct the PD_Notation of
the crossing between these two loops, so we can figure out whether this invisible crossing is a left or right crossing.
From there, we can add it to the matrix. 

The code for the entire class is below, as well as the add_entries function, which is the main function for creating the Siefert Matrix
```
import networkx as nx
from collections import Counter
import numpy as np
import sympy
from scipy.linalg import det


class Matrix_test():
    def __init__(self , loops_we_have , left_right_crossings , surface_rolidex , over_unders , prev_crossing , all_crossings , surface_orientation):
        self.loops_we_have = loops_we_have
        self.left_right_crossings = left_right_crossings
        self.surface_rolidex = surface_rolidex
        self.over_unders = over_unders
        self.prev_crossings = prev_crossing
        self.all_crossings = all_crossings
        self.surface_orientation = surface_orientation
        self.matrix = np.zeros((len(loops_we_have) , len(loops_we_have)))
    
    
    def compute_determinant_invarient(self):
        matrix_T = np.transpose(self.matrix)
        neg_mat = matrix_T * -1
        
        resulting_mat = self.matrix - neg_mat
        det1 = det(resulting_mat)
        return np.abs(round(det1))
    
    
    def get_same_surfaces(self , loop1, loop2):
        #get surfaces for loop1
        loop1_surfaces = []
        loop2_surfaces = []
        for crossing in loop1:
            over , under = self.all_crossings[crossing]
            if over not in loop1_surfaces:
                loop1_surfaces.append(over)
            if under not in loop1_surfaces:
                loop1_surfaces.append(under)
        for crossing in loop2:
            over , under = self.all_crossings[crossing]
            if over not in loop2_surfaces:
                loop2_surfaces.append(over)
            if under not in loop2_surfaces:
                loop2_surfaces.append(under)
                
        #find if they share similar surfaces 
        surfaces1 = set(loop1_surfaces)
        surfaces2 = set(loop2_surfaces)
        sharedSurfaces = list(surfaces1.intersection(surfaces2))
        if len(sharedSurfaces) == 0:
            return None
        else:
            return sharedSurfaces
        
    
    def add_entries(self):
        for row in range(len(self.matrix)):
            for column in range(len(self.matrix)):
                
                #diagonal condition first (this part we have figured out)
                if row == column:
                    loop_studied = self.loops_we_have[row]
                    linking_num = 0
                    for crossing in loop_studied:
                        linking_num += self.left_right_crossings[crossing]
                
                    self.matrix[row][column] = (linking_num/2)
                else:
                    row_loop = self.loops_we_have[row]
                    column_loop = self.loops_we_have[column]
            
                    row_set = set(row_loop)
                    column_set = set(column_loop)
            
                    common_crossings = list(row_set.intersection(column_set))
                    linking_num = 0
                    if common_crossings:
                        #iterate through shared crossings
                        for same_cross in common_crossings:
                            row_cross_is_over = self.over_unders[row][row_loop.index(same_cross)]
                            column_cross_is_over = self.over_unders[column][column_loop.index(same_cross)]
                            #find if crossing is right or left
                            crossing_is_right = self.left_right_crossings[same_cross] == 1
                            if column_cross_is_over and not row_cross_is_over:
                                continue
                
                            elif not column_cross_is_over and row_cross_is_over:
                                if crossing_is_right:
                                    linking_num += 1
                                else:
                                    linking_num -= 1
                
                            elif column_cross_is_over and row_cross_is_over:
                                #Get rolidex positions
                                initial_surface = self.all_crossings[same_cross][0]

                                rolodex = self.surface_rolidex[initial_surface]
                    
                                clockwise = self.surface_orientation[initial_surface]
                                row_prev_cross = self.prev_crossings[row][row_loop.index(same_cross)]
                                column_prev_cross = self.prev_crossings[column][column_loop.index(same_cross)]
                                row_prev_cross_index = rolodex.index(row_prev_cross)
                                column_prev_cross_index = rolodex.index(column_prev_cross)
                                if row_prev_cross_index == column_prev_cross_index:
                                    if column > row:
                                        linking_num += 1 if crossing_is_right else -1
                                    else:
                                        continue
                                elif clockwise:
                                    if crossing_is_right:
                                        linking_num += 1 if row_prev_cross_index >= column_prev_cross_index else 0
                                    else:
                                        linking_num -= 1 if row_prev_cross_index <= column_prev_cross_index else 0
                    
                    
                                else:
                                    if crossing_is_right:
                                        linking_num += 1 if row_prev_cross_index <= column_prev_cross_index else 0
                                    else:
                                        linking_num -= 1 if row_prev_cross_index >= column_prev_cross_index else 0
                    
                    
                            else: #both are under
                                print(f"Both under {row , column}")
                                initial_surface = self.all_crossings[same_cross][1]
                        
                                rolodex = self.surface_rolidex[initial_surface]
                                
                                clockwise = self.surface_orientation[initial_surface]
                        
                                row_prev_cross = self.prev_crossing[row][row_loop.index(same_cross)]
                                column_prev_cross = self.prev_crossing[column][column_loop.index(same_cross)]
                    
                                row_prev_cross_index = rolodex.index(row_prev_cross)
                                column_prev_cross_index = rolodex.index(column_prev_cross)
                                if row_prev_cross_index == column_prev_cross_index:
                                    if column > row:
                                        linking_num += 1 if crossing_is_right else -1
                                    else:
                                        continue
                                elif clockwise:
                                    if crossing_is_right:
                                        linking_num += 1 if row_prev_cross_index >= column_prev_cross_index else 0
                                    else:
                                        linking_num -= 1 if row_prev_cross_index <= column_prev_cross_index else 0
                    
                    
                                else:
                                    if crossing_is_right:
                                        linking_num += 1 if row_prev_cross_index <= column_prev_cross_index else 0
                                    else:
                                        linking_num -= 1 if row_prev_cross_index >= column_prev_cross_index else 0
                        
                    
                    
                    else:
                        #time for invisible case
                        #The possible invisible case
                        shared_surfaces =self.get_same_surfaces(row_loop , column_loop)
                        #from that surface, find the crossings it goes through on this surface
                        if not shared_surfaces:
                            linking_num += 0
                        else:    
                            for shared_surface in shared_surfaces:
                                row_loop_crossings_through_surface = []
                                column_loop_crossings_through_surface = []
                                for crossing in row_loop:
                                    over , under = self.all_crossings[crossing]
                                    if over == shared_surface or under == shared_surface:
                                        row_loop_crossings_through_surface.append(crossing)
                                for crossing in column_loop:
                                    over, under = self.all_crossings[crossing]
                                    if over == shared_surface or under == shared_surface:
                                        column_loop_crossings_through_surface.append(crossing)
                                
                                
                                row_loop_surface_indices = [self.surface_rolidex[shared_surface].index(row_loop_crossings_through_surface[0]) , self.surface_rolidex[shared_surface].index(row_loop_crossings_through_surface[1])]
                                column_loop_surface_indices = [self.surface_rolidex[shared_surface].index(column_loop_crossings_through_surface[0]) , self.surface_rolidex[shared_surface].index(column_loop_crossings_through_surface[1])]
                                inRange = False
                                r1 , r2 = row_loop_surface_indices
                                condition1 = r1 in range(column_loop_surface_indices[0], column_loop_surface_indices[1])
                                condition2 = r2 in range(column_loop_surface_indices[0], column_loop_surface_indices[1])
                                condition3 = r1 in range(column_loop_surface_indices[1], column_loop_surface_indices[0])
                                condition4 = r2 in range(column_loop_surface_indices[1], column_loop_surface_indices[0])
                                if (condition1 ^ condition2) or (condition3 ^ condition4):
                                    inRange = True
                                if not inRange:
                                    c1 , c2 = column_loop_surface_indices
                                    condition1 = c1 in range(row_loop_surface_indices[0], row_loop_surface_indices[1])
                                    condition2 = c2 in range(row_loop_surface_indices[0], row_loop_surface_indices[1])
                                    condition3 = c1 in range(row_loop_surface_indices[1], row_loop_surface_indices[0])
                                    condition4 = c2 in range(row_loop_surface_indices[1], row_loop_surface_indices[0])
                                    if (condition1 ^ condition2) or (condition3 ^ condition4):
                                        inRange = True
                                if not inRange:
                                    continue
                                else:
                                    #calculate if horizontal loop or vertical loop (horizontal -> changes clockwise, vertical -> does not)
                                    #first_surface_row , second_surface_row = self.all_crossings[row_loop[0]]
                                    #first_surface_column , second_surface_column = self.all_crossings[column_loop[0]]
                                    for loop in row_loop:
                                        frst , scnd = self.all_crossings[loop]
                                        if frst == shared_surface or scnd == shared_surface:
                                            first_surface_row , second_surface_row = frst , scnd
                                    for loop in column_loop:
                                        frst , scnd = self.all_crossings[loop]
                                        if frst == shared_surface or scnd == shared_surface:
                                            first_surface_column , second_surface_column = frst , scnd
                                    
                                    #true if horizonatal, false if vertical
                                    horiz_vert_loops = []
                                    horiz_vert_loops.append(True if self.surface_orientation[first_surface_row] != self.surface_orientation[second_surface_row] else False)
                                    horiz_vert_loops.append(True if self.surface_orientation[first_surface_column] != self.surface_orientation[second_surface_column] else False)
                                    if horiz_vert_loops[0] == horiz_vert_loops[1]:
                                        continue
                                    
                                    vert_isRow = False
                                    vertical_path = None
                                    if horiz_vert_loops[0]:
                                        vert_isRow = True
                                        vertical_path = row_loop_crossings_through_surface
                                    else:
                                        vertical_path = column_loop_crossings_through_surface
                                    
                                    
                                    #find if its lower or above
                                    lower_or_above = None
                                    vert_crossing = self.all_crossings[vertical_path[1]]
                                    if vert_crossing[0] == shared_surface:
                                        lower_or_above = "lower"
                                    else:
                                        lower_or_above = "upper"
                                    
                                    #vert is row is wrong
                                    if (lower_or_above == "upper" and not vert_isRow) or (lower_or_above == "lower" and vert_isRow):
                                        continue
                                    #figure out whether its a right handed or not
                                    combined_path = row_loop_surface_indices + column_loop_surface_indices
                                    surf_orient = self.surface_orientation[shared_surface]
                                    max_value = max(combined_path)
                                    pd_notation = []
                                    if vert_isRow:
                                        pd_notation = [column_loop_surface_indices[0] , row_loop_surface_indices[0] , column_loop_surface_indices[1] , row_loop_surface_indices[1]]
                                        
                                    else:
                                        pd_notation = [row_loop_surface_indices[0] , column_loop_surface_indices[0] , row_loop_surface_indices[1] , column_loop_surface_indices[1]]                                
                                    linking_num += 1 if (pd_notation[1] == pd_notation[3]) else -1
                                    
                                    
                         
                                
                                
                    
                    
                    
                    
                    
                    
                    self.matrix[row][column] = linking_num   
                    
                      
        print(self.matrix)
```
Final Step: Computing the knot invarient Determinant from the Siefert Matrix. This is a very simple step, and this
function is built into the custom siefert matrix class I built for the previous step. This determinent 
function involves finding the transpose, finding the determinent of the subtraction of the two matrices, 
and returning the absolute value.

```
def compute_determinant_invarient(self):
        matrix_T = np.transpose(self.matrix)
        neg_mat = matrix_T * -1
        
        resulting_mat = self.matrix - neg_mat
        det1 = det(resulting_mat)
        return np.abs(round(det1))
```




























