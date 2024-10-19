'''
import graph class for simple graph creation, and the the two necessary classes 
that calculate the matrix
'''

from testingLoopCreation import Crossing_Graph
from matrixTest import Siefert_Matrix

#start off with the pd_notation and find the max value of this matrix. Needed to create the graph of the knot
pd_notation = [[1,11,2,10],[3,15,4,14],[5,13,6,12],[7,17,8,16],[9,1,10,18],[11,3,12,2],[13,5,14,4],[15,7,16,6],[17,9,18,8]]
max_value = max(max(row) for row in pd_notation)

#import graph class to make creating the graph easier
   
import numpy as np
import sympy
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.path import Path


#calculate number of nodes in the graph. (This will be the number of crossings) , equal to the rows in the pd_not
num_crossings = len(pd_notation)

#add these nodes to the graph
#keep a dictionary to store the incoming and outcoming edges for each vertice
#that is how we will construct this knot graph

def find_incoming_outgoing(pd_notation):
    max_value = max(max(row) for row in pd_notation)
    incoming_edges = {}
    outcoming_edges = {}
    #add all of the nodes as keys for outcoming and incoming keys
    for node in range(len(pd_notation)):
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

incoming_edges , outcoming_edges = find_incoming_outgoing(pd_notation)
#print(f"incoming_edges : {incoming_edges}")
#print(f"outcoming_edges : {outcoming_edges}")

'''
For each crossing in our knot diagram, we can use the 
ordering of the incoming and outcoming edges to determine which edges go over and
which edges go under. 
'''

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
knot_graph = create_knot_graph(incoming_edges , outcoming_edges)                
k_e = [(u,v) for u,v in knot_graph.edges()]
print(k_e)
print("all good before here--------------")

'''  
In this section, we can determine at each crossing the position of each edge that goes into the crossing
with this information of the incoming, outcoming, under and over crossing knowledge we can create the 
Siefert surface
'''

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

newComponents = get_components(incoming_edges , outcoming_edges)
print(newComponents)
print("all good before here--------------")
#SECTION TO GET IT READY FOR SURFACE ORIENTATION
edge_versions = []
for comp in newComponents:
    ev = []
    for ind in comp:
        ev += (list((u, v) for u, v, data in knot_graph.edges(data=True) if data.get('index') == ind))
    edge_versions.append(ev)
print(f"Edge_versions: {edge_versions}")

finalized_versions = []
for edg in edge_versions:
    new_cycle = []
    new_cycle += [edg[0][0] , edg[0][1]]
    for i in range(1, len(edg)):
        new_cycle.append(edg[i][1])
    finalized_versions.append(new_cycle)
print(f"Finalized versions: {finalized_versions}")    
print(f"k_e {k_e}")
#WHAT WE SEND IN, FINALIZED_VERSIONS and k_e
'''
Now its time to calculate the crossings. Loop through the crossings in the pd_notation,
and get the original crossing of it. Get the incoming and outcoming crossings and also
determine at the crossing which edges correlate to the understrand and which edges
correlate to the overstrand, and then add this crossing to the list as a tuple with
its overstrand and understrand. Another list also stores whether it is a right or left
crossing. 
'''
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

all_crossings , left_right_crossings = find_crossings(incoming_edges , outcoming_edges)
print(all_crossings)
print(left_right_crossings)

#finding the neighboring surface function. Used later to help find the orientation of the 
#surfaces
def find_orientations(Finalized_versions, k_e , all_crossings):
    knot = nx.DiGraph()
    knot.add_edges_from(k_e)
    orig_pos = nx.planar_layout(knot)
    #nx.draw(knot , orig_pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, arrows=True)
    #plt.show()

    
    actual_points = [orig_pos[u].tolist() for u in orig_pos]

    points_version_of_fv = []

    for fv in Finalized_versions:
        newList = []
        for surf in fv:
            newList.append(tuple(actual_points[surf]))
        points_version_of_fv.append(newList)
    
    #print(points_version_of_fv)


    def find_point(surface):
        big_x = sum([i[0] for i in surface])
        big_y = sum([i[1] for i in surface])
        return [big_x / len(surface) , big_y / len(surface)]


    isupperON = [None for i in Finalized_versions]
    for i , fv in enumerate(Finalized_versions):
        for j , sv in enumerate(Finalized_versions):
            s = list(set(fv) - set(sv))
            if(len(s) == 0 and fv != sv):
                average_point = find_point(points_version_of_fv[i])
                polygon_path = Path(points_version_of_fv[j])
                is_inside = polygon_path.contains_point(average_point)
                if is_inside:
                    isupperON[i] = j
    print(isupperON)

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
    if not starting_node:
        starting_node = 0

    print(neighbors , starting_node)
    print(k_e)
    clockwise = True
    visited = [False for i in range(len(Finalized_versions))]
    orientation = [False for i in range(len(Finalized_versions))]

    def find_orientations(surface , clockwise):
        orientation[surface] = clockwise
        visited[surface] = True
    
        surf_neighbors = neighbors[surface]
        for surf in surf_neighbors:
            if not visited[surf]:
                if isupperON[surf] == surface or isupperON[surface] == surf:
                    find_orientations(surf , not clockwise)
                else:
                    find_orientations(surf , clockwise)
    
    find_orientations(starting_node , clockwise)
    
    
    surface_rolidex = []
    for i , surface in enumerate(Finalized_versions):
        s = surface
        s.pop()
        if not orientation[i]:
            surface_rolidex.append(s[::-1])
        else:
            surface_rolidex.append(s)
        
    
    return orientation , surface_rolidex
                


surface_orientation , surface_rolidex = find_orientations(finalized_versions , k_e , all_crossings)
 
          
'''
Get the rolidex for each surface. This helps with the order of crossings for our surfaces. 
Based on if the surface is clockwise or counterclockwise, the idea is to order the crossings
based on their oreintation so we can find loop intersections that occur on the Siefert Surface  
'''


print(all_crossings)
print(f"surface_orientation: {surface_orientation}")
print(f"New components: {newComponents}")
print(f"surface_rolidex: {surface_rolidex}")



'''
Crossing Graph class. This class is useful for creating the graph from the given PD notation and
finding the linearly independent loops for the loop. From there, we can figure out the matrice's dimensions.
'''

#get linear independent loops
newGraph = Crossing_Graph()
newGraph.create_graph_from_crossings(all_crossings)
loops_we_have =newGraph.find_linearly_independent_loops(all_crossings)

print(loops_we_have)

#Finding the over_unders
'''
From the loops we have, we can calculate the over unders, which desribe the 
loop's process as it travels through each of its crossings and whether it
will travel over or under that given crossing.
'''
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
over_unders = find_over_unders(loops_we_have , all_crossings)
#get previous crossings as well
def find_previous_crossings(loops_we_have):
    prev_crossing = []
    for loop in loops_we_have:
        prev = []
        prev.append(loop[-1])
        for crossing_index in range(1 , len(loop)):
            prev.append(loop[crossing_index-1])
        
        prev_crossing.append(prev) 
    return prev_crossing   

prev_crossing = find_previous_crossings(loops_we_have)

#Matrix_test class, allows us to calculate the Siefert matrix from the information we received from above
#also this class allows us to calculate certain knot invarients given the matrix, such as the determinant
SM = Siefert_Matrix(loops_we_have , left_right_crossings , surface_rolidex , over_unders , prev_crossing , all_crossings , surface_orientation)
SM.add_entries()

det = SM.compute_determinant_invarient()
print(int(det))


