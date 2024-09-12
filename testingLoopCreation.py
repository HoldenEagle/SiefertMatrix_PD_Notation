import networkx as nx
from collections import Counter
import numpy as np
import sympy
#all_crossings = [(1, 0), (1, 0), (1, 0), (1, 2), (1, 2), (1, 0)]
#components = 3

#start creating the loops
#loops = []

class Crossing_Graph():
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_edge_indices = {}
        self.indices = 0
        self.visited_surfaces = None
        self.loops =[]
        self.all_crossings = None
        
    
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

'''
newGraph = Crossing_Graph()
newGraph.create_graph_from_crossings(all_crossings)
newGraph.display()
#loops = newGraph.find_cycles()
newGraph.find_linearly_independent_loops(all_crossings)
'''




