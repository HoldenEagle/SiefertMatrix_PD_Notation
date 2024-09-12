import networkx as nx
from collections import Counter
import numpy as np
import sympy
from scipy.linalg import det


class Siefert_Matrix():
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