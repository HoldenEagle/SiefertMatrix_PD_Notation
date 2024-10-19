import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.path import Path

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
    print(Finalized_versions)
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
                if isupperON[surf] == surface:
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
                

Finalized_versions= [[2, 0, 3, 1, 2], [1, 0, 1], [3, 2, 3]]
k_e= [(0, 1), (0, 3), (1, 0), (1, 2), (2, 0), (2, 3), (3, 1), (3, 2)]
all_crossings = [(1, 0), (1, 0), (2, 0), (2, 0)]
orientations , surface_rolidex = find_orientations(Finalized_versions , k_e , all_crossings)
print(orientations, surface_rolidex)



            
            
            
            