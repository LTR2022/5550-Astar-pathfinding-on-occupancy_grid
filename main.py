# Load the PIL, numpy, and heapq libraries
from PIL import Image, ImageDraw
import numpy as np
import heapq as q
from matplotlib.pyplot import imshow

# Read image from disk using PIL
occupancy_map_img = Image.open('occupancy_map.png')

# Interpret this image as a numpy array, and threshold its values to {0,1}
occupancy_grid = (np.asarray(occupancy_map_img) > 0).astype(int)

# Create a Vertex list
V = []
for i in range(occupancy_grid.shape[0]):
    for j in range(occupancy_grid.shape[1]):
        if occupancy_grid[i][j] == 1:
            V.append((i, j))
print("V[] is ready")

# Assing the starting and goal
s = (635, 140)
# g = (630, 145)
# g = (570, 150)
g = (350, 400)


def RecoverPath(s, g, pred):
    path = []
    path.insert(0, g)
    v = g
    while s not in path:
        path.insert(0, pred[v])
        v = pred[v]

    return path


def N(v):
    neighbors = set()  # emprty neighbors set
    # check if 8 connected vertices around the vertex v are in the occupancy grid
    for i in range(-1, 2):
        for j in range(-1, 2):
            # if ((v[0] + i, v[1] + j) in V) and (i != 0 or j != 0):
            if (occupancy_grid[v[0] + i][v[1] + j] == 1) and (i != 0 or j != 0):
                neighbors.add((v[0] + i, v[1] + j))
    return neighbors


def d(v1, v2):
    # return the euclidian distance between the two vertices
    point1 = np.array(v1)
    point2 = np.array(v2)
    return np.linalg.norm(point1 - point2)


def search(V, s, g, N, d):  # w() and h() are replaced with d()
    # Initialization
    CostTo = {}
    EstTotalCost = {}
    pred = {}
    for v in V:
        CostTo[v] = np.inf
        EstTotalCost[v] = np.inf

    CostTo[s] = 0
    EstTotalCost[s] = d(s, g)

    Q = []
    q.heapify(Q)
    q.heappush(Q, (d(s, g), s))

    # Main loop
    while len(Q) != 0:
        Qv = q.heappop(Q)
        vertex = Qv[1]
        if vertex == g:
            # Print the total length
            print(EstTotalCost[pred[g]])  # The last emstimation is the same as the total length

            path = RecoverPath(s, g, pred)

            # Plot the path on the image
            draw = ImageDraw.Draw(occupancy_map_img)
            for draw_idx in range(len(path) - 1):
                # flip the tuple element of the path list since draw.line recognizes the coordinates as (x,y)
                draw.line((path[draw_idx][1], path[draw_idx][0]) + (path[draw_idx + 1][1], path[draw_idx + 1][0]), fill="red")
            imshow(occupancy_map_img)

            return path

        for k in N(vertex):
            pvi = CostTo[vertex] + d(vertex, k)
            if pvi < CostTo[k]:
                # The path to i through v is better than the previously known best path to i,
                # so record it as the new best path to i
                pred[k] = vertex
                CostTo[k] = pvi
                EstTotalCost[k] = pvi + d(k, g)

                idx = 0
                m = []
                for l in range(len(Q)):
                    m.insert(0, Q[l][1])
                for l in m:
                    # Update i's priority
                    if k == l:
                        Q[idx] = (EstTotalCost[k], k)
                        break
                    idx = idx + 1
                # if i is not in Q
                if idx == len(Q):
                    q.heappush(Q, (EstTotalCost[k], k))

    # Print the total length
    print(CostTo[g])

    return {}  # Return empty set



if __name__ == '__main__':
    print("A* pathfinding start")
    search(V, s, g, N, d)
    occupancy_map_img.save('output.png', 'PNG')
    occupancy_map_img.show()
    print("finish")