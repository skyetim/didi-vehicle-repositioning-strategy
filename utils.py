import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import fiona
import shapely
from shapely.geometry import shape,mapping, Point, Polygon, MultiPolygon
import networkx as nx
import pickle


def plot_optimal_q(q_path = 'Q1.pkl', shp_file = 'taxi_zones.shp', t = 32):
    
    q = open(q_path, 'rb')
    data = pickle.load(q)

    edge = [] # action to another zone
    wait = [] # wait in the same zone
    self = [] # action within the same zone
    for i in data:
        if i[1]==t:
            now = data[i]
            # if there is optimal action
            if (max(now)!=0):
                start = int(i[0])
                end = int(np.argmax(now))
                if (end == 0):
                    wait.append(start)
                elif (start == end):
                    self.append(end)
                else:
                    edge.append((start, end))
                    
    G = nx.DiGraph()
    taxi_zones = fiona.open(shp_file)

    color=[] # red for action, blue for weight
    width = [] # bolder if there is optimal action
    for i in range(len(taxi_zones)):
        zone = taxi_zones[i]
        i = int(zone['id']) + 1
        shape = shapely.geometry.asShape(zone['geometry'])
        center = shape.centroid.coords[0]
        G.add_node(i,pos=center) # add node with position
        if (i in wait):
            color.append('blue')
            width.append(3)
        elif (i in self):
            color.append('red')
            width.append(3)
        else:
            color.append('black')
            width.append(1)
    
    # add edge
    G.add_edges_from(edge)
    
    
    p = nx.get_node_attributes(G,'pos')
    plt.figure(3,figsize=(30,30)) 
    nx.draw_networkx_nodes(G, pos=p, node_color='white', node_size=500, edgecolors = color, linewidths=width)
    nx.draw_networkx_labels(G, pos=p, label=G.nodes, font_size=10)
    nx.draw_networkx_edges(G, pos = p, width=3, edge_color='red')
    ax = plt.gca() # get the current axis
    ax.collections[0].set_edgecolor(color) 
    # plt.savefig('../graph2.png')
    plt.show()

    return
