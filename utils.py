import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import fiona
import shapely
from shapely.geometry import shape,mapping, Point, Polygon, MultiPolygon
import networkx as nx
import pickle


def plot_optimal_q(q_path='Q1.pkl', shp_file='taxi_zones.shp', t=32, all_nodes=False, save_path = None):
    q = open(q_path, 'rb')
    data = pickle.load(q)
    edge = [] # action to another zone
    wait = [] # wait in the same zone
    self = [] # action within the same zone
    a = np.zeros((264,))
    v = []
    nodes = []
    
    for i in data:
        if i[1]==t:
            now = data[i]
            # if there is optimal action
            if ((sum(np.equal(now,a))!=264)):
                start = int(i[0])
                end = int(np.argmax(now))
                v.append([start, np.amax(now)])
                nodes.append(start)
                nodes.append(end)
                    
                if (end == 0):
                    wait.append(start)
                elif (start == end):
                    self.append(end)
                else:
                    edge.append((start, end))
                    
    nodes = list(set(nodes))
    
    G = nx.DiGraph()
    taxi_zones = fiona.open(shp_file)
    for j in range(len(taxi_zones)):
        zone = taxi_zones[j]
        i = int(zone['id']) + 1
        shape = shapely.geometry.asShape(zone['geometry'])
        center = shape.centroid.coords[0]
        
        #add node
        if(all_nodes==True):
            G.add_node(i, pos=center) # add node with position
        else:
            if(i in nodes):
                G.add_node(i, pos=center)
    
    color = ['' for k in range(len(list(G.nodes)))] # red for action, blue for wait
    width = np.zeros(len(list(G.nodes)))  # bolder if there is optimal action
    
    for i in range(len(list(G.nodes))):
        n = list(G.nodes)[i]
        if (n in wait):
            color[i] = 'blue'
            width[i] = 3
        elif (n in self):
            color[i] ='red'
            width[i] = 3
        else:
            color[i] ='black'
            width[i] = 1
    
    # add edge
    G.add_edges_from(edge)
    p = nx.get_node_attributes(G,'pos')
    fig = plt.figure(3,figsize=(30,30)) 
    nx.draw_networkx_nodes(G, pos=p, node_color='white', node_size=500, edgecolors = color, linewidths=width)
    nx.draw_networkx_labels(G, pos=p, font_size=10)
    nx.draw_networkx_edges(G, pos = p, width=3, edge_color='red')
    ax = plt.gca() # get the current axis
    ax.collections[0].set_edgecolor(color) 
    fig.suptitle('Optimal Action for Taxi Zones', fontsize=30, y=0.9)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches = 'tight')
    plt.show()

    return v



def plot_v_s(v):
    df = pd.DataFrame(v, columns =['s', 'max_1']) 
    df = df.sort_values(by = 's') 
    df['s'] = df['s'].astype(str)


    fig = plt.figure(figsize=(18,6)) 
    plt.bar(df['s'], df['max_1'])
    plt.xticks(rotation=30, horizontalalignment="center")
    ax = plt.gca()
    myLocator = mticker.MultipleLocator(10)
    ax.xaxis.set_major_locator(myLocator)
    
    plt.xlabel('Taxi Zone ID') 
    plt.ylabel('Maximum q') 
    plt.title('Maximum q for Taxi Zone at Given Time') 
    #plt.savefig('../v_s.png', bbox_inches = 'tight')
    plt.show() 
    return 


def plot_td_error(mean_td_delta, save_path=None, n=5000):
    '''
    Plot the td error
    Args:
    @mean_td_delta: mean td error. Comes from training history['mean_td_delta']
    @n: window size of moving average
    @save_path: plot saving path
    '''
    mean_td_delta = pd.DataFrame(mean_td_delta)
    n = 5000
    mean_td_delta['MA'] = mean_td_delta.rolling(window=n).mean()
    mean_td_delta.MA.plot(figsize=(15,5),legend=None);

    plt.rcParams.update({'font.size': 20})
    plt.xlabel('iterations');
    plt.ylabel('Mean TD Error');
    # plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{}'.format(x/1000) + 'K'))
    plt.gca().xaxis.set_major_formatter(mticker.EngFormatter())
    plt.title('Average TD Error (moving average = {})'.format(n));
    if save_path is not None:
        plt.savefig(save_path, bbox_inches = 'tight')
        print('saved at ',save_path)
    plt.show()
    
def plot_start_end_time(df):
    '''
    Plot start and end time indices of each episode.
    
    Args:
        df: Full SARSA dataset to be plot
    '''
    
    samples = df.copy()
    # extract time from 'state' column (second element in the tuple)
    samples['cur_time'] = [i[1] for i in samples['state']]
    
    # get the start and end time of each episode
    episode_time = samples.groupby('episode').\
                    agg(['first', 'last'])[['cur_time']].rename(columns={'first':'start_time','last':'end_time'})
    episode_time.columns = episode_time.columns.droplevel(0)
    
    # plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].hist(episode_time.start_time)
    ax[1].hist(episode_time.end_time)
    ax[0].set_title('Start')
    ax[0].set_xlabel('Time Index')
    ax[1].set_title('End')
    ax[1].set_xlabel('Time Index')
#     fig.savefig('../start_end_time.png', bbox_inches = 'tight')   

