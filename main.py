import networkx as nx
import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import random
import math
from scipy.optimize import fsolve
import csv
from tqdm import tqdm
import copy
import pandas as pd
import multiprocessing as mp
import traceback
import os


#NETWORK_RATIONALITY = 0.00000000000000001
NETWORK_RATIONALITY = 0.01
iterations = 100

def reassign_lr_dict(lr_dict):
    up_sort_keys = list(dict(sorted(lr_dict.items(),  key=lambda d: d[1], reverse=False)).keys())
    down_sort_values = list(lr_dict.values())
    down_sort_values.sort(reverse=True)
    reassigned_dict = dict(zip(up_sort_keys, down_sort_values))

    return reassigned_dict

def read_all_links(file_path):
    df = pd.read_csv(file_path, header=None)
    df_list = df.values.tolist()
    return df_list

def set_graph(all_links):#implementing the graph from the link list
    G1 = nx.Graph()
    for i in range(len(all_links)):
        G1.add_edge(all_links[i][0],all_links[i][1])
    return G1

def create(n,m): ### creation of a random fully connceted network with N nodes and E edges
    G = nx.gnm_random_graph(n, m, seed=0)

    if nx.number_connected_components(G) > 1: #check number of the connected component of the network > 1
         while nx.number_connected_components(G) > 1:
            G = nx.gnm_random_graph(n, m)              # generate a network untill number of the connected component of the network == 1

    return G

def nodes_connectivity(u,v,g):     #checking the connection between nodes
    return u in g.neighbors(v)

def get_links(N,G): #get the link list of the network
    links=[]
    e=0
    for i in range(N):
        for j in range(i+1,N):
            if ((nodes_connectivity(j,i,G)) and (i!=j)):
                links.append([int(i),int(j),int(e)]) # append each link with a unique link number to links array
                e=e+1
    return(links)

def set_graph(all_links): # implementing the graph from the link list
    G1 = nx.Graph()
    for i in range(len(all_links)):
        G1.add_edge(all_links[i][0],all_links[i][1])
    return G1

def check_neighbour(all_links,iNode,jNode): #check whether link is exist with given 2 nodes
    s=False
    E=len(all_links)
    for i in range(E):
        [p,q,r]=all_links[i]
        if((p==iNode and q==jNode) or(q==iNode and p==jNode)):
            s=True
    return s

def neighbour_nodes(all_links,Node):

    node_list = []
    E=len(all_links)
    for i in range(E):
        [p,q,r]=all_links[i]

        if p == Node:
            node_list.append(q)

        if q == Node:
            node_list.append(p)

    return node_list

def init_rationility(G, init_r_dist): # intitialize rationality value for each node and return dict with node and rationalty of that node

    degree_dict = dict(G.degree())

    def rationality_func(value, init_r_dist): # calculate rationaly value using degree of the node
        if init_r_dist == "pow(degree, 3)":
            network_rationality = NETWORK_RATIONALITY
            result =  math.exp(network_rationality * value * value * value)

        elif init_r_dist == "pow(degree, 2)":
            network_rationality = NETWORK_RATIONALITY
            result =  math.exp(network_rationality * value * value)

        elif init_r_dist == "pow(degree, 1)":
            network_rationality = NETWORK_RATIONALITY
            result =  math.exp(network_rationality * value)

        elif init_r_dist == "uniform":
            result = 1

        elif init_r_dist == "linear":
            network_rationality = NETWORK_RATIONALITY
            result = network_rationality * value 

        elif init_r_dist == "squared":
            network_rationality = NETWORK_RATIONALITY
            result = network_rationality * value * value

        elif init_r_dist == "cubed":
            network_rationality = NETWORK_RATIONALITY
            result = network_rationality * value * value * value


        else:
            print("r_dist not valid!")

        return result

    result_dict = {}

    for key, value in degree_dict.items():
        result_dict[key] = rationality_func(value, init_r_dist)

    return result_dict

def init_learning_rate(G,N, lr_dist): # initalize the learning rate for each node

    keys = list(range(N))
    if lr_dist == "degree_log":
        lr_dict = dict((key, np.log(G.degree[key] + 1) / np.log(1000 + 1)) for key in keys)

    elif lr_dist == "degree_linear_pos":
        lr_dict = dict((key, (G.degree[key] + 1) / (1000 + 1)) for key in keys)

    elif lr_dist == "degree_linear_neg":
        lr_dict = dict((key, (G.degree[key] + 1) / (1000 + 1)) for key in keys)
        lr_dict = reassign_lr_dict(lr_dict)

    elif lr_dist == "degree_convex":
        lr_dict = dict((key,pow((G.degree[key] + 1) / (1000 + 1),2)) for key in keys)

    elif lr_dist == "degree_exponential-3":
        lr_dict = dict((key, pow(3, G.degree[key]) / pow(3, 1000)) for key in keys)

    elif lr_dist == "normal":
        mu, sigma = 0.5, 0.8
        lr_dict = dict((key, np.random.normal(mu, sigma)) for key in keys)
        for key, value in lr_dict.items():
            if value > 1:
                lr_dict[key] = 1
            elif value < 0:
                lr_dict[key] = 0

    elif lr_dist == "log_normal":
        mu, sigma = 0.5, 0.4
        lr_dict = dict((key, np.random.lognormal(mu, sigma)-1) for key in keys)
        for key, value in lr_dict.items():
            if value > 1:
                lr_dict[key] = 1
            elif value < 0:
                lr_dict[key] = 0

    elif lr_dist == "uniform-dist":
        lr_dict = dict((key, np.random.uniform(low=0.0, high=1.0, size=None)) for key in keys)

    elif lr_dist == "uniform-0.1":
        lr_dict = dict((key, 0.1) for key in keys)

    elif lr_dist == "uniform-0.3":
        lr_dict = dict((key, 0.3) for key in keys)

    elif lr_dist == "uniform-0.5":
        lr_dict = dict((key, 0.5) for key in keys)

    elif lr_dist == "uniform-0.7":
        lr_dict = dict((key, 0.7) for key in keys)

    elif lr_dist == "uniform-0.9":
        lr_dict = dict((key, 0.9) for key in keys)

    else:
        print("lr_dist is not valid")

    return lr_dict

def update_learning_rate(G, N, lr_dist):

    keys = list(range(N))

    if lr_dist == "degree_log":
        lr_dict = dict((key, np.log(G.degree[key] + 1) / np.log(1000 + 1)) for key in keys)

    elif lr_dist == "degree_linear_pos":
        lr_dict = dict((key, (G.degree[key] + 1) / (1000 + 1)) for key in keys)

    elif lr_dist == "degree_linear_neg":
        lr_dict = dict((key, (G.degree[key] + 1) / (1000 + 1)) for key in keys)
        lr_dict = reassign_lr_dict(lr_dict)

    elif lr_dist == "degree_convex":
        lr_dict = dict((key,pow((G.degree[key] + 1) / (1000 + 1),2)) for key in keys)

    elif lr_dist == "degree_exponential-3":
        lr_dict = dict((key, pow(3, G.degree[key]) / pow(3, 1000)) for key in keys)

    elif lr_dist == "normal":
        mu, sigma = 0.5, 0.8
        lr_dict = dict((key, np.random.normal(mu, sigma)) for key in keys)
        for key, value in lr_dict.items():
            if value > 1:
                lr_dict[key] = 1
            elif value < 0:
                lr_dict[key] = 0

    elif lr_dist == "log_normal":
        mu, sigma = 0.5, 0.4
        lr_dict = dict((key, np.random.lognormal(mu, sigma)-1) for key in keys)
        for key, value in lr_dict.items():
            if value > 1:
                lr_dict[key] = 1
            elif value < 0:
                lr_dict[key] = 0

    elif lr_dist == "uniform-dist":
        lr_dict = dict((key, np.random.uniform(low=0.0, high=1.0, size=None)) for key in keys)


    elif lr_dist == "uniform-0.1":
        lr_dict = dict((key, 0.1) for key in keys)

    elif lr_dist == "uniform-0.3":
        lr_dict = dict((key, 0.3) for key in keys)

    elif lr_dist == "uniform-0.5":
        lr_dict = dict((key, 0.5) for key in keys)

    elif lr_dist == "uniform-0.7":
        lr_dict = dict((key, 0.7) for key in keys)

    elif lr_dist == "uniform-0.9":
        lr_dict = dict((key, 0.9) for key in keys)

    else:
        print("lr_dist is not valid")

    return lr_dict


def update_node_rationality_with_node_increase(G,node,R,LR):

    network_rationality = NETWORK_RATIONALITY
    Rest = math.exp(network_rationality * (G.degree[node] + 1) * (G.degree[node] + 1) * (G.degree[node] + 1))
    rational = (1 - LR[node])*R + LR[node]*Rest

    return rational

def update_node_rationality_with_node_decrease(G,node,R,LR):

    network_rationality = NETWORK_RATIONALITY
    Rest = math.exp(network_rationality * (G.degree[node] - 1) * (G.degree[node] - 1) * (G.degree[node] - 1))
    rational = (1 - LR[node])*R + LR[node]*Rest

    return rational

def update_rationility(G,N,R,LR, rest_dist): # update rationality for each node  using learning rate

    network_rationality = NETWORK_RATIONALITY

    for i in range(N):

        try:

            if rest_dist =="pow(degree, 3)":
                Rest = math.exp(network_rationality * G.degree[i] * G.degree[i] * G.degree[i])

            elif rest_dist =="pow(degree, 2)":
                Rest = math.exp(network_rationality * G.degree[i] * G.degree[i])

            elif rest_dist =="pow(degree, 1)":
                Rest = math.exp(network_rationality * G.degree[i])

            elif rest_dist == "linear":
                #network_rationality = 0.00001
                Rest = network_rationality * G.degree[i] 

            elif  rest_dist == "squared":
                Rest = network_rationality * G.degree[i] * G.degree[i]
            

            elif rest_dist == "cubed":
                Rest = network_rationality * G.degree[i] * G.degree[i] * G.degree[i]

            else:
                print("rest_dist unvalid")

        except:
            Rest = 0


        if R[i] != Rest:

            R[i] = (1 - LR[i])*R[i] + LR[i]*Rest # rationality update function

    return R

def write_rationality(data,csv_file): # write rationality of each node to a csv file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for key, value in data.items():
            writer.writerow((key,value))

    file.close()

def write_learning_rate(data,csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer = csv.writer(file)
        for key, value in data.items():
            writer.writerow((key,value))

    file.close()

def cpt_degree_assort(G):

    degree_assort = nx.degree_assortativity_coefficient(G)

    return degree_assort

def cpt_lr_assort(G,LR):      ## LR is the dictionary for learning rate
    nx.set_node_attributes(G,LR,"learning rate")
    lr_assort = nx.numeric_assortativity_coefficient(G, attribute = 'learning rate')

    return lr_assort

def cpt_r_assort(G, R):

    nx.set_node_attributes(G,R,"Rationality")
    r_assort = nx.numeric_assortativity_coefficient(G, attribute = 'Rationality')

    return r_assort

def recording_degree_assort(degree_assort_ls,experiment, route = ''):
    with open(f'{route}degree_assortativity-{experiment}.csv', "w", newline = "") as file:
        writer = csv.writer(file)
        for i in degree_assort_ls:
            writer.writerow([i])

def recording_lr_assort(lr_assort, experiment, route = ''):
    with open(f'{route}learning_rate_assortativity-{experiment}.csv', "w", newline = "") as file:
        writer = csv.writer(file)
        for i in lr_assort:
            writer.writerow([i])

def recording_r_assort(r_assort, experiment, route = ''):
    with open(f'{route}rationality_assortativity-{experiment}.csv', "w", newline = "") as file:
        writer = csv.writer(file)
        for i in r_assort:
            writer.writerow([i])

def recording_system_r(system_r_ls, experiment, route = ''):
    with open(f'{route}system_rationality-{experiment}.csv', "w", newline = "") as file:
        writer = csv.writer(file)
        for i in system_r_ls:
            writer.writerow([i])

def getEqm(R1,R2): #eqm calculation method

    '''
     #for positive correlation############
    try:
        iRationality = math.exp(0.001*iDeg)#exponential function where degree of the node is used
    except OverflowError:
        iRationality =float('inf')
    try:
        jRationality = math.exp(0.001*jDeg)#exponential function where degree of the node is used
    except OverflowError:
        jRationality =float('inf')
     ##################################

    '''

    '''
    #for negative correlation############
    try:
        iRationality = math.exp(1/(iDeg+1))#exponential function where degree of the node is used
    except OverflowError:
        iRationality =float('inf')
    try:
        jRationality = math.exp(1/(jDeg+1))#exponential function where degree of the node is used
    except OverflowError:
        jRationality =float('inf')
    ##################################
    '''
    '''
    #for random correlation############
    try:
      iRationality = math.exp(0.001*random.randint(1, 1000))#exponential function where degree of the node is used
    except OverflowError:
      iRationality =float('inf')
    try:
      jRationality = math.exp(0.001*random.randint(1, 1000))#exponential function where degree of the node is used
    except OverflowError:
      jRationality =float('inf')
    ##################################
    '''
    #network_rationality = 0.1

    #try:
        #iRationality = network_rationality*(iDeg)**2
    #except OverflowError:
        #iRationality =float('inf')
    #try:
        #jRationality = network_rationality*(jDeg)**2
    #except OverflowError:
        #jRationality =float('inf')


    L1=R1
    L2=R2
    try:
        eq1,eq2 =qre(L1,L2)#Quantal response equillibria: See up for the method
    except OverflowError:
        eq1=float('inf')
        eq2=float('inf')
    return eq1, eq2 #return calculated values

def KL_divergence(P,Q): # calculate KL divergence using P nad Q probability distribution
    if len(P) == len(Q):
        div = 0
        for i in range(len(P)):
            div = div +  math.log(P[i]/Q[i])*P[i]
    else:
        print('length of the P and Q should be same')
    return div

def JS_divergence(P,Q): # calculate JS divergence using P nad Q probability distribution
    if len(P) == len(Q):
        div = 0
        for i in range(len(P)):
            M = (P[i] + Q[i])/2
            div = div +  (math.log(P[i]/M)*P[i] + math.log(Q[i]/M)*Q[i])/2
    else:
        print('length of the P and Q should be same')
    return div

'''
def calcCrossEntropy(p1,p2,p3,p4):#entropy calculation
    try:
        entropy = math.log(p1/p3)*p1 + math.log(p2/p4)*p2;
    except :
        entropy=float('inf')
    return entropy

'''

def qre(L1,L2):#Quantal response equllibria
    beta=1.67;
    u111=1;
    u112=0;
    u121=beta;
    u122=0;
    u211=1;
    u212=beta;
    u221=0;
    u222=0;
    def equations(p):
        p1, p2 = p
        return (p1 - (math.exp(L1*(p2*u111 + (1-p2)*u112))/(math.exp(L1*(p2*u111 + (1-p2)*u112)) + math.exp(L1*(p2*u121 + (1-p2)*u122)))),
                p2 - (math.exp(L2*(p1*u211 + (1-p1)*u221))/(math.exp(L2*(p1*u211 + (1-p1)*u221)) + math.exp(L2*(p1*u212 + (1-p1)*u222)))))
    initial_guess = (0.5, 0.5)
    p1, p2 =  fsolve(equations, initial_guess)
    return p1,p2

def write_linklist(w,output): #write the link list into a csv file
    csvData=[]
    N=len(w)
    for i in range(N):
        [p,q,r]=w[i]
        csvData.append([p,q,r])
    with open(output,'w',newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()


def calculate_div(R1,R2):

    eqm1, eqm2 = getEqm(R1,R2)#get eqm1 and eqm2 for entropy calculation : see below for method
    P = [1,1] # probability distrbution P
    Q = [1 - eqm1, 1 - eqm2] # probability distrbution  Q
    div =  JS_divergence(P,Q) # calculate divergence of the iNode and jNode

    return div

def graph_dev(G, R, all_links):
    E = len(all_links)
    total_div = 0
    for idx in range(E):
        [i,j,link] = all_links[idx]
        R1 = R[i]
        R2 = R[j]
        link_div = calculate_div(R1,R2)
        total_div += link_div

    return total_div

def net_sim(k,G1,N,E, method,all_links,R,LR, r_assort_ls, lr_assort_ls, d_assort_ls, sys_r_ls, rest_dist, lr_dist, route):# network simulation
    # k : kth run
    # G1 : fully connected graph
    # N : number of nodes
    # E : number of edges
    # all_links : link list
    # R : dictionary of Rationality
    # LR : dictionary of Learning Rate

    if method == "Net_L_F_0.5":
        F = 0.5
    elif method == "Net_L_F_0.2":
        F = 0.2
    elif method == "Net_L_F_0.1":
        F = 0.1
    elif method == "Net_L_F_1":
        F = 1

    total_rewire = int(iterations*E/300)
    test_rewire_num = 1/F           # the interval num of rewire for testing system_r
    record_rewire_num = 10          # the interval num of rewiring for recording

    record = False
    testing_iter = 0
    recording_iter = 0
    j = 0
    before_j = 0
    before_all_links = copy.deepcopy(all_links)
    before_R = copy.deepcopy(R)
    before_LR = copy.deepcopy(LR)
    before_G = set_graph(before_all_links)
    temp_all_links = copy.deepcopy(all_links)
    temp_R = copy.deepcopy(R)
    temp_LR = copy.deepcopy(LR)
    temp_G = set_graph(temp_all_links)
    prev_div = graph_dev(before_G, before_R, before_all_links)

    recording_dic = {}
    r_assort_ls = []
    lr_assort_ls = []
    d_assort_ls = []
    sys_r_ls = []
    recording_r_assort_ls = []
    recording_lr_assort_ls = []
    recording_d_assort_ls = []
    recording_sys_r_ls = []

    with tqdm(total=int(total_rewire/record_rewire_num)) as pbar:
        j = 0
        while j < total_rewire:
        # for j in tqdm(range(total_rewire)):

            # Try to get a rewire that will not break the network
            link_list = set()
            EXIST = True
            trying_all_links = copy.deepcopy(temp_all_links)
            trying_G = set_graph(trying_all_links)

            while EXIST:
                link = random.randint(0,E-1) #choose a random link out of the E edges

                if link in link_list:
                    EXIST = True
                else:
                    link_list.add(link)
                    EXIST = False
            [inode,jnode,link]=trying_all_links[link]#get the nodes from the link

            jDeg = trying_G.degree[jnode]  #get the degree of the second node

            while jDeg == 1: # this suitable link selection process go untill jDeg == 1 otherwise it is not generate fully connected graph
                EXIST = True

                while EXIST:
                    link = random.randint(0,E-1)

                    if link in link_list:
                        EXIST = True
                    else:
                        link_list.add(link)
                        EXIST = False
                [inode,jnode,link]=trying_all_links[link]#get the nodes from the link
                jDeg = trying_G.degree[jnode]  #get the degree of the second node

            jnodeNew= random.randint(0,N-1)  #pick a random node
            while((jnodeNew==inode) or (jnodeNew==jnode) or check_neighbour(trying_all_links,inode,jnodeNew)):#clarify that it is not in the neighbourhood or in the given link(self node) or all-ready connected
                jnodeNew= random.randint(0,N-1)       #if yes pick another

            trying_all_links[link][1] = jnodeNew
            trying_G = set_graph(trying_all_links)
            if N != len(trying_G.nodes):

                j=0
                trying_all_links = copy.deepcopy(temp_all_links)
                link_list = set()
                continue

            temp_all_links = copy.deepcopy(trying_all_links)
            temp_G = set_graph(temp_all_links)
            temp_R = update_rationility(temp_G,N,temp_R,temp_LR,rest_dist)
            if lr_dist != "normal" and lr_dist != "log_normal" and lr_dist != "uniform-dist":
                temp_LR = update_learning_rate(temp_G,N,lr_dist)



            # testing system rationality
            if j // test_rewire_num > testing_iter:
                new_div = graph_dev(temp_G, temp_R, temp_all_links)
                diff_dev = new_div - prev_div

                # if the system rationality test passes
                if diff_dev <= 0:
                    before_all_links = copy.deepcopy(temp_all_links)
                    before_G = set_graph(before_all_links)
                    before_R = copy.deepcopy(temp_R)
                    before_LR = copy.deepcopy(temp_LR)
                    before_j = j
                    prev_div = graph_dev(before_G, before_R, before_all_links)

                    testing_iter += 1
                    record = True

                # if it does not pass
                else:
                    temp_all_links = copy.deepcopy(before_all_links)
                    temp_G = set_graph(temp_all_links)
                    temp_R = copy.deepcopy(before_R)
                    temp_LR = copy.deepcopy(before_LR)
                    j = before_j

                    # delete the recorded networks
                    recording_iter = j // record_rewire_num
                    recording_dic = {}
                    recording_r_assort_ls = []
                    recording_lr_assort_ls = []
                    recording_d_assort_ls = []
                    recording_sys_r_ls = []

                    continue

            # recording turn
            temp_recording_iter = j // record_rewire_num
            if temp_recording_iter> recording_iter:
                recording_dic[temp_recording_iter] = [copy.deepcopy(temp_R), copy.deepcopy(temp_LR), copy.deepcopy(temp_all_links)]

                recording_r_assort_ls.append(cpt_r_assort(temp_G, temp_R))
                recording_lr_assort_ls.append(cpt_lr_assort(temp_G, temp_LR))
                recording_d_assort_ls.append(cpt_degree_assort(temp_G))
                recording_sys_r_ls.append(-new_div)

                recording_iter = temp_recording_iter

                if record:
                    for key,value in recording_dic.items():
                        [R, LR, all_links] = value
                        write_rationality(R, route+'node_rationalities-'+ str(k) + '-' + str(key) + '.csv')
                        write_linklist(all_links,route+'topology-'+ str(k) + '-' + str(key) + '.csv')# write link list to csv file for later analysis
                        write_learning_rate(LR, route+'learning_rate-'+ str(k) + '-' + str(key) + '.csv')

                    r_assort_ls.extend(recording_r_assort_ls)
                    lr_assort_ls.extend(recording_lr_assort_ls)
                    d_assort_ls.extend(recording_d_assort_ls)
                    sys_r_ls.extend(recording_sys_r_ls)
                    all_links = temp_all_links
                    pbar.update(1)
                    record = False

            j += 1


    return all_links, r_assort_ls, lr_assort_ls, d_assort_ls, sys_r_ls



#def running_experiment(R_dist, LR_dist, ER_dist, runs, N, E, method):
#    print('SIMULATION STARTED WITH NODES: '+ str(N) +'  EDGES: '+ str(E))
#    print('============================================ \n')
#    #loop for networks
#    for k in range(runs):
#        r_assort_ls, lr_assort_ls, d_assort_ls, system_r_ls = [], [], [], []
#        G=create(N,E)#create a random network with N nodes and E edges : method can be found in net_lib file
#        R=init_rationility(G,R_dist) #initiate Rationality for each node
#        write_rationality(R, route+'node_rationalities-'+ str(k) + '-0.csv')
#        LR = init_learning_rate(G,N,LR_dist) #initiate Learning Rate for each node
#        write_learning_rate(LR, route+'learning_rate-'+ str(k) + '-0.csv')
#        all_links=get_links(N,G)#get the linklist of the network
#        write_linklist(all_links,route+'topology-'+ str(k) + '-0.csv') # write link list to csv file for later analysis
#        all_links, r_assort_ls, lr_assort_ls, d_assort_ls, system_r_ls = net_sim(k,G,N,E,method,all_links,R, LR, r_assort_ls, lr_assort_ls, d_assort_ls, system_r_ls,ER_dist,LR_dist)#initialise the simulated annealing: code is in net_lib
#
#        recording_degree_assort(d_assort_ls,k, route)
#        recording_lr_assort(lr_assort_ls, k, route)
#        recording_r_assort(r_assort_ls, k, route)
#        recording_system_r(system_r_ls, k, route)
#
#        #print(k+1)# print the iteration number
#    print('SIMULATION ENDED WITH NODES: ')


def run_single_experiment(k, N, E, R_dist, LR_dist, ER_dist, method, route, runs):
    """
    This function encapsulates the logic for a single simulation run.
    It's designed to be called by the multiprocessing pool.
    
    Parameters:
    args (tuple): A tuple containing all the necessary arguments.
                  This is required for use with pool.starmap.
    """
    try:

        print(f"Starting run {k+1}/{runs}...")

        # --- The logic from your original for-loop ---
        r_assort_ls, lr_assort_ls, d_assort_ls, system_r_ls = [], [], [], []
    
        G = create(N, E) # create a random network
    
        R = init_rationility(G, R_dist) # initiate Rationality
        write_rationality(R, os.path.join(route, f'node_rationalities-{k}-0.csv'))
    
        LR = init_learning_rate(G, N, LR_dist) # initiate Learning Rate
        write_learning_rate(LR, os.path.join(route, f'learning_rate-{k}-0.csv'))
    
        all_links = get_links(N, G) # get the linklist
        write_linklist(all_links, os.path.join(route, f'topology-{k}-0.csv'))
    
        # Run the core simulation
        all_links, r_assort_ls, lr_assort_ls, d_assort_ls, system_r_ls = net_sim(
            k, G, N, E, method, all_links, R, LR, r_assort_ls, 
            lr_assort_ls, d_assort_ls, system_r_ls, ER_dist, LR_dist, route
        )

        # Record the results for this run
        recording_degree_assort(d_assort_ls, k, route)
        recording_lr_assort(lr_assort_ls, k, route)
        recording_r_assort(r_assort_ls, k, route)
        recording_system_r(system_r_ls, k, route)
    
        print(f"Finished run {k+1}/{runs}.")
    
        # This function doesn't need to return anything because it writes files directly.
        # If you needed to aggregate results in memory, you would return them here.
        return f"Run {k} completed successfully."
    
    except Exception as e:
        # If any error occurs in the 'try' block, this code will run
        print(f"!!!!!!!! An error occurred in worker k={k} !!!!!!!!")
        # traceback.format_exc() provides a detailed error message
        error_details = traceback.format_exc()
        print(error_details)
        return f"Run {k} FAILED with error: {e}"



def running_experiment_parallel(R_dist, LR_dist, ER_dist, runs, N, E, method, route):
    """
    This function sets up a multiprocessing pool to run all simulations in parallel.
    """
    print(f'SIMULATION STARTED WITH NODES: {N} EDGES: {E}')
    print('============================================\n')
    
    # Prepare the arguments for each run.
    # Each element in this list is a tuple of arguments for one call to run_single_experiment.
    task_args = [(k, N, E, R_dist, LR_dist, ER_dist, method, route, runs) for k in range(runs)]
    
    # Determine the number of processes to use.
    # Using cpu_count() is a good default. You can also set it to a specific number.
    num_processes = mp.cpu_count()
    num_processes = min(num_processes, runs)  # Ensure we don't create more processes than runs
    print(f"Creating a pool of {num_processes} worker processes.")

    # Create a pool of worker processes.
    # The 'with' statement ensures the pool is properly closed.
    with mp.Pool(processes=num_processes) as pool:
        # Use starmap to apply the worker function to the list of arguments.
        # starmap is like map, but it unpacks argument tuples for the worker function.
        # This will block until all processes are complete.
        results = pool.starmap(run_single_experiment, task_args)
    
    print('\n============================================')
    print('ALL SIMULATION RUNS HAVE COMPLETED.')
    # You can optionally print the results returned by each worker
    # for result in results:
    #     print(result)

if __name__ == '__main__':
   
    N=1000 #number of nodes
    E=3000 #number of edges
    runs=1 #number of iterations
    route = "squared_01/" ## Data Storage Position
    init_R_dist = "squared"    # pow(degree, 3), pow(degree, 2), pow(degree, 1),
    ER_dist = "squared"   # pow(degree, 3), pow(degree, 2), pow(degree, 1)

    # Learning Rate Distribution
    LR_dist = "uniform-0.1"           # "normal"   # uniform-0.1, uniform-0.3, uniform-0.5, uniform-0.7, uniform-0.9
                # If you are using normal, log-normal or uniform-dist, make sure to
                # adjust the code in both function "init_learning_rate" and "update_learning_rate"

    # Rewire Update Rate
    rewire_method = "Net_L_F_1" # Net_L_F_1, Net_L_F_0.5, Net_L_F_0.2, Net_L_F_0.1

    running_experiment_parallel(init_R_dist, LR_dist, ER_dist, runs, N, E, rewire_method, route)