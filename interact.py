# network structure
import networkx as nx
# count time passing
import time
# efficient looping
import itertools
# count repeated inferred categories
from collections import Counter
# speed up processing
import multiprocessing as mp
# import agent classes
from agent import *
import pandas as pd


################################ Interaction #############################################

def generate_agents(N, agent_ratio, bias, steps, equalNorm):
    '''
    Generate agents based on different ToM agents ratio in the population, shared conformity bias by all agents, total timesteps
    Assign each agents an unique mu, their drinking group, initial subjective norm weights ... by calling ToM_0 and ToM_1 classes defined in agent.py

        Parameters:
               N (int): total number of agents (multiplier of 10)
               agent_ratio (float): ratio of Complex Inference agents, 0 stands for all Simple Inference agent, 1 stands for all Complex Inference agents, 
                                    0.5 means half ToM0 half ToM1 agents
               bias (float): confirmity bias shared by all agents which determined how agent finds its action production "middle ground"
               steps (int): total timesteps

        Returns:
               agents (list): All agents created in a list
               initial_norm (numpy array): initial proportion of different drinking group in the population
    '''
    # call global shared variables shared by all agents for all simulations
    global k 
    global norm_mus 
    global norm_sd 
    
    # all agents start with equal weights 1/k over all components
    # init_w = np.ones_like(np.arange(0.1, 1.1, 0.1)) / 10 --> BIG BUG shared memory location
    # learning rate: how the inferenced component changes its weight
    init_update = 0.01
    
    ###### Generate Initial Population and Groups
    if equalNorm == False:
        initial_pop = [0]*k
        # initiate agent population and norm
        for i in range(N):
            # randomly pick a drinking group
            random_idx = random.randint(0, (k-1))
            # adding one member to that group
            initial_pop[random_idx] += 1
    else:  
        # set equal group size for all drinking group
        initial_pop = [int(N/k)]*k
        
    # initial norm in weights (sum to 1)
    initial_norm = np.divide(initial_pop, N)

    ###### Generate Agents' Desire Means and Actual Groups
    # assigning unique desire mu to each agent
    # match agent to their true drinking group
    agent_mus = [np.random.normal(norm_mus[i], norm_sd[i], initial_pop[i]) for i in range(k)]
    agent_mus = np.concatenate(agent_mus, axis = None)
    agent_groups = [np.repeat(np.round(norm_mus[i], decimals = 1), initial_pop[i]) for i in range(k)]
    agent_groups = np.concatenate(agent_groups, axis = None)

    ###### Generate Sequence of ToM_0 and ToM_1 Agents by Ratio
    # decision making on agent_ratio = [0,1]
    # create number of 1s equal to agent_ratio*N, if agent_ratio = 0, all agents are ToM_0 agents
    # create number of 0s equal to (1-agent_ratio)*N
    num_ToM1 = np.ones(int(agent_ratio * N), dtype=int)
    num_ToM0 = np.zeros(int((1-agent_ratio)*N), dtype=int)
    agent_sequence = np.append(num_ToM1, num_ToM0)
    # randomly shuffle numpy array
    np.random.shuffle(agent_sequence) 
    
    ###### Generate Actual Agents 
    agents=[]
    for i in range(N): 
      if (agent_sequence[i] == 0):
        agents.append(ToM_0(agent_mus[i], agent_groups[i], np.ones(k)* (1/k), bias, init_update, steps))   
      else:
        agents.append(ToM_1(agent_mus[i], agent_groups[i], np.ones(k)* (1/k), bias, init_update, steps))  
      
    # randomly shuffle agents List
    random.shuffle(agents)
      
    return agents, initial_norm


def set_network(N, net_type):
    '''
    Create Network where all agents interact on

          Parameters:
                 N (int): total number of agents
                 net_type (int): either 0, 1 or 2

          Returns:
                 networkx Graph object
      '''
    # net_type = [0,1,2]
    # 1 - full-connected
    if net_type == 0:
      return nx.complete_graph(N)
    # 2 - scale-free directed graph
    elif net_type == 1:
      return nx.barabasi_albert_graph(n = N, m = 2, seed = None, initial_graph = None) 
    # 3 - small world
    else:
      if (N == 100):
        return nx.watts_strogatz_graph(N, 20, .1, seed=None)
      elif (N == 50):
        return nx.watts_strogatz_graph(N, 10, .1, seed=None)



def actual_norm(current_actions):
    '''
    Aggregate current actions produced by all agents at one timestep into bins 
    return bin_weights as current norm (collective + objective + behaviour)

        Parameters:
               current_actions (numpy array): all actions produced by all agents at one timestep

        Returns:
               bin_weights (numpy array): weights for each drinking category 
    '''
    global k 
    global norm_mus 
    global norm_sd 

    current_actions = np.where(current_actions > 1, 1, current_actions)
    current_actions = np.where(current_actions < 0, 0, current_actions)

    # Define bin edges and calculate bin centers
    bin_edges = norm_mus + norm_sd
    bin_centers = norm_mus
    # Use `digitize` function to group data into bins
    bin_indices = np.digitize(current_actions, bin_edges)

    # Use `bincount` function to obtain the weights for each bin
    bin_weights = np.divide(np.bincount(bin_indices, minlength=len(bin_centers)), k)
    bin_weights = bin_weights/np.sum(bin_weights)
    return bin_weights


def sender_receiver(agents, network, current_agent_idx, step):
    '''
    Parallel processing each sender agent's action production 
    inference made by receiver agents on each action

        Parameters:
               agents (list): all agents in a list
               network: networkx Graph object
               current_agent_idx (int): sender agent's index 
               step (int): current step or current iteration

        Returns:
               current_agent_idx (int): sender agent's index
               agents[current_agent_idx].group: sender agent's drinking group mu
               current_action (float): action produced by sender agent
               current_receiver_idx (int): reciver agent's index (who made inference on current_action)
               current_infer (int): what current_receiver_idx inferred group based on current_action,
                                    index of the drinking group 
               step: current step or current iteration


    '''
    # produce action
    current_action = agents[current_agent_idx].action_production()
    # pick receiver
    current_receiver_idx = random.sample(list(network.neighbors(current_agent_idx)),1)[0]
    # receiver make inference
    current_infer = agents[current_receiver_idx].inference(current_action)
    return [current_agent_idx, agents[current_agent_idx].group, current_action, current_receiver_idx, current_infer, step]


def update_all(agents, stepOut, N):
    '''
    All receiver agents update their subjective social norm weights based on what they received and inferred from sender_receiver
    Record everything happened in current interaction: sender agent record its action

        Parameters:
               agents (list): all agents in a list
               stepOut (list): what was output by sender_receiver where one agent send signal another agent make inference
                               [sender index, sender group, action, receiver index, inferred group, step]
               N (int): total number of agents

        Returns:
               df (pandas DataFrame): return stepOut into DataFrame
               current_norm: Current Actual Action Norm at current timestep (Collective + Objective + Behaviour)
    '''
    # set timer1  
    # time1 = time.time()

    # convert parallel action-inference cycle to dataframe
    df = pd.DataFrame(stepOut)
    df.columns = ['sender', 'senderGroup', 'action', 'receiver', 'infer', 'step']

    ##### All agents record their action for N agents
    current_step = stepOut[0][5]
    for i in range(N):
      temp_agent = stepOut[i][0]
      temp_action = stepOut[i][2]
      agents[temp_agent].action_record(current_step, temp_action)

    ##### For current_norm record
    actions = df['action'].to_numpy()
    current_norm = actual_norm(actions)
    
    ##### aggregate receivers and their signals
    grouped_df = df.groupby('receiver')['infer'].agg(list)
    grouped_df = grouped_df.to_frame()
    # Apply Counter to the 'infer' column using apply() and Counter()
    counts = grouped_df['infer'].apply(Counter)

    # Modify the original column to be the dictionary keys returned by Counter
    grouped_df['infer'] = counts.apply(lambda c: list(c.keys()))
    grouped_df['counts'] = counts.apply(lambda c: list(c.values()))
    grouped_df = grouped_df.reset_index()
    grouped_df = grouped_df.values

    for i in range(np.shape(grouped_df)[0]):
      update_agent, update_cate, update_times = grouped_df[i]
      agents[update_agent].para_update(update_cate, update_times)
     
    # set timer2  
    # time2 = time.time()
    # print("Update took: ", (time2 - time1))
    return df, current_norm


def let_interact(agents, network, N, steps, IN, pool):
    '''
    Loop through all timesteps and record variables for future analysis 
    Every agent record their subjective norm belief updated after each timestep
    Record current actual norm (Collective + Objective + Behaviour)

        Parameters:
               agents (list): all agents in a list
               network: networkx Graph object
               N (int): total number of agents
               steps (int): total iteraction timesteps
               IN (numpy array): initial norm (Collective + Objective + Belief)
               pool: Python Multiprocessing Pool class 


        Returns:
               results (Dictionary): 
               'CN': current_Norms: current action norm for all timesteps
               'agents': agents: all agents who update and record its subjective norm belief updates and action produced
               'record': interact_record: sender index, receiver index, inferred drinking group, which step it all happened
               'IN': IN: initial norm
               'network': network: the network all agents interact on
    '''

    time1 = time.time()

    global k 

    # set empty current social norms
    current_Norms = np.zeros((steps, k))

    # set empty irecord pandas table
    # record every action-inference: action agent, action, inference agent, inference results, step
    # Define the column names
    columns = ['sender', 'senderGroup', 'action', 'receiver', 'infer', 'step']
    # Create an empty DataFrame with the specified columns and number of rows
    interact_record = pd.DataFrame(columns=columns)
    
    # actinon production & parallel inferences
    agent_args = list(np.arange(0, N, 1, dtype=int))

    for step in range(steps):
      timestep1 = time.time()
      # Map the input values to the function using the pool
      # call the same function with different data sequentially
      # parallel action-infer cycle
      # timeA = time.time()
      out = pool.starmap(sender_receiver, zip(itertools.repeat(agents), itertools.repeat(network), agent_args, itertools.repeat(step)))
      # timeB = time.time()
      # print('cycle done in', (timeB-timeA))
      # print(out)
      # out = para_step(agents, network, N, step)
      
      # update subjective norm weights 
      df, current_norm = update_all(agents, out, N)

      ##### recording
      # record current actual norm
      current_Norms[step] = current_norm
      # record interaction record
      interact_record = pd.concat([interact_record, df])
      # record indiviudal norms for every agent
      for agent in range(N):
        agents[agent].subject_record(step)
     
      timestep2 = time.time()
      if step in [0, 500, 1000, 2000, 3000, 4000, 5000]:
         print("This step took: ", (timestep2 - timestep1))
      
    ## out of the steps loop
    # current_Norms, agents, interact_record 
    # what to save?
    interact_record = interact_record.astype({'sender':'int', 'receiver':'int', 'infer':'int', 'step': 'int'})
    results = {'CN': current_Norms, 'agents': agents, 'record': interact_record, 'IN': IN, 'network': network}
    
    return results


