######## run simulation on servers
from agent import *
from interact import *
import multiprocessing as mp
import time
import os
import pickle

# path to save simulated data
# cwd = os.getcwd()
# print('current path',cwd)
cwd = '/scratch/yuechen'

# Set Initial Norm
# that means all drinking groups has same numbers of agents
q0 = input('Do you want equal Initial Norm? Y/N ')

# If Yes, euqalNorm == True
if q0 == 'Y':
  equalNorm = True
# otherwise False
else:
  equalNorm = False


# set simulation conditions
q1 = input('Enter ToM conditions Separate by SPACE: ')
ratio_range = np.array(list(map(int, q1.split())))

q2 = input('Enter Conformity Bias Conditions Separate by SPACE: ')
bias_range = np.array(list(map(float, q2.split())))

q3 = input('Enter Network Conditions Separate by SPACE: ')
net_range = np.array(list(map(int, q3.split())),  dtype=int)

q4 = input('Enter Start Run Index: ')
q5 = input('Enter End Run Index: ')
runs = np.arange(int(q4), int(q5), 1, dtype=int) 

ratio_range, bias_range, net_range, runs = np.meshgrid(ratio_range, bias_range, net_range, runs)
params = np.stack((ratio_range.ravel(), bias_range.ravel(), net_range.ravel(), runs.ravel()), axis = 1)



if __name__ == '__main__':
  # Define the number of processes to run in parallel
  num_processes = mp.cpu_count()
  print('cpu number', num_processes)
  q6 = input('Enter how many core you want to use on this server: ')
  num_processes = int(q6)

  # Create a multiprocessing pool
  pool = mp.Pool(processes = num_processes)
  
  time1 = time.time()

  # loop through all parameter settings
  for i in range(np.shape(params)[0]):
    agent_ratio, bias, net_type, run = params[i]
    # same number of agents and steps every run
    N = 50
    steps = 5000
    # generate agents
    agents, IN = generate_agents(N, agent_ratio, bias, steps, equalNorm)
    # set network
    network = set_network(N, net_type)
    
    results = let_interact(agents, network, N, steps,IN, pool)
      

    # create filename according to condition/parameter
    filename = f"ratio_{agent_ratio}_bias_{bias}_net_{net_type.astype(int)}_run_{run.astype(int)}.pkl"
    # set file path
    filepath = os.path.join(cwd, filename)
      
    # save the results to a file using pickle
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

    time2 = time.time()
    print("Finish all requires: ", (time2 - time1))
    print("########################################################################################")


  # Close the pool and wait for all processes to finish
  pool.close()
  pool.join()
