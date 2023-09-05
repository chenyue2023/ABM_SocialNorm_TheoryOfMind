import numpy as np
import pandas as pd
# KL divergence
from scipy.special import rel_entr
# emd
from scipy.stats import wasserstein_distance
# JS divergence
from scipy.spatial.distance import jensenshannon


######## collective level discrepancy
def preceived_c_norm(current_d, k):
    '''
    Similar to how current behaviour norm is computed out of all actions
    this function aggregate all preceived dominant norms (in index) and 
    convert them into a weight-like norm

        Parameters:
               current_d (numpy array): stands for current dominant categories, it is a numpy array that contains INDEXes of each agent's believed dominant category.
               
                 For example: agent 1 believe 0.5 is the most dominant category at timestep 0, 
                 the numpy array where have 2 which is the index for 0.5 in the list of category mus
                 as the 1st item in current_d

        Returns:
               bin_weights (numpy array): it returns a weight in form of 5-item numpy array as current dominant norm
    '''
    
    norm_mus = np.arange(0, k, 1)

    # Define bin edges and calculate bin centers
    bin_edges = np.arange(0.5, k, 1)
    bin_centers = norm_mus
    # Use `digitize` function to group data into bins
    bin_indices = np.digitize(current_d, bin_edges)

    # Use `bincount` function to obtain the weights for each bin
    bin_weights = np.divide(np.bincount(bin_indices, minlength=len(bin_centers)), len(current_d))
    return bin_weights

def dominant_one(agents, steps, N, k):
    '''
    aggregate the most dominant perceived norm by All Agents for all Timestep
    it returns collective subjective beliefs
    '''
    # global mus_str 
    dominant_container = np.empty(shape = (N, steps))
    dominant_ones = np.empty(shape = (steps, k))
    for n in range(N):
        agent_dominant = np.argmax(agents[n].subject_weights, axis = 1)
        dominant_container[n] = agent_dominant
    dominant_container = np.transpose(dominant_container)

    for p in range(steps):
        dominant_ones[p] = preceived_c_norm(dominant_container[p], k)
    
    return dominant_ones

def dominant_confidence(agents, N, step):
    '''
    aggregate the weight of the most dominant perceived norm by All Agents at the perticular Timestep
    to check how confident the collectives is with their self-believed norm
    '''
    dominant_container = []
    for n in range(N):
        agent_dominant = np.argmax(agents[n].subject_weights[step])
        dominant_container.append(agents[n].subject_weights[step][agent_dominant])
    
    dominant_container = np.asarray(dominant_container)
    dominant_container = dominant_container.round(3)
    return dominant_container

def ignorance(collective_norm, initial_norm, steps):
  '''
  comepare collective level norm with initial norm in KL divergence
  collective_norm:
  1. collective objective behaviour aggregated by All action per Timestep
  2. collective subjective belief aggregated by Each agent's believed most dominant norm per Timestep
  initial_norm: the collective objective belief

  Use collective norm to approximate initial norm, therefore: P = collective norm
  Q = initial norm
  I do not expect the divergence between collective and intial norm is symmetrical, since collective change dynamically along the timestep
  Therefore, I use asymmetric KLdivergence
  '''
  norm_pl = []
  for i in range(steps):
    p = collective_norm[i]
    q = initial_norm # one and only
    # add 1e-10 as Laplace smoothing 
    kl_pq = rel_entr(p+ 1e-10, q+ 1e-10)
    norm_pl = np.append(norm_pl, sum(kl_pq))
  
  norm_pl = pd.DataFrame({'KLdiv':norm_pl})
  norm_pl['50_rolling_avg'] = norm_pl.KLdiv.rolling(50).mean()
  return norm_pl


def misperception_colcol(current_norm, current_beliefs, steps):
    '''
    compare current norm in terms of collective action with current beliefs
    aka the top1 perceived dominant norm by All Agents per Timestep 
    using dominant_one()
    '''
    mc_kl = []
    for i in range(steps):
      p = current_norm[i]
      q = current_beliefs[i]
      kl_pq = rel_entr(p+ 1e-10, q+ 1e-10)
      mc_kl = np.append(mc_kl, sum(kl_pq))
    
    mc_kl = pd.DataFrame({'KLdiv':mc_kl})
    mc_kl['50_rolling_avg'] = mc_kl.KLdiv.rolling(50).mean()
    return mc_kl


def misperception_colcol_js(current_norm, current_beliefs, steps):
  '''
  Note: the pandas column is still called KLdiv due to plotting reason 
        remain the column name fixed, we can loop throught everything easily.

  The reason why I employ JS divergence is that: I expect symmetry when 2 things are at the same scale
  The reason why I abandon EMD is that it is not quantifying information loss along the way, 
  EMD measures dissimilarity based on transportation cost and spatial arrangement, while KL divergence measures dissimilarity based on information content
  To make all measures 'meaningly' consistent, I go with JS divergence. Since all I want is symmetric between 2 distributions
  Therefore, ...colcol_js() can replace ..._colcol() in measure the discrepancy between 2 collective level constructs
  '''
  norm_pl = []
  for i in range(steps):
    p = current_beliefs[i]
    q = current_norm[i]
    js_pq = jensenshannon(p, q)
    norm_pl = np.append(norm_pl, js_pq)
  
  norm_pl= pd.DataFrame({'KLdiv':norm_pl})
  norm_pl['50_rolling_avg'] = norm_pl.KLdiv.rolling(50).mean()
  return norm_pl


######## inividual to collective discrepancy
def misperception_initial(subject_norm, initial_norm, steps):
  '''
  compare subjective norm perception per Agent per Timestep 
  with initial norm in KL divergence

  subjective_norm: ONE agent's subjective norm belief
  initial_norm: the collective objective belief

  I use subjecive norm perception to approximate intial norm
  '''
  mi_pl = []
  for i in range(steps): 
    p = subject_norm[i] 
    q = initial_norm # one and only
    kl_pq = rel_entr(p+ 1e-10, q+ 1e-10)
    mi_pl = np.append(mi_pl, sum(kl_pq))

  return mi_pl

def misperception_current(subject_norm, collective_norm, steps):
    '''
    compare subjective norm perception per Agent per Timestep
    with a collective level norm per Timestep in KL divergence
    
    subjective_norm: ONE agent's subjective norm belief

	collective_norm:
	1. collective objective behaviour aggregated by All action per Timestep
	2. collective subjective belief aggregated by Each agent's believed most dominant norm per Timestep

    Iuse subjective norm perception to approximate collective norm
    '''
    mc_kl = []
    for i in range(steps):
      p = subject_norm[i]
      q = collective_norm[i]
      kl_pq = rel_entr(p+ 1e-10, q+ 1e-10)
      mc_kl = np.append(mc_kl, sum(kl_pq))
    
    return mc_kl

def ts_initial_MP(N, steps, agents, IN):
    '''
    discrepancy between initial norm with subjective norm perception 
    in KL divergence per Agent per Timestep for All Agents

    it calls misperception_initial() function that compute the discrepancy along the time for ONE agent
    therefore, the whole function compute misperception to initial norm for ALL agents
    '''
    initial_MP = np.empty(shape = (N, steps))
    for i in range(N):
      SN = agents[i].subject_weights
      initial_MP[i] = misperception_initial(SN, IN, steps)
    return initial_MP

def ts_current_MP(N, steps, agents, CN):
    '''
    discrepancy between collective level norm with subjective norm perception
    in KL divergence per Agent per Timestep for All Agents

    it calls misperception_current() function that compute the discrepancy along the time for ONE agent
    between a collective behaviour or belief norm vs inidividual subjective belief for ONE Agent
    therefore, the whole function compute misperception to a collective norm for ALL agents
    '''
    current_MP = np.empty(shape = (N, steps))
    for i in range(N):
        SN = agents[i].subject_weights
        current_MP[i] = misperception_current(SN, CN, steps)
    return current_MP







