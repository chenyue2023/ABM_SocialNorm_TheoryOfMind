# libraries： random, numpy, pandas, scipy, warnings, searborn, matplotlib
import random
import numpy as np
import pandas as pd
from scipy import stats 
from scipy.stats import norm
from scipy.stats import mode

## ignore warnings
import warnings
warnings.filterwarnings("ignore")

## ploting
import seaborn as sns
import matplotlib.pyplot as plt


############################### Global Variable #############################################
"""
k: total number of drinking groups shared by all agents
norm_mus: each drinking group's means from 0.1 to 1.0 step by 0.2
norm_sd: the standard deviation shared by all drinking groups
desire_sd: the standard deviation shared by all agents in all drinking groups
"""

k = 5
norm_mus = np.arange(0.1, 1.0, 0.2)
norm_sd = np.ones(k)* 0.1
desire_sd = np.asarray(0.05)

##################################### Agent #################################################

class Agent:
  """
    A class to represent a Agent

    ...
    Class Variables:
    ----------
    k: int
        total number of drinking groups (categories)
    norms_mus: numpy array (a sequence)
        mus for each normal distributed drinking group
    norm_sd: numpy array (a sequence)
        standard deviation for each normal distributed drinking group
    desire_sd: numpy array (just a float number in numpy array)
        standard deviation for each agent's desire distribution

    Attributes
    ----------
    desire_mu: float
        mu for each agent's desire distribution
    group: float
        group for each agent, also the mu for that drinking group's normal distribution
    norm_weights: numpy array
        agent's subjective belief about the norm
    bias (conformity degree): float
        agent's conformity bias, probability of produced an action from one's subjective norm perception
    update_rate: float
        the amount agent added to its inferred drinking group to update its norm_weights
    nmus: numpy array
        combined new mus = desire mu + norm mus 
    nsigmas: numpy array
        combined sigmas (standard deviation) = desire_sd + norm_sd (for Gaussian Mixture)
    nweights: numpy array
        combined weights for action prodcution by combining subjective norm weights with (1-conformity bias) for desire
    subject_weights: numpy array
        recording subjective norm perception weight at every timestep per agent for future analysis
    all_actions: numpy array
        recording all actions produced by self for future analysis

    Methods
    -------
    action_production():
        produced an action from one's desire and subjective norm distribution
    print_desire():
        print agent's desire distribution for inspection and debug
    print_subject_norm():
        print agent's subejctive norm perception 
    print_action_production():
        print agent's action production distribution
    predict_category(estimate_mu):
        predict which drinking categroy an estiamted mu came from
    posterior(x, signal):
        posterior distribution when received an drinking signal
    inference(signal):
        make inference over an drinking signal
    update(update_cate):
        update one drinking category 
    para_update(update_cate, update_times): 
        update several drinking categories multiple times in parallel processing 
    subject_record(step):
        record updated subjective social norm for each agent at each timestep
    action_record(step, my_action):
        record the action produced by self at given timestep
    """
  
  # Class Variables
  global k 
  global norm_mus
  global norm_sd 
  global desire_sd 
  
  # tau stands for precision 1/(standard devision**2)
  norm_tau = 1/norm_sd**2
  desire_tau = 1/desire_sd**2
  

  def __init__(self, desire_mu, group, norm_weights, bias, update_rate, steps):
      self.desire_mu = desire_mu
      self.group = group
      self.norm_weights = norm_weights
      self.bias = bias
      self.update_rate = update_rate

      # n stands for new, they are combined for action production
      self.nmus = np.append(norm_mus, self.desire_mu)
      self.nsigmas = np.append(norm_sd, desire_sd)
      self.nweights = np.append(self.norm_weights*self.bias, (1-self.bias))

      # record subjective social norm 
      self.subject_weights = np.zeros((steps, k))
      # record produced actions
      self.all_actions = np.zeros(steps)
  
  def action_production(self):
      """
      produce an action based on agent's desire distribution and subjective social norm distribution

      Parameters
      ----------
      None

      Returns
      -------
      observed (float): an action produced and can be observed by a receiver agent
      """
      # Define the probability distributions for each component
      dists = [norm(mu, sigma) for mu, sigma in zip(self.nmus, self.nsigmas)]
      # solving roundoff error!!!! Need double checking
      self.nweights[-1] = 1 - np.sum(self.nweights[0:-1])
      # Generate a random number from the mixture distribution
      observed = np.random.choice(dists, p=self.nweights).rvs()
      return observed

  def print_desire(self):
    """
    print desire distribution

    Parameters
    ----------
    None

    Returns
    -------
    A desire distribution plot

    """
    x = np.linspace(0, 1.5, 1000)
    plt.plot(x, stats.norm.pdf(x, self.desire_mu, desire_sd))
    plt.show()
   
  def print_subject_norm(self):
      """
      produce subjective social norm distribution

      Parameters
      ----------
      None

      Returns
      -------
      A plot of subjective social norm distribution

      """
      # Define the Gaussian mixture distribution using a lambda function
      gaussian_mixture = lambda x, w, mu, sigma: np.sum([w[i] * np.exp(-0.5 * ((x - mu[i]) / sigma[i])**2) for i in range(len(w))], axis=0)
      # Define the parameters of the Gaussian mixture distribution
      x = np.linspace(0, 1.3, 1000)
      # Evaluate the Gaussian mixture distribution at x
      y = gaussian_mixture(x, self.norm_weights, norm_mus, norm_sd)
      # Plot the Gaussian mixture distribution
      plt.plot(x, y)
      plt.show()
  
  def print_action_production(self):
      """
      print the distribution where an action is produced from

      Parameters
      ----------
      None

      Returns
      -------
      A plot 

      """  
      n_samples = 1000
      samples = np.random.choice(self.nmus, size=n_samples, p=self.nweights) + np.random.choice(self.nsigmas, size=n_samples) * np.random.randn(n_samples)

      fig, ax = plt.subplots()
      ax.hist(samples, bins=30, density=True, alpha=0.5, label='Sum of Normal and Mixture of Gaussian')
      x = np.linspace(0, 1, 100)
      combine_dist = np.sum([w * norm(loc=m, scale=s).pdf(x) for w, m, s in zip(self.nweights, self.nmus, self.nsigmas)], axis=0)
      ax.plot(x, combine_dist, label='Normal Distribution')

  def predict_category(self, estimate_mu):
      """
      predict which drinking category this inferred desire mu came from

      Parameters
      ----------
      estimate_mu (float): an estiamted desire mu based on a action (drinking) signal

      Returns
      -------
      pred (int): predicted drinking category's index

      """
      # calculate the probability density function of each component at the data point
      p_c = [0]*k
      for i in range(k):
          p_c[i] = ((self.norm_weights[i] * norm.pdf(estimate_mu, loc=norm_mus[i], scale=norm_sd[i])))
          
      # predict the most likely component for the data point
      pred = np.argmax(p_c)

      # print("The most likely component for the data point is:", self.norm_mus[pred])
      return pred
  
  def posterior(self, x, signal):
      """
      solve posterior distribution based on signal given 

      Parameters
      ----------
      x (numpy array): parameter space?
      signal (float): observed drinking signal from an sender agent

      Returns
      -------
      posterior distribution P(mu|signal)

      """

      # only one signal is inferred at a time
      n = 1
      # components' posterior tau (precision)
      tau_c_p = self.norm_tau + self.desire_tau*n
      # components' prior tau divide posterior tau
      tau_c_l = self.norm_tau/tau_c_p
      
      post_mus = [0]*k
      c_part = [0]*k
      C_j = [0]*k
      w_C_j = [0]*k
      
      for i in range(k):
          post_mus[i] = (self.norm_tau[i]*norm_mus[i] + self.desire_tau*n*signal)/tau_c_p[i]
          c_part[i] = self.norm_tau[i]*norm_mus[i]**2 + n*self.desire_tau*signal**2 - tau_c_p[i]*post_mus[i]**2
          C_j[i] = np.sqrt(tau_c_l[i])*np.exp((-1/2)*c_part[i])
          w_C_j[i] = self.norm_weights[i]*C_j[i]

      sum_Cs = np.sum(w_C_j, axis = 0)
      post_w = w_C_j/sum_Cs
      post_sigmas = np.sqrt(1/tau_c_p)
      
      posterior_distribution = sum([w * norm.pdf(x, loc=mu, scale=sigma) for w, mu, sigma in zip(post_w, post_mus, post_sigmas)])
      return posterior_distribution

  def inference(self, signal):
      """
      make inference and infer which drinking category observed signal belongs to

      Parameters
      ----------
      signal (float): an drinking signal produced by a sender agent

      Returns
      -------
      temp_cate (int): index of drinking group along norm_mus

      """
      x = np.linspace(0, 1, 1000)
      mode_mix = x[np.argmax(self.posterior(x, signal))]
      # temp_cate is the actual index of the category index [3] means 0.4
      temp_cate = self.predict_category(mode_mix)
      
      return temp_cate

  def update(self, update_cate):
      """
      update the inferred category (update one at a time)

      Parameters
      ----------
      update_cate (int): index for the updated category

      Returns
      -------
      None

      """
      self.norm_weights[update_cate] += self.update_rate
      self.norm_weights = self.norm_weights/sum(self.norm_weights)
  
  def para_update(self, update_cate, update_times):
    self.norm_weights[update_cate] += [x * self.update_rate for x in update_times]
    self.norm_weights = self.norm_weights/np.sum(self.norm_weights)

  def subject_record(self, step):
      """
      record updated subjective norm perception weight per step

      Parameters
      ----------
      step (int): the current step

      Returns
      -------
      None

      """
      self.subject_weights[step] = self.norm_weights

  def action_record(self, step, my_action):
      """
      record the action the agent itself produced at current timestep

      Parameters
      ----------
      step (int): current timestep
      my_action (float): the action agent just produced

      Returns
      -------
      None

      """
      self.all_actions[step] = my_action


class ToM_0(Agent):
  """
  A class to represent a Simple Inference Agent (Here we refer to it as ToM_0)
  -------------------------
  A Subclass of Agent Class
  It requires no additional attributes or methods 
  It heritages everything from its base Class Agent
  Therefore, I 'pass' here
  """
  pass

class ToM_1(Agent):
  """
  A class to represent a Complex Inference Agent (Here we refer to it as ToM_1)
  --------------------------
  Another Subclass of Agent Class
  It heritages all Attributes from its base Class Agent
  It rewrites the inference method from its base Class Agent

  Methods
  -------
  partner_desire(y, signal):
     partner (sender) agent's desire distribution (that to be inferred)
  partner_norm(signal):
     parnter agent's subjective norm perception 
  log_infer_prior(y):
     prior the receiver hold on probability of which drinking category the sender agent belongs to
  log_posterior(y, signal):
     the log posterior over P(sender's mu|signal)
  metropolis_hastings_tom1(x0, num_samples, signal):
     metroplis hastings algorithm to sample parameter space for log_posterior distribution given signal
  inference(signal):
     make inference over which drinking category (index) this signal came from

  """
  def __init__(self, desire_mu, group, norm_weights, bias, update_rate, steps):
       super(ToM_1, self).__init__(desire_mu, group, norm_weights, bias, update_rate, steps)
  
  def partner_desire(self, y, signal):
      return (1-self.bias)*np.sum(norm.pdf(signal, loc=y, scale = np.sqrt(1/self.desire_tau)), axis = 0)
  
  def partner_norm(self, signal):
      return self.bias*np.sum(self.norm_weights * np.sum(norm.pdf(signal, loc=norm_mus, scale=np.sqrt(1/self.norm_tau))), axis=0)
  
  def log_infer_prior(self, y):
      return np.log(np.sum(self.norm_weights * norm.pdf(y, loc=norm_mus, scale=np.sqrt(1/self.norm_tau)), axis=0))

  def log_posterior(self, y, signal):
      return np.log(self.partner_desire(y, signal) + self.partner_norm(signal)) + self.log_infer_prior(y)
  
  # Define the Metropolis-Hastings algorithm for sampling from the posterior
  def metropolis_hastings_tom1(self, x0, num_samples, signal):
      samples = np.zeros(num_samples)
      theta = x0
      acceptances = 0
      for i in range(num_samples):
          # Propose a new theta from a normal distribution
          theta_proposed = np.random.normal(loc=theta, scale=norm_sd[0])
          # Compute the acceptance ratio
          # log_acceptance_ratio = log_posterior(theta_proposed, data) - log_posterior(theta, data)
          log_acceptance_ratio = self.log_posterior(theta_proposed, signal) - self.log_posterior(theta, signal)
          # Accept or reject the proposed theta
          if np.random.uniform() < np.exp(log_acceptance_ratio):
              theta = theta_proposed
              # acceptances += 1
          # Apply parameter constraint
          # theta = max(min(theta, 1), 0) --> MH natrually set constraints
          samples[i] = theta
      # acceptance_rate = acceptances / num_samples
      return samples  # acceptance_rate

  def inference(self, signal):
    x0 = 0.5
    # sample 2500 times (Can change to reduce running time)
    samples = self.metropolis_hastings_tom1(x0, 2500, signal)
    # find MAP of all 2500 samples
    samples_temp = pd.Series(samples.round(2))
    current = pd.Series.tolist(samples_temp.mode())[0]
    # find which drinking group this MAP belongs to
    cate_temp = self.predict_category(current)
    return cate_temp
