import numpy as np

######### load all conditions
def set_parameter():
    '''
    Generate simulated conditions by parameters given through input()

        Parameters:
               None

        Returns:
               params (numpy array)
    '''
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

    return params