from agent import *
from interact import *
from inspection import *
from param import *
import pickle
import operator
import os
import matplotlib.cm as cm
import antropy as ant
from scipy.stats import entropy
from scipy.spatial.distance import braycurtis

# path to where data are stored
cwd = os.getcwd()+ '\data\yuechen'


########### Generate file names ########## 
# how many runs per condition?
runs = 20
# where do you want to start run?
q_run = input('Where do you want to start run?: ')
start = int(q_run)
# all simulation conditions
params = set_parameter()


########## Set filename according to params
fileNum = len(params)
fileList = [None] * fileNum
fileNames = [None] * fileNum

for i in range(fileNum):
    '''
    fileList: a list of variable names each point to a file containing 1 simulated run
    fileNames: a list of actual pickle file names stored somewhere 
    '''
    agent_ratio, bias, net_type, run = params[i]
    fileList[i] = f"{agent_ratio.astype(int)}{(10*round(bias,1)).astype(int)}{net_type.astype(int)}n{run.astype(int)}"
    fileNames[i] = f"ratio_{agent_ratio}_bias_{bias}_net_{net_type.astype(int)}_run_{run.astype(int)}.pkl"


########### Load all pickled original files ########## 

# import pickled data files 
for i in range(fileNum):
    filename = fileNames[i]
    var_names = ['record', 'CN', 'IN', 'agents', 'network'] 
    variables = []
    
    for var_name in var_names:
        variables.append(f"{var_name}{fileList[i]}")
    
    filepath = os.path.join(cwd, filename)
    # Load pickled file
    with open(filepath, 'rb') as file:
        pickled_data = pickle.load(file)
    
    # Assign dictionary elements to variables dynamically using itemgetter
    itemgetter_func = operator.itemgetter(*var_names)
    values = itemgetter_func(pickled_data)
    
    for x in range(5):
       exec(f"{variables[x]} = values[x]")

    exec(f'{variables[0]}["actualGroup"] = ({variables[0]}["senderGroup"]*5).astype(int)') 

########### Create lists of runs for all conditions ########## 
conditions = input('How many conditions do you have? ')
for i in range(int(conditions)):
    exec(f"{'list'}{fileList[i*runs][0:3]} = {fileList[i*runs:(i+1)*runs]}")



########### Generate Indirect Measures ########## 

for i in range(len(fileList)):
    '''
    CN: Collective Objective Behaviour
    CB: Collective Subjective Belief
    ColAction: CN in dataframe
    ColBelief: CB in dataframe
    ColAction_IN: discrepancy between IN and collective objective behaviour
    ColBelief_IN: discrepancy between IN and collective subjective belief
    ColAction_ColBelief: discrepancy between collective objective behaviour & collective subjective belief
    IndBelief_IN: discrepancy between IN and subjective social norm perception per step 
                (indiviudal + subjective + belief) vs (collective + objective + beliefs) 
    IndBelief_CN: discrepancy between CN and subjective social norm perception per step
                (indiviudal + subjective + belief) vs (collective + objective + behaviour)
    
    '''
   
    steps = 5000
    mus = np.arange(0.1, 1, 0.2)
    mus = np.round(mus,2)
    mus_str = ["".join(item) for item in mus.astype(str)]
    sds = 0.1
    k = 5
    N = 50
    
    ####### Collective Level
    # 1. Collective Objective Behaviour: ColAction
    exec(f"{'ColAction'}{fileList[i]} = pd.DataFrame(np.transpose({'CN'}{fileList[i]}), index = mus_str)")
    # 2. Collective Subjective Belief: ColBelief
    exec(f"{'CB'}{fileList[i]} = dominant_one({'agents'}{fileList[i]},steps,N,k)")
    exec(f"{'ColBelief'}{fileList[i]} = pd.DataFrame(np.transpose({'CB'}{fileList[i]}), index = mus_str)")
    # 3. Discrepancy in KL divergence between Collective Objective Behaviour and Collective Objective Belief (Initial Norm)
    exec(f"{'ColAction_IN'}{fileList[i]} = ignorance({'CN'}{fileList[i]}, {'IN'}{fileList[i]}, steps)")
    # 4. Discrepancy in KL divergence between Collective Subjective Belief and Collective Objective Belief (Initial Norm)
    exec(f"{'ColBelief_IN'}{fileList[i]} = ignorance({'CB'}{fileList[i]}, {'IN'}{fileList[i]}, steps)")
    # 5. Discrepancy in JS divergence between Collective Subjective Belief and Collective Objective Behaviour 
    exec(f"{'ColAction_ColBelief'}{fileList[i]} = misperception_colcol_js({'CN'}{fileList[i]}, {'CB'}{fileList[i]}, steps)")
    
    
    ####### Individual to Collective Level
    # Discrepancy in KL divergence between Each individual agent's subjective social norm perception 
    # 6. and Initial Norm
    exec(f"{'IndBelief_IN'}{fileList[i]} = ts_initial_MP(N, steps, {'agents'}{fileList[i]}, {'IN'}{fileList[i]})")
    # 7. and Collective Objective Behaviour
    exec(f"{'IndBelief_CN'}{fileList[i]} = ts_current_MP(N, steps, {'agents'}{fileList[i]}, {'CN'}{fileList[i]})")
    # 8. and Collective Subjective Belief
    exec(f"{'IndBelief_CB'}{fileList[i]} = ts_current_MP(N, steps, {'agents'}{fileList[i]}, {'CB'}{fileList[i]})")
    


######### Save Measures to Dictionary and save to pickle files
CB = {}
ColBelief = {}
ColAction = {}
ColAction_IN = {}
ColBelief_IN = {}
ColAction_ColBelief = {}
IndBelief_IN = {}
IndBelief_CN = {}
IndBelief_CB = {}


for i in range(len(fileList)):
    
    exec(f"CB['CB{fileList[i]}'] = {'CB'}{fileList[i]}")
    exec(f"ColBelief['ColBelief{fileList[i]}'] = {'ColBelief'}{fileList[i]}")
    exec(f"ColAction['ColAction{fileList[i]}'] = {'ColAction'}{fileList[i]}")
    exec(f"ColAction_IN['ColAction_IN{fileList[i]}'] = {'ColAction_IN'}{fileList[i]}")
    exec(f"ColBelief_IN['ColBelief_IN{fileList[i]}'] = {'ColBelief_IN'}{fileList[i]}")
    exec(f"ColAction_ColBelief['ColAction_ColBelief{fileList[i]}']={'ColAction_ColBelief'}{fileList[i]}")
    exec(f"IndBelief_IN['IndBelief_IN{fileList[i]}']= {'IndBelief_IN'}{fileList[i]}")
    exec(f"IndBelief_CN['IndBelief_CN{fileList[i]}'] = {'IndBelief_CN'}{fileList[i]}")
    exec(f"IndBelief_CB['IndBelief_CB{fileList[i]}'] = {'IndBelief_CB'}{fileList[i]}")



### save to pickled files
# pickle them 
with open('CB.pkl', 'wb') as f:
    pickle.dump(CB, f)
    
with open('ColBelief.pkl', 'wb') as f:
    pickle.dump(ColBelief, f)

with open('ColAction.pkl', 'wb') as f:
    pickle.dump(ColAction, f)
    
with open('ColAction_IN.pkl', 'wb') as f:
    pickle.dump(ColAction_IN, f)
    
with open('ColBelief_IN.pkl', 'wb') as f:
    pickle.dump(ColBelief_IN, f)
    
with open('ColAction_ColBelief.pkl', 'wb') as f:
    pickle.dump(ColAction_ColBelief, f)

with open('IndBelief_IN.pkl', 'wb') as f:
    pickle.dump(IndBelief_IN, f)

with open('IndBelief_CN.pkl', 'wb') as f:
    pickle.dump(IndBelief_CN, f)

with open('IndBelief_CB.pkl', 'wb') as f:
    pickle.dump(IndBelief_CB, f)


Col_Col = ['ColBelief_IN','ColAction_IN', 'ColAction_ColBelief'] 
In_Col = ['IndBelief_IN','IndBelief_CN','IndBelief_CB']


####### Individual to Collective Average all runs all agents
IndBelief_IN_avg = {}
IndBelief_CN_avg = {}
IndBelief_CB_avg = {}
for condition in range(int(conditions)):
    # e.g.: listCondition = list120, where list120 contains ['120n0', '120n1', '120n2'....]
    exec(f"listCondition = {'list'}{fileList[condition*runs][0:3]}")

    for in_col in In_Col:
        # In_Col = ['IndBelief_IN','IndBelief_CN'] which is all Individual-to-Collective discrepancy measures
        # Create an empty list called in_col120 where 120 stands for condition
        exec(f"{in_col}{listCondition[0][0:3]} = []")

        # Loop through all runs 
        # add in_col120n0 to in_col120
        # combine all runs into 1 list
        for i in range(runs):
            exec(f"{in_col}{listCondition[0][0:3]}.append({in_col}{listCondition[i]})")
        
        # from list to np.array to dataframe
        # add mean for all runs, .25 and .75 quantile 
        # Concatenate the arrays along axis=0
        exec(f"{in_col}{listCondition[0][0:3]} = np.concatenate({in_col}{listCondition[0][0:3]}, axis=0)")
        exec(f"{in_col}{listCondition[0][0:3]} = pd.DataFrame(np.transpose({in_col}{listCondition[0][0:3]}))")
        exec(f"{in_col}{listCondition[0][0:3]}['mean'] = {in_col}{listCondition[0][0:3]}.mean(axis = 1)")
        exec(f"{in_col}{listCondition[0][0:3]}['upper'] = {in_col}{listCondition[0][0:3]}.quantile(0.75, axis = 1)")
        exec(f"{in_col}{listCondition[0][0:3]}['lower'] = {in_col}{listCondition[0][0:3]}.quantile(0.25, axis = 1)") 
        # this triple loop create IndBelief_IN120 out of IndBelief_IN120n0
        # one dataframe that contains all runs rather than separate run separate varible
        exec(f"{in_col}_avg['{in_col}{listCondition[0][0:3]}'] = {in_col}{listCondition[0][0:3]}")
        

# save pickle file
with open('IndBelief_IN_avg.pkl', 'wb') as f:
    pickle.dump(IndBelief_IN_avg, f)
    
with open('IndBelief_CN_avg.pkl', 'wb') as f:
    pickle.dump(IndBelief_CN_avg , f)

with open('IndBelief_CB_avg.pkl', 'wb') as f:
    pickle.dump(IndBelief_CB_avg , f)


########### Generate Direct Measures ########## 
q_network = input('Which network do you want to generate direct measrures? 0 fully connected, 2 small-world: ')

if q_network == '0':
    # all measures for fully-connected network
    totalList = ['list080','list050','list020','list120', 'list150', 'list180']
elif q_network == '2':
    # all measures for small world network
    totalList = ['list082', 'list052', 'list022', 'list122', 'list152','list182']

newList = [item[4:] for item in totalList]



############ Agent Action
data_action = {key: None for key in newList}
for l in totalList:
    exec(f"{'action'+l[-3:]} = []")
    for i in range(start, start+runs):
        for j in range(N):
            exec(f"x = {'agents'+l[-3:]+'n'}{i}{[j]}.all_actions")
            y = ant.sample_entropy(x)
            exec(f"{'action'+l[-3:]}.append(y)") 

    exec(f"data_action[l[-3:]] =  {'action'+l[-3:]}")


with open('action_sample_entropy.pkl', 'wb') as f:
    pickle.dump(data_action, f)



############ Subjective Social Norm Perception ###########
sub_norm_entropy = {'df_ent' + key: None for key in newList}
for l in totalList:
    exec(f"{'df_ent'+l[-3:]} = np.empty([50*runs, steps])")
    for n in range(start, start+runs):
        # at each run: n
        # one particular agent: j
        # at per timestep: i
        for j in range(N):
            ent = []
            for i in range(steps):
                exec(f"p = {'agents'+l[-3:]+'n'}{n}{[j]}.subject_weights{[i]}")
                ent.append(entropy(p))
            
            exec(f"{'df_ent'+l[-3:]}[(n-start)*50+j,:] = ent")
   
    exec(f"sub_norm_entropy['df_ent'+l[-3:]] = {'df_ent'+l[-3:]}")



# save to pickle file
with open('sub_norm_entropy.pkl', 'wb') as f:
    pickle.dump(sub_norm_entropy, f)



############ Collective Action Entropy ###########
collect_action_entropy = {'ent_cn' + key: None for key in newList}
for l in totalList:
    exec(f"{'ent_cn' + l[-3:]} = np.empty([runs, steps])")
    for n in range(start, start+runs):
        ent = []
        for i in range(steps):
            exec(f"p = {'CN'+l[-3:]+'n'}{n}{[i]}")
            ent.append(entropy(p))
            
        exec(f"{'ent_cn'+l[-3:]}[{n-start},:] = ent") 

    exec(f"collect_action_entropy['ent_cn'+l[-3:]] = {'ent_cn'+l[-3:]}")              


# save fickle file
with open('collect_action_entropy.pkl', 'wb') as f:
    pickle.dump(collect_action_entropy, f)


######## Collective Belief Entropy #########
collect_belief_entropy = {'ent_cb' + key: None for key in newList}
for l in totalList:
    exec(f"{'ent_cb' + l[-3:]} = np.empty([runs, steps])")
    for n in range(start, start+runs):
        ent = []
        for i in range(steps):
            exec(f"p = {'CB'+l[-3:]+'n'}{n}{[i]}")
            ent.append(entropy(p))
            
        exec(f"{'ent_cb'+l[-3:]}[{n-start},:] = ent")  

    exec(f"collect_belief_entropy['ent_cb'+l[-3:]] = {'ent_cb'+l[-3:]}")          

# save to pickled file
with open('collect_belief_entropy.pkl', 'wb') as f:
    pickle.dump(collect_belief_entropy, f)             



####### Accuracy 
inference_accuracy = {'correct' + key: None for key in newList}
for l in totalList:
    exec(f"{'correct'+l[-3:]} = []")
    for i in range(start, start+runs):
        exec(f"correctNum = sum({'record'+l[-3:]+'n'}{i}['infer'] == {'record'+l[-3:]+'n'}{i}['actualGroup'])")
        exec(f"correct = correctNum/(50*5000)")
        exec(f"{'correct'+l[-3:]}.append(correct)")
    
    exec(f"inference_accuracy['correct'+l[-3:]] = {'correct'+l[-3:]}")       

with open('inference_accuracy.pkl', 'wb') as f:
    pickle.dump(inference_accuracy, f)             


####### Agreeableness
# agree to each other
group_harmony = {'bray' + key: None for key in newList}
for l in totalList:
    exec(f"{'bray'+l[-3:]} = []")
    for i in range(start, start+runs):
        for j in range(N):
            exec(f"heatmap1 = {'agents'+l[-3:]+'n'}{i}{[j]}.subject_weights")
            heatmap1 = heatmap1.flatten()
            for p in range(N-j-1):
                exec(f"heatmap2 = {'agents'+l[-3:]+'n'}{i}{[j+p+1]}.subject_weights")
                heatmap2 = heatmap2.flatten()
                cur_bray = braycurtis(heatmap1, heatmap2)
                exec(f"{'bray'+l[-3:]}.append(cur_bray)") 

    exec(f"group_harmony['bray'+l[-3:]] = {'bray'+l[-3:]}")    

with open('group_harmony.pkl', 'wb') as f:
    pickle.dump(group_harmony, f)     


####### Agreeable to collective belief
agree_collect = {'bray' + key: None for key in newList}
for l in totalList:
    exec(f"{'bray'+l[-3:]} = []")
    for i in range(start, start+runs):
        exec(f"heatmap2 = {'CB'+l[-3:]+'n'}{i}")
        heatmap2 = heatmap2.flatten()
        for j in range(N):
            exec(f"heatmap1 = {'agents'+l[-3:]+'n'}{i}{[j]}.subject_weights")
            heatmap1 = heatmap1.flatten()
            cur_bray = braycurtis(heatmap1, heatmap2)
            exec(f"{'bray'+l[-3:]}.append(cur_bray)") 

    exec(f"agree_collect['bray'+l[-3:]] = {'bray'+l[-3:]}")    

with open('agree_collect.pkl', 'wb') as f:
    pickle.dump(agree_collect, f)    
