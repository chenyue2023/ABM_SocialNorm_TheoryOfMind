from agent2 import *
from interact2 import *
from inspection import *
import matplotlib.cm as cm


def plot_ColAction_ColBelief(list0, list1, group0, group1, cmax):
    '''

    Plot 2 compared conditions' all runs collective behaviour/collective belief side-to-side
    If you want to inspect all runs from 2 particular conditions
    Plot for condition comparision 
    
    Parameters:
                list0, list1:    2 conditions you want to compare
                group0, group1:  2 conditions' name in string for the title setting
                cmax:            the max colorbar limit

    Call example:
                plot_ColAction_ColBelief(list080, list180, '080', '180', 0.5)

                list080 contains all runs' names
                for example: ['080n0','080n1', '080n2']
                '080' is for the title
                cmax = 0.5 means max weight = 0.5
    Return:
                all heatmaps for all runs from conditions given as list0, list1

    '''
    f, axes = plt.subplots(len(list0), 4, figsize=(40, 8))
    axes[0, 0].set_title(group0+' Col Obj Behaviour')
    axes[0, 1].set_title(group0+' Col Sub Belief')
    axes[0, 2].set_title(group1+' Col Obj Behaviour')
    axes[0, 3].set_title(group1+' Col Sub Belief')
    
    for i in range(len(list0)):
        exec(f"sns.heatmap(ax = axes[i, 0], data ={'ColAction'}{list0[i]}, cmap = 'viridis', xticklabels = False, cbar = False, vmax = cmax)")
        exec(f"sns.heatmap(ax = axes[i, 1], data = {'ColBelief'}{list0[i]}, cmap = 'viridis', xticklabels = False, yticklabels = False,cbar = False, vmax = cmax)")
        exec(f"sns.heatmap(ax = axes[i, 2], data = {'ColAction'}{list1[i]}, cmap = 'viridis', xticklabels = False, cbar = False, vmax = cmax)")
        exec(f"sns.heatmap(ax = axes[i, 3], data = {'ColBelief'}{list1[i]}, cmap = 'viridis', xticklabels = False, yticklabels = False, cbar = False, vmax = cmax)")


def plot_CN_dominentN_IN(c2, under_cn, cmax):
    '''

    Plot a particular run's collective behaviour and collective action side-by-side 
    If you want to inspect what happened in a particular run
    Plot for per run inspection

    Parameter:
                c2:       a particular run's collective belief in pandas DataFrame 
                under_cn: same run's collective action in pandas DataFrame
                cmax:     max colorbar value
    Call example: 
                plot_CN_dominentN_IN(ColBelief150n18, ColAction150n18, cmax = 0.5)
                meanning: compare collective belief and collective action from condition 150n18 
                          (ToM1, conformity bias = 0.5, fully-connected network, number 18 run)

                If you want a closer look at all runs per condition (larger plot than plot_ColAction_ColBelief()),
                You could call it in a loop:
                list080 contains all runs' names, for example: ['080n0','080n1', '080n2']
                for i in range(runs):
                        exec(f"plot_CN_dominentN_IN({'ColBelief'}{list080[i]}, {'ColAction'}{list080[i]},  cmax = 0.5)")

    Return:
                2 heatmaps size-by-size

    '''
    f, axes = plt.subplots(1, 2, figsize=(20, 5), gridspec_kw={'width_ratios': [5,5], 'height_ratios':[4]})
    f.subplots_adjust(hspace=0.0001, wspace=0.13)
    #axes[0].set_title('Collective + Objective action Norm')
    #axes[1].set_title('Collective + Subjective Dominant Norm')
    subtitle0 = input('First subtitle is: ')
    subtitle1 = input('Second subtitle is: ')
    axes[0].set_title(subtitle0)
    axes[1].set_title(subtitle1)
    sns.heatmap(ax = axes[0], data = under_cn, cmap = 'viridis', xticklabels = False, cbar = True, vmin = 0,  vmax = cmax)
    sns.heatmap(ax = axes[1], data = c2, cmap = 'viridis', xticklabels = False, yticklabels = False, cbar = True, vmin = 0, vmax = cmax)


def inspection_4agents(agents, mus_str, agent_list):
    '''
    Inspect 4 agents' subjective norm perception heatmaps

    Parameters:
                agents:      a dictionary contains all agents ran in the simulation
                mus_str:     category means in string like ['0.1','0.3','0.5','0.7','0.9']
                agent_list:  4 number smaller than N (number of agent) that represent agent index


    Call example:           
                inspection_4agents(agents180n14, mus_str, [1,24,14,45])

    Return:
                4X4 cube-like heatmaps represent agent's subjective social norm perception change over timesteps
    '''
    # Set up the matplotlib figure
    f, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axs = [(0,0),(0,1),(1,0),(1,1)]
    for i in range(4):
        cur = agent_list[i]
        b = np.transpose(agents[cur].subject_weights)
        b = pd.DataFrame(b, index = mus_str)
        if str(type(agents[cur]).__name__) == 'ToM_1':
           agent_type = 'Complex'
        else:
            agent_type = 'Simple'

        axes[axs[i]].set_title(str(agents[cur].bias)+ 'biased ' + agent_type +' belongs to group:'+str(agents[cur].group))
        sns.heatmap(ax=axes[axs[i]], data = b, cmap = 'viridis', xticklabels = False, vmax = 1)#vmax = 1 #robust = True
        # print('agent belongs to group:', agents[cur].group, 'this is a', type(agents[cur]).__name__, 'agent')


# Collective-to-Collective Discrepancy Measures we are interested in        
# Col_Col = ['ColBelief_IN','ColAction_IN', 'ColAction_ColBelief']

def which_title(ColCol):
    '''

    Generate bold titles for collective-to-collective discrepancy plotting

    Parameters:
                ColCol(list): a list of collective to collective measures we want to plot in string
    Return:
                Title in string

    '''
    if ColCol == 'ColBelief_IN':
        t = " Discrepancy between "+ r"$\bf{"+"Collective \,Belief"+"}$" +" and "+ r"$\bf{"+"Initial \,Norm"+"}$"
    elif ColCol == 'ColAction_IN':
        t = " Discrepancy between "+ r"$\bf{"+"Collective \,Action"+"}$" + " and "+ r"$\bf{"+" Initial \,Norm"+"}$"
    elif ColCol == 'ColAction_ColBelief':
        t = " Discrepancy between "+ r"$\bf{"+"Collective \,Action"+"}$" + " and "+ r"$\bf{"+" Collective \,Belief"+"}$"
    
    return t

"""
def plot_Col_Col(ColCol, list0, list1, ylim0, ylim1, color0, color1):
    '''
    Plot a Collective-to-Collective measure for ALL runs for 2 conditions

    Paramters:
                ColCol(string): Name of that collective-to-collective measure you want to plot from Col_Col
                list0, list1:   2 conditions you are interested in
                ylim0, ylim1:   upper limit for 2 plots
                color0, color1: color0 for original data, color1 for 50 running average line


    Call Example:
                plot_Col_Col('ColBelief_IN', list020, list120, .5, .5, 'g', 'b')

    Return:
                2 plots each has all runs per condition in one plot

    '''

    t = which_title(ColCol)
    f, axes = plt.subplots(2, 1, figsize=(20, 8))
    f.suptitle(list0[0][1:-2] + t, fontsize = 22)
    axes[0].set_title(list0[0][:-2], fontsize = 15)
    axes[1].set_title(list1[0][:-2], fontsize = 15)
    axes[0].set_ylim(0, ylim0)
    axes[1].set_ylim(0, ylim1)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].set_xticks([])
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    for i in range(len(list0)):
        exec(f"axes[0].plot({ColCol}{list0[i]}['KLdiv'], color = color0, alpha = 0.1)")
        exec(f"axes[0].plot({ColCol}{list0[i]}['50_rolling_avg'], color = color1, alpha = .5)")
        exec(f"axes[1].plot({ColCol}{list1[i]}['KLdiv'], color = color0, alpha = 0.1)")
        exec(f"axes[1].plot({ColCol}{list1[i]}['50_rolling_avg'], color = color1, alpha = .5)")

    plt.show()


def Col_Col_Avg(ColCol, list0, remove):
    '''

    Aggregate a particular Collective-to-Collective measures' for all Runs in one DataFrame
    Then, compute average for all Runs either remove min/max or not

    Parameters:
                ColCol: particular col-to-col measures you want to compute average over all conditions in string
                list0:  the condition you are interested in, like list080
                remove: True or False, Do you want to remove min/max when averaging?

    Call example:
                Col_Col_Avg('ColBelief_IN', list080, remove = True)

    Return:
                a pandas DataFrame with all KLdiv for that measures for that condition in one dataframe
                and their mean
                and the 50 running average of their mean

    '''

    # remove max and min version
    avg_ColCol = pd.DataFrame()
    for i in range(len(list0)):
        exec(f"avg_ColCol['{list0[i][-5:]}'] = {ColCol}{list0[i]}['KLdiv']")
    
    if remove == True:
        arr = avg_ColCol.values
        # Find the indices of the smallest and largest values along each row
        min_indices = np.argmin(arr, axis=1)
        max_indices = np.argmax(arr, axis=1)
        # Create a copy of the array
        arr_copy = np.copy(arr)
        # Set the smallest and largest values to NaN
        arr_copy[np.arange(arr.shape[0]), min_indices] = np.nan
        arr_copy[np.arange(arr.shape[0]), max_indices] = np.nan

        # Calculate the mean along each row, excluding the NaN values
        mean_values = np.nanmean(arr_copy, axis=1)

        avg_ColCol[f"{'mean'}"] = mean_values
    else:
        avg_ColCol[f"{'mean'}"] = avg_ColCol.mean(axis=1)
    
    avg_ColCol['50_rolling_avg'] = avg_ColCol['mean'].rolling(50).mean()
    
    return avg_ColCol


def plot_Col_Col_avg(ColCol, totalList, ylim, remove, rolling, original):
    '''

    Plot a particular col-to-col measures for all conditions required by totalList

    Parameters:
            ColCol(str):     meansurement name in string
            totalList(list): all conditions interested 
            ylim:            y limit
            remove:          True/False, whether remove min/max when taking average
            rolling:         True/False, show 50-running average
            original:        True/False, show original time series? 

    Call example:
            plot_Col_Col_avg('ColBelief_IN', totalList, ylim=1, remove = True, rolling = True, original = True)

    Return:
            a line plots with different conditions as separate lines.

    '''

    t = which_title(ColCol)
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.set_figwidth(15)
    fig.suptitle(t, fontsize = 22)
    # Create a color palette using a blue colormap
    cmap = cm.get_cmap('bwr')
    ax.set_ylim(0,ylim)
    for i in range(len(totalList)):
        color = cmap(i * .2) 
        exec(f"{'Col_Col_avg'}{totalList[i][4:]} = Col_Col_Avg(ColCol, {totalList[i]}, remove)['mean']")
        
        if rolling == True:
            exec(f"ax.plot(Col_Col_Avg(ColCol, {totalList[i]}, remove)['50_rolling_avg'], linewidth = 4, color = color, alpha = 0.9)") #label = totalList[i][4:]
        if original == True:  
            exec(f"ax.plot({'Col_Col_avg'}{totalList[i][4:]}, color = color, alpha = 0.6, label = totalList[i][4:])")
        
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()
    

def compare_plot_Col_Col_avg0(ColCol, totalList, compareCondition, rmTitle, ylim, remove, rolling, origial, colors):
    '''

    Rather than plot all measures for all conditions in 1 plot, 
    this plotting function allows to to plot compared condition side by side

    Parameters:
                ColCol(string):     Any col-to-col measurements you want to plot
                totalList:          Total list of all conditions
                comapreCondition:   "Compare ToM" or "Compare Bias"
                rmTitle:            ?Do you need to removeTitle
                ylim:               set max y value
                remove:             reove mein/max
                rolling:            show 50 running average
                origial:            show original time series
                colors:             choose a color set you like

    Call example:
                compare_plot_Col_Col_avg0(Col_Col[i], tomList[0], "Compare ToM", False, ylim, True, True, True, colors1)
                compare_plot_Col_Col_avg0(Col_Col[i], biasList[0], "Compare bias", False, ylim, True, True, True, colors2)
                where biasList = [[['list020', 'list050', 'list080'], ['list120', 'list150', 'list180']]]
                biasList[0] = [['list020', 'list050', 'list080'], ['list120', 'list150', 'list180']]
                All items in a sub-list will be plot into 1 plot

    Return: 
                Many plots compare by condition required

    '''

    # check list shape
    # row contains data that need to be plot into separate 'row' subplots
    # column contains data that need to be in one subplo
    # therefore, this function can either plot 2 subplots or 3 subplots horizontally
    row = len(totalList)
    column = len(totalList[0])
    
    fig, ax = plt.subplots(1, row)
    fig.set_figwidth(20)
    if rmTitle == False:
        t = which_title(ColCol)
        fig.suptitle(t, fontsize=20)
    # Create a color palette using a viridis colormap
    cmap = cm.get_cmap("viridis")
    # Set y label
    ax[0].set_ylabel(compareCondition, fontsize = 18)
    # for each subplot: remove top and right border lines and set all ylim to be the same
    for r in range(row):
        ax[r].spines['top'].set_visible(False)
        ax[r].spines['right'].set_visible(False)
        ax[r].set_ylim(0,ylim)
        
        # for each data in one subplot
        for i in range(column):
            color = colors[i]
            exec(f"{'ColCol_avg'}{totalList[r][i][4:]} = Col_Col_Avg(ColCol, {totalList[r][i]}, remove)['mean']")
            if rolling == True:
                exec(f"ax[r].plot(Col_Col_Avg(ColCol, {totalList[r][i]}, remove)['50_rolling_avg'], linewidth = 2, color = color, alpha = 1)") # label = totalList[r][i][4:]
            if origial == True:
                exec(f"ax[r].plot({'ColCol_avg'}{totalList[r][i][4:]}, color = color, alpha = .8, linewidth = 2, label = totalList[r][i][4:])")

        ax[r].legend(loc='upper left')

    plt.show()


"""

def which_title2(InCol):
    '''

    Generate bold titles for indiviudal-to-collective discrepancy plotting

    Parameters:
                InCol(list): a list of individual to collective measures we want to plot in string
    Return:
                Title in string

    '''

    if InCol == 'IndBelief_IN':
        t = " Discrepancy between " + r"$\bf{"+"Individual \,Subjective \,Belief"+"}$" + " and " + r"$\bf{"+'Initial \,Norm'+"}$"
    elif InCol == 'IndBelief_CN':
        t = " Discrepancy between "+ r"$\bf{"+"Individual \,Subjective \,Belief"+"}$"+ " and " + r"$\bf{"+'Collective \,Action'+"}$"

    return t


"""
def plot_In_Col_avg(in_col, totalList, ylim):
    '''

    Plot a particular in-to-col measures for all conditions required

    Parameters:
            in_col(str):     meansurement name in string 'IndBelief_IN' or 'IndBelief_CN'
            totalList(list): all conditions interested 
            ylim:            y limit

    Call example:
            plot_In_Col_avg(In_Col[0], totalList, ylim = 2.5)

    Return:
            a line plots with all conditions plot in 1 plot

    '''
    t = which_title2(in_col)
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.set_figwidth(20)
    fig.suptitle(t, fontsize=22)
    # Create a color palette using a blue colormap
    cmap = cm.get_cmap("bwr")
    ax.set_ylim(0,ylim)
    for i in range(len(totalList)):
        color = cmap(i*.2) 
        exec(f"plot_target = {in_col}{totalList[i][4:]}")
        exec(f"ax.plot(plot_target['mean'], linewidth = 4, color = color, alpha = .8, label = totalList[i][4:])")
        exec(f"ax.fill_between(np.arange(0, steps, 1), plot_target['lower'], plot_target['upper'], color=color, alpha=0.1)")
        
    plt.legend(loc = 'upper left')
    plt.show()


def compare_plot_In_Col_avg0(in_col, totalList, compareCondition, ylim, colors):
    '''

    Rather than plot all measures for all conditions in 1 plot, 
    this plotting function allows to to plot compared condition side by side for in-to-col measures

    Parameters:
                in_col(string):     Any col-to-col measurements you want to plot
                totalList:          Total list of all conditions
                comapreCondition:   "Compare ToM" or "Compare Bias"
                ylim:               set max y value
                colors:             choose a color set you like

    Call example:
                compare_plot_In_Col_avg0(In_Col[0], conditionList[0], conditionName, ylim, colors)
                conditionList[0] = [['list020', 'list050', 'list080'], ['list120', 'list150', 'list180']]
                OR something like: [['list020', 'list120'], ['list050', 'list150'], ['list080', 'list180']] when comparing ToM
                Make sure that lines need to be plotted in one figure plot are in 1 sub-list
                conditionName = 'Compare Conformity Bias' OR 'Compare ToM'
                cmap = cm.get_cmap("Spectral")
                colors = [cmap(0.75), cmap(75)]

    Return: 
                Many plots compare by condition required

    '''

    # This function can either plot 2 subplots or 3 subplots horizontally
    row = len(totalList)
    column = len(totalList[0])
    
    fig, ax = plt.subplots(1, row)
    fig.set_figwidth(20)
    # set title
    t = which_title2(in_col)
    fig.suptitle(t, fontsize=20, y = 0.96)
    # Create a color palette using a viridis colormap
    # cmap = cm.get_cmap(colors)
    # Set y label
    fig.text(0.09, 0.5, compareCondition, va='center', rotation='vertical', fontsize = 18)
    
    # for each subplot: remove top and right border lines and set all ylim to be the same
    for r in range(row):
        ax[r].spines['top'].set_visible(False)
        ax[r].spines['right'].set_visible(False)
        ax[r].set_ylim(0,ylim)
       
        for i in range(column):
                color = colors[i]
                exec(f"plot_target = {in_col}{totalList[r][i][4:]}")
                exec(f"ax[r].plot(plot_target['mean'], linewidth = 4, color = color, alpha = 1, label = totalList[r][i][4:])")
                exec(f"ax[r].fill_between(np.arange(0, steps, 1), plot_target['lower'], plot_target['upper'], color=color, alpha=0.1)")

        ax[r].legend(loc='upper left')
            
    plt.show()

"""

def plot_sample_entropy_actions(data_to_plot):
    '''

    plot Sample Entropy of all agents all action time series for all conditions

    Parameters:
                data_to_plot (dict): in dictionary, contains all conditions' agents' action sample entropy
    Return:
                a violin plot

    '''

    fig, ax = plt.subplots()
    fig.set_figwidth(8)
    fig.set_figheight(7)
    fig.suptitle("Unpredictability of Action Produced by Agents", fontsize=20, y = 0.96)

    # Create the violin plot
    # data_to_plot are action data
    ax.violinplot(data_to_plot.values())
    # Set x-axis tick labels
    ax.set_xticks(np.arange(1, len(data_to_plot) + 1))
    ax.set_xticklabels(data_to_plot.keys(), fontsize = 10)

    ax.set_ylim(0, 2.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.show()