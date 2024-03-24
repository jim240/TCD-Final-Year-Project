"""
James O'Reilly

Learning Shortcuts in the Chemical Space:
A Search for High Performance Single Molecule
Magnet Candidates via Reinfocement Learning

B.A. Theoretical Physics
Final Year Capstone Project
School of Physics
Trinity College Dublin

Example source code for varying exploration parameter epsilon vs 
indicator ratios for both simple and complex synthetic datasets
"""

import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
import random
from datetime import datetime
import time
import itertools
from collections import Counter
from math import comb as nCr

# =============================================================================
# _________________________USER DEFINED PARAMETERS_____________________________
# =============================================================================

#Define N number of ligands available and sized M final molecules
N = 24
M = 4

epsilons = np.arange(0.05,1.05,0.05) #Range of epsilon exploration parameters that will be analysed

num_runs =  [100] * (len(epsilons))

seed(8) #Can use set seed to ensure sufficient complexity in optimal molecule
use_complex_dataset = True #Set True to use complex dataset or False to use simple dataset
n_param = 3 #dimensions of vectors/matrices in complex dataset model of ligand interaction

nth_best_value_to_consider_as_goal = 1 

allow_improved_updates = True #Allow the improved update scheme of the Q-learning algorithm

# =============================================================================
# ___________________________FUNCTION DEFINITIONS______________________________
# =============================================================================

def generate_simple_dataset(ligand_IDs, M):
    ligand_to_value_dict = {}
    for ID in ligand_IDs:
        ligand_to_value_dict[ID] = np.random.uniform(-10,10)
    
    reward_dict = {}
    for ID_tuple_as_molecule in itertools.combinations_with_replacement(ligand_IDs, M):
        reward = 0
        for ID in ID_tuple_as_molecule:
            reward += ligand_to_value_dict[ID]
        reward_dict[ID_tuple_as_molecule] = reward   
    return reward_dict

def generate_symmetric_full_rank_square_matrix(n):
    A = np.random.rand(n,n)
    A = A + np.transpose(A)
    
    if np.linalg.matrix_rank(A) == n:
        return A
    else:
        generate_symmetric_full_rank_square_matrix(n)

def generate_complex_dataset(ligand_IDs, M, n_param):
    ligand_to_vec_dict = {}
    for ID in ligand_IDs:
        rand_vec = np.random.rand(n_param)
        ligand_to_vec_dict[ID] = rand_vec / np.linalg.norm(rand_vec)
    
    reward_dict = {}
    A_matrices = [generate_symmetric_full_rank_square_matrix(n_param) for _ in range(nCr(M,2))]
    for ID_tuple_as_molecule in itertools.combinations_with_replacement(ligand_IDs, M):
        reward = 0
        for i, ID_pair in enumerate(itertools.combinations(ID_tuple_as_molecule, 2)):
            reward += ligand_to_vec_dict[ID_pair[0]] @ A_matrices[i] @ ligand_to_vec_dict[ID_pair[1]]
        reward_dict[ID_tuple_as_molecule] = reward   
    return reward_dict

#Function to generate all allowed states in the state space in order to index into the Q table
def generate_state_space(ligand_IDs, M):
    all_states = [] #[] is the initial empty state
    for m in range(M+1):
        states = [list(s) for s in itertools.combinations_with_replacement(ligand_IDs, m)] #with_replacement allows repeated elements in combinations
        all_states.extend(states) 
    final_states = [list(s) for s in itertools.combinations_with_replacement(ligand_IDs, M)]
    return all_states, final_states
  
#Function to take actions by adding on ligands
def action_step(state, action, reward_dict, M):
    copy_state = state.copy()
    copy_state.append(action)
    next_state = sorted(copy_state)
        
    #if moving to terminal state
    if len(next_state) == M:
        reward = reward_dict[tuple(next_state)]
    else:
        reward = 0
            
    return next_state, reward

#Function to choose actions via epsilon greedy policy over Q table
def get_action(state_index, epsilon, Q_table, ligand_IDs):
    #Randomly choose action to explore
    if np.random.rand() <= epsilon:
        action = random.choice(ligand_IDs)
    #Exploit knowledge by selecting best action from Q table values
    else:
        #May be one or multiple actions that result in best Q value
        actions = [i for i, j in enumerate(Q_table[state_index]) if j == max(Q_table[state_index])]
        action = random.choice(actions)   
    return action

#Function to determine the maximum Q value the algorithm should terminate after, here n for nth largest possible D_value
def max_Q_target(reward_dict, n):
    sorted_rewards = sorted(reward_dict.values(), reverse = True)
    target = sorted_rewards[n-1]
    return target

# =============================================================================
# =============================================================================
# =============================================================================

def main():
    ligand_IDs = np.arange(0,N,1) #Use ID system for referencing ligands
    
    if use_complex_dataset:
        reward_dict = generate_complex_dataset(ligand_IDs, M, n_param)
        dataset_type = "Complex"
    else:
        reward_dict = generate_simple_dataset(ligand_IDs, M)
        dataset_type = "Simple"
    
    max_element = max(reward_dict, key=reward_dict.get)
    print(f"The best molecule is {max_element} with value {reward_dict[max_element]}\n")
    
    all_states, final_states = generate_state_space(ligand_IDs, M)
    
    terminal_state_space_size = len(final_states)
    
    seed(int(time.time()))
    random.seed(datetime.now().timestamp()) #Can have set seed or random seed for each run taken from datetime
    
    target = max_Q_target(reward_dict, nth_best_value_to_consider_as_goal)
    
    initial_Q_values = min(reward_dict.values())
    
    #Initialising Parameters for Q Learning
    alpha = 1 #We want to completely replace initial Q values immediately with full -D values so alpha = 1
    gamma = 1 #We don't require discounting
    
    overall_performance_ratios = []
    overall_performance_ratios_sigmas = []
    overall_cost_ratios = []
    overall_cost_ratios_sigmas = []
    
    for i in range(len(epsilons)):
        print(f"epsilon = {epsilons[i]}\nRun no.")
        total_number_of_terminal_states_reached = []
        unique_number_of_terminal_states_reached = []
        max_Q_lists = []
        for r in range(num_runs[i]):
            print(r+1)
            Q_table = np.full((len(all_states)-len(final_states), N), initial_Q_values) #Initialise Q table for each run  
            max_Q = initial_Q_values
            max_Q_list = [max_Q] #list evolution of max Q value for each run
            terminal_states_reached = [] #before optimal Q reached
            n = 0 #no. episodes   
            while max_Q_list[-1] < target:
                state = [] #Initial empty state
                states_this_ep = [[]]
                actions_this_ep = []
                while len(state) < M:   
                    state_index = all_states.index(state)
                    action = get_action(state_index, epsilons[i], Q_table, ligand_IDs)
                    next_state, reward = action_step(state, action, reward_dict, M)
                    if len(next_state) < M:
                        next_state_index = all_states.index(next_state)
                    
                    states_this_ep.append(next_state)
                    actions_this_ep.append(action)
                    
                    #Need to find max Q value of the next state for the Q learning algorithm
                    if len(next_state) < M:
                        max_next_Q = np.amax(Q_table[next_state_index])
                    else:
                        max_next_Q = 0
                        terminal_states_reached.append(next_state)
                        
                    #Performing the Q Learning algorithm
                    updated_Q_value = (1 - alpha) * Q_table[state_index, action] + \
                        alpha * (reward + gamma * max_next_Q)
                        
                    Q_table[state_index, action] = updated_Q_value
                    
                    max_Q = max(max_Q, updated_Q_value)
        
                    state = next_state
                    
                #Backup updates
                if allow_improved_updates:
                    c1 = Counter(state)
                    for m in range(M):
                        possible_prior_states = itertools.combinations(state, m)
                        for prior_state in possible_prior_states:
                            c2 = Counter(prior_state)
                            diff = c1-c2
                            
                            actions = set(diff.elements())
                            
                            state_index = all_states.index(list(prior_state))
                            for action in actions:
                                Q_table[state_index, action] = max(updated_Q_value, Q_table[state_index, action])
    
                #append max Q at end of each episode
                max_Q_list.append(max_Q)
                    
                n += 1
                
            max_Q_lists.append(max_Q_list)
                
            total_number_of_terminal_states_reached.append(len(terminal_states_reached))
            #Need to convert from lists to tuples for set otherwise unhashable type error
            unique_terminal_states_reached = {tuple(s) for s in terminal_states_reached}
            unique_number_of_terminal_states_reached.append(len(unique_terminal_states_reached))
          
        # standard_deviations = [np.std(max_Q_list_at_ep) for max_Q_list_at_ep in itertools.zip_longest(*max_Q_lists, fillvalue=target)]
        
        # sum_max_Q_lists = [sum(Q) for Q in itertools.zip_longest(*max_Q_lists, fillvalue=target)]
        # average_max_Q_list = [sum_max_Q / num_runs[i] for sum_max_Q in sum_max_Q_lists]
        
        # upper_bound = [sum(x) for x in zip(average_max_Q_list, standard_deviations)]
        # lower_bound = [(a - b) for a, b in zip(average_max_Q_list, standard_deviations)]
        
        average_total_number_of_terminal_states_reached = np.mean(total_number_of_terminal_states_reached)
        average_unique_number_of_terminal_states_reached = round(np.mean(unique_number_of_terminal_states_reached), 6)
        
        overall_performance_ratio_for_each_run = [no_states_reached/terminal_state_space_size for no_states_reached in total_number_of_terminal_states_reached]
        overall_cost_ratio_for_each_run = [no_states_reached/terminal_state_space_size for no_states_reached in unique_number_of_terminal_states_reached]
        
        overall_performance_ratio = round(average_total_number_of_terminal_states_reached / terminal_state_space_size, 4)   
        computational_performance_ratio = round(average_unique_number_of_terminal_states_reached / terminal_state_space_size, 4)
        
        overall_performance_ratios.append(overall_performance_ratio)
        overall_cost_ratios.append(computational_performance_ratio)
        overall_performance_ratios_sigmas.append(np.std(overall_performance_ratio_for_each_run)) 
        overall_cost_ratios_sigmas.append(np.std(overall_cost_ratio_for_each_run))
      
    fig1 = plt.figure(1, figsize=(12,14))
    gs = fig1.add_gridspec(2, hspace=0.05)
    (ax1, ax2) = gs.subplots(sharex=True)
    
    ax1.errorbar(epsilons, overall_performance_ratios, yerr = overall_performance_ratios_sigmas, color = 'deepskyblue', lw = 4, ecolor = 'k', elinewidth = 3, capsize=6)
    
    # x1, x2, y1, y2 = 0.04, 0.81, 0, 1.3
    # axins = ax1.inset_axes(
    #     [0.274, 0.24, 0.556, 0.72],
    #     xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    # #axins.plot(epsilons, I1_multiple, color = 'red', lw = 4)
    # axins.errorbar(epsilons, overall_performance_ratios, yerr = overall_performance_ratios_sigmas, color = 'deepskyblue', lw = 4, ecolor = 'k', elinewidth = 3, capsize=6)
    # axins.grid()
    # axins.set_yticks(np.arange(0,1.4,0.2), labels = ["0","0.2","0.4","0.6","0.8", "1", "1.2"], fontsize = 16)
    # axins.set_xticks(np.arange(0.1,0.9,0.1), labels = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"], fontsize = 16)
    
    ax2.set_xlabel("$\epsilon$", fontsize = 24)
    ax2.tick_params(axis='x', labelsize=22)
    ax2.set_xticks(np.arange(0,1.1,0.1))
    ax2.set_yticks(np.arange(0,1.1,0.1))
    ax2.tick_params(axis='y', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.set_ylabel("Performance Ratio", fontsize = 24)
    if allow_improved_updates:
        with_or_without_IU = "With"
    else:
        with_or_without_IU = "Without"
    fig1.suptitle(f"Varying $\epsilon$ for {dataset_type} Synthetic Dataset, M = {M}, N = {N} \n {with_or_without_IU} Improved Update Scheme", fontsize = 24, y=0.95, fontweight = 'bold')
    #ax1.legend(["Default", "Single Step", "Full Backup"], prop={'size': 18}, loc = "upper right")
    ax1.grid()
    
    ax2.errorbar(epsilons, overall_cost_ratios, yerr = overall_cost_ratios_sigmas, color = 'lime', lw = 4, ecolor = 'k', elinewidth = 3, capsize=6)
    ax2.set_ylabel("Cost Ratio", fontsize = 24)
    ax2.set_ylim(0,1)
    ax2.grid()
    plt.show()
    
    # plt.savefig('file_name.png', bbox_inches='tight', dpi=300)
    
    # with open("file_name.txt", "w") as f:
    #     f.write("\n")
    #     f.write(f"M = {M}, N = {N}")
    #     f.write("\n")
    #     f.write(f"Num Runs = {num_runs}")
    #     f.write("\n\n")
    #     f.write(f"Epsilons = {epsilons}")
    #     f.write("\n\n")
    #     f.write(f"Performance Ratio = {overall_performance_ratios}")
    #     f.write("\n\n")
    #     f.write(f"Performance Ratio Std Dev = {overall_performance_ratios_sigmas}")
    #     f.write("\n\n")
    #     f.write(f"Cost Ratio = {overall_cost_ratios}")
    #     f.write("\n\n")
    #     f.write(f"Cost Ratio Std Dev = {overall_cost_ratios_sigmas}")
    
if __name__ == '__main__':
    main()

