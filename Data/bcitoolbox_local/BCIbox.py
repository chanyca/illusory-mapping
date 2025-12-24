#!/usr/bin/env python
# coding: utf-8

# =====================================
# Updates
# Modified simulateVV according to BCIboxGUI/simulateVV_GUI (source code v0.1.0.6, not the one on GitHub) for bug fixes
# ============================================

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import random
import os 
from scipy.optimize import minimize
from scipy.fftpack import dct
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from .BCIboxGUI import simulateVV_GUI


def prod_gaus(mu1, mu2, sigma1, sigma2):
    mu1 = np.array(mu1)
    mu2 = np.array(mu2)
    sigma1 = np.array(sigma1)
    sigma2 = np.array(sigma2)
    
    a = 1 / (2 * sigma1**2)
    b = 1 / (2 * sigma2**2)
    
    # New mean
    muN = (sigma2**2 * mu1 + sigma1**2 * mu2) / (sigma1**2 + sigma2**2)

    # New sigma
    sigmaN = np.sqrt(1 / (2 * (a + b)))

    # Integral
    pre = np.exp(-(a * b / (a + b) * (mu1 - mu2)**2)) / (sigma1 * sigma2 * 2 * np.pi)
    c = pre * np.sqrt(np.pi / (a + b))

    return c, muN, sigmaN, pre


def prod3gauss(mu1, mu2, mu3, sigma1, sigma2, sigma3):
    c, muN, sigmaN, _ = prod_gaus(mu1, mu2, sigma1, sigma2)
    c *= prod_gaus(muN, mu3, sigmaN, sigma3)[0]

    return c, muN, sigmaN


def simulateVV(paras, n, data, biOnly=0,  strategy='ave' , fitType='dif', 
               es_para=[1,1,1,1,1,0,0], fixvalue=[0.5,0.4,0.8,4000,2,0,0]):
    # Import Data into Python
    dtl = data[0,:]
    cols = len(dtl)
    responses = data[:, [2, 3]]
    stimuli = data[:, [0, 1]]
    N = np.max(stimuli) + 1
    modelprop = []
    dataprop = []
    plt.clf()
    print(f"fit type = {fitType}")
    
    # Set Random Seed
    s1 = np.random.get_state()
    s2 = np.random.get_state()
    np.random.seed(13)
    
    # Default Parameters
    pcommon, sigmaU, sigmaD, sigmaZ, PZ_center, dU, dD = fixvalue
    print(f"===== DEBUG: fixed values = {fixvalue} =====")

    sU, sD, sUm, sDm = 70000.5, 70000.5, 0, 0
    p_cutoff = 0.5

    # =============================================================
    # ======================== Ailene ==============================
    param_configs = [
        {'name': 'pcommon', 'transform': lambda x: min(abs(x), 1),   'fixed_idx': 0},
        {'name': 'sigma_v', 'transform': lambda x: max(0.1, abs(x)), 'fixed_idx': 1},
        {'name': 'sigma_a', 'transform': lambda x: max(0.1, abs(x)), 'fixed_idx': 2},
        {'name': 'sigma_p', 'transform': lambda x: abs(x),           'fixed_idx': 3},
        {'name': 'mu_p',    'transform': lambda x: abs(x),           'fixed_idx': 4},
        {'name': 'dU',      'transform': lambda x: abs(x),           'fixed_idx': 5},
        {'name': 'dD',      'transform': lambda x: abs(x),           'fixed_idx': 6},
    ]
    params = {}
    pa_index = 0
    for i, config in enumerate(param_configs):
        if es_para[i] == 1:
            raw_value = paras[pa_index]
            params[config['name']] = config['transform'](raw_value)
            pa_index += 1
        else:
            params[config['name']] = fixvalue[config['fixed_idx']]

    pcommon, sigmaU, sigmaD, sigmaZ, PZ_center, dU, dD = params.values()

    print(f"===== DEBUG: pcommon = {pcommon} =====")
    print(f"===== DEBUG: sigma_v = {sigmaU} =====")
    print(f"===== DEBUG: sigma_a = {sigmaD} =====")
    print(f"===== DEBUG: sigma_p = {sigmaZ} =====")
    print(f"===== DEBUG: mu_p = {PZ_center} =====")
    # =============================================================
    # =============================================================

    conditions = np.unique(stimuli, axis=0)
    
    # Real Stimuli, Repeat n times
    real = np.tile(conditions, (n, 1))
    
    # Create Mean of Distribution 
    # sU and sD is the 1/rate of increase away from center
    sigma_like = np.zeros((real.shape[0], 2))
    sigma_like[:, 0] = sigmaU * (1 + np.power(np.abs(sUm - real[:, 0]), 1) / sU)
    sigma_like[:, 1] = sigmaD * (1 + np.power(np.abs(sDm - real[:, 1]), 1) / sD)
    
    # Add noise
    noisy = np.zeros_like(real)
    noisy[:, 0] = real[:, 0] + np.random.randn(real.shape[0]) * sigma_like[:, 0] + dU
    noisy[:, 1] = real[:, 1] + np.random.randn(real.shape[0]) * sigma_like[:, 1] + dD
    
    # New sigma_like based on the added noise
    sigma_like[:, 0] = sigmaU * (1 + np.power(np.abs(sUm - np.maximum(0, noisy[:, 0])), 1) / sU)
    sigma_like[:, 1] = sigmaD * (1 + np.power(np.abs(sDm - np.maximum(0, noisy[:, 1])), 1) / sD)

    # Make sure no missing stimuli
    sigma_like[real[:, 0] == 0, 0] = 1000
    sigma_like[real[:, 1] == 0, 1] = 1000
    
    #########
    # Calculate p(C|D,U)
    #########
    
    # CalculetP(U,D|C)
    # Integral of P(U|Z)*P(D|Z)*P(Z)
    PDUC = prod3gauss(noisy[:, 0], noisy[:, 1], PZ_center, sigma_like[:, 0], sigma_like[:, 1], sigmaZ)[0]
    # Integral of P(U|Z)*P(Z) times integral of P(D|Z)*P(Z)
    PDUnC = prod_gaus(noisy[:, 0], PZ_center, sigma_like[:, 0], sigmaZ)[0] * prod_gaus(noisy[:, 1], PZ_center, sigma_like[:, 1],
                                                                                       sigmaZ)[0]

    # Posterior of Common Cause Given Signals
    PCDU = np.multiply(PDUC, pcommon) / (np.multiply(PDUC, pcommon) + np.multiply(PDUnC, 1 - pcommon))
    #print(np.shape(PCDU))
    #########
    # Calculate Sc_hat
    #########
    
    Sc = (
    (sigma_like[:, 1] ** 2 * sigmaZ ** 2 * noisy[:, 0]) +
    (sigma_like[:, 0] ** 2 * sigmaZ ** 2 * noisy[:, 1]) +
    (sigma_like[:, 0] ** 2 * sigma_like[:, 1] ** 2 * PZ_center)
    ) / (
    (sigma_like[:, 0] ** 2 * sigma_like[:, 1] ** 2) +
    (sigma_like[:, 1] ** 2 * sigmaZ ** 2) +
    (sigma_like[:, 0] ** 2 * sigmaZ ** 2)
    )
    
    Snc1 = (
    (sigmaZ ** 2 / (sigma_like[:, 0] ** 2 + sigmaZ ** 2)) * noisy[:, 0] +
    (sigma_like[:, 0] ** 2 / (sigma_like[:, 0] ** 2 + sigmaZ ** 2)) * PZ_center
    )

    Snc2 = (
    (sigmaZ ** 2 / (sigma_like[:, 1] ** 2 + sigmaZ ** 2)) * noisy[:, 1] +
    (sigma_like[:, 1] ** 2 / (sigma_like[:, 1] ** 2 + sigmaZ ** 2)) * PZ_center
    )
    
    # Mean Responses (Sim)
    
    responsesSim = np.zeros((Sc.shape[0], 2))
    if strategy == 'ave' :
        # Averaging 
        responsesSim[:, 0] = PCDU * Sc + (1 - PCDU) * Snc1
        responsesSim[:, 1] = PCDU * Sc + (1 - PCDU) * Snc2
    
    elif strategy == 'sel': 
        # Selecting 
        responsesSim[:, 0] = np.where(PCDU > p_cutoff, Sc, Snc1)
        responsesSim[:, 1] = np.where(PCDU > p_cutoff, Sc, Snc2)
    
    elif strategy == 'mat': 
        # Matching
        p_cutoff = np.random.rand(Sc.shape[0])
        responsesSim[:, 0] = np.where(PCDU > p_cutoff, Sc, Snc1)
        responsesSim[:, 1] = np.where(PCDU > p_cutoff, Sc, Snc2)

    else:
        print("No valid strategy selected")
        return -1

    # ===========================================================
    # Added from bcitoolbox source code
    # Only account for unique, PRESENTED conditions
    # ===========================================================

    def assign_indices(matrix):
        _, indices = np.unique(matrix, axis=0, return_inverse=True)
        return indices.reshape(-1, 1)

    def assign_indices_f0(matrix):
    
        unique_rows, indices = np.unique(matrix, axis=0, return_inverse=True)

    
        non_zero_rows = (matrix[:, 0] != 0) & (matrix[:, 1] != 0)

        non_zero_indices = indices[non_zero_rows]
        return non_zero_indices

    trialType = assign_indices(stimuli)
    trialType = np.squeeze(trialType)
    trialTypeSim = assign_indices(real)
    trialTypeSim = np.squeeze(trialTypeSim) 
    
    # # Define Trial Type 
    # trialType = 1 + np.dot(stimuli, [N, 1])
    # trialTypeSim = 1 + np.dot(real, [N, 1])
    
    #########################################################
    # Create model probabilities/proportions from simulated responses
    modelprop = np.zeros((2, int(np.max(trialType).item())+1, int(N.item())))
    dataprop = np.zeros((2, int(np.max(trialType).item())+1, int(N.item())))
    # Numerical list of bimodal conditions
    # vector1 = np.ceil(np.arange(1, (N - 1) ** 2 + 1) / (N - 1))
    # vector2 = np.tile(np.arange(1, N), int(N.item()) - 1)
    # matrix2 = np.vstack((vector1, vector2))
    # matrix1 = np.array([N, 1])
    # bimodalList = matrix1 @ matrix2

    bimodalList = assign_indices_f0(conditions)
    bimodalList = bimodalList.astype(int)

    #print (np.max(trialType))
    '''
    for i in range(2):
        for j in range(1, int(np.max(trialType).item())+1):
            k1 = np.minimum(np.maximum(1, np.round(responsesSim[np.where(trialTypeSim==j),i]).astype(int)),N-1)#need to check
            k1 = k1.ravel().astype(int)
           
            for k in range(len(k1)):
                modelprop[i,j-1,k1[k]] += 1
            print(modelprop)
            k2 = np.minimum(np.maximum(0, np.round(responses[np.where(trialType==j),i]).astype(int)),N-1)
            k2 = k2.ravel().astype(int)
            for k in range(len(k2)):
                 dataprop[i,j-1,k2[k]] += 1
            
            dataprop[i,bimodalList,0] = 0
            dataprop[i,j-1,:] /= (1e-10+np.sum(np.squeeze(dataprop[i,j-1,:])))
            modelprop[i,j-1,:] /= (1e-10+np.sum(np.squeeze(modelprop[i,j-1,:])))
    print(modelprop)
    '''
    for i in range(2):
        for j in range(0, int(np.max(trialType).item())+1):
            k1 = np.minimum(np.maximum(1, np.round(responsesSim[np.where(trialTypeSim==j),i]).astype(int)), N-1)
            k1 = k1.ravel().astype(int)
            counts = np.bincount(k1, minlength=int(N)) 
            modelprop[i, j, :] = counts
            
            #print(modelprop)
            if cols == 3: 
                k2 = np.minimum(np.maximum(0, np.round(responses[np.where(trialType==j)]).astype(int)), N-1)
                k2 = k2.ravel().astype(int)
                counts2 = np.bincount(k2, minlength=int(N)) 
                dataprop[0,j,:] = counts2 #np.pad(counts2, (0, 0), mode='constant')

                dataprop[0, bimodalList, 0] = 0
                dataprop[0,j,:] /= (1e-10+np.sum(np.squeeze(dataprop[0,j,:])))
                
            else:

                k2 = np.minimum(np.maximum(0, np.round(responses[np.where(trialType==j),i]).astype(int)), N-1)
                k2 = k2.ravel().astype(int)
                counts2 = np.bincount(k2, minlength=int(N)) 
                dataprop[i,j,:] = counts2 #np.pad(counts2, (0, 0), mode='constant')

                dataprop[i, bimodalList, 0] = 0
                dataprop[i,j,:] /= (1e-10+np.sum(np.squeeze(dataprop[i,j,:])))
                
            modelprop[i,j,:] /= (1e-10+np.sum(np.squeeze(modelprop[i,j,:])))
     
    '''
    for i in range(2):
            
        trial_types = np.unique(trialType)
        max_trial_type = int(np.max(trialType).item())

        trialTypeMaskSim = trialTypeSim[:, np.newaxis] == trial_types[np.newaxis, :]
        trialTypeMask = trialType[:, np.newaxis] == trial_types[np.newaxis, :]

        k1_counts = np.bincount(np.minimum(np.maximum(1, np.round(responsesSim[np.where(trialTypeMaskSim[:, :, i])]).astype(int)), N-1).ravel(), minlength=4)
        modelprop[i, trial_types-1, :] = k1_counts

        k2_counts = np.bincount(np.minimum(np.maximum(0, np.round(responses[np.where(trialTypeMask[:, :, i])]).astype(int)), N-1).ravel(), minlength=4)
        dataprop[i, trial_types-1, :] = k2_counts


    dataprop[:, bimodalList, 0] = 0
    dataprop /= (1e-10 + np.sum(dataprop, axis=2, keepdims=True))
    modelprop /= (1e-10 + np.sum(modelprop, axis=2, keepdims=True))
    '''
    
    # dataprop = dataprop[:, 1:, :]
    # modelprop = modelprop[:, 1:, :]

    bimodalList = np.sum(conditions > 0, axis=1) > 1
    unimodal = np.sum(conditions > 0, axis=1) < 2
    
    modelprop[0, conditions[:, 0] == 0, 0] = 1 #recognize no stimulus
    modelprop[0, conditions[:, 0] == 0, 1:] = 0 #never respond 1, 2, 3...N

    modelprop[1, conditions[:, 1] == 0, 0] = 1
    modelprop[1, conditions[:, 1] == 0, 1:] = 0
    
    # Log likelihood
    modelTH = (1-0.001)*modelprop + 0.001*(1/N)
    
    # revert to numbers of responses
    npc = len(stimuli) // (N**2 - 1)  # n responses per condition
    if cols == 3: 
        loglike = npc * dataprop[0,:,:] * np.log(modelTH[0,:,:])
    
    else:  
        loglike = npc * dataprop * np.log(modelTH)
    
    if biOnly == 1:  # bimodal only
        biORuni = bimodalList
    else:
        biORuni = np.logical_or(unimodal, bimodalList)
        
    model_dat = modelprop[:, biORuni, :]  
    data_dat = dataprop[:, biORuni, :]
    
    # Multiple ways to define error, to be minimized:
    # Negative sum of log-likelihood
    if cols == 3: 
        minus_sll = -np.sum(np.sum(np.sum(loglike[biORuni,:])))
       
    else:
        minus_sll = -np.sum(np.sum(np.sum(loglike[:,biORuni,:])))

    # Negative R2
    if cols == 3:
        model_dat = modelprop[0, biORuni, :]  
        data_dat = dataprop[0, biORuni, :]  
        x = np.corrcoef(model_dat.ravel(), data_dat.ravel())  
    else:
        
        x = np.corrcoef(model_dat.ravel(), data_dat.ravel())
        
    mr2 = -(x[1, 0]**2)

    # Sum Squared Error
    sse = np.sum(np.sum(np.sum((model_dat - data_dat)**2)))

    print("R2 on all")
    print(x[1, 0] ** 2)

    # Calculate sum log likelihood, and the 'best' model if it perfectly matched data
    print("Sum Loglike    Optimal")
    if cols == 3:
        sum_loglike = np.sum(
        npc * dataprop[0, biORuni, :] * np.log((1 - 0.001) * modelprop[0, biORuni, :] + 0.001 / N)
        )
        optimal = np.sum(
        npc * dataprop[0, biORuni, :] * np.log((1 - 0.001) * dataprop[0, biORuni, :] + 0.001 / N)
        )
    else:
        sum_loglike = np.sum(
        npc * dataprop[:, biORuni, :] * np.log((1 - 0.001) * modelprop[:, biORuni, :] + 0.001 / N)
        )
        optimal = np.sum(
        npc * dataprop[:, biORuni, :] * np.log((1 - 0.001) * dataprop[:, biORuni, :] + 0.001 / N)
        )
        
    print(sum_loglike, optimal)

    # Different way of calculating R2
    if cols == 3:
        A = dataprop[0, biORuni, :]
        B = modelprop[0, biORuni, :]
    else:
        A = dataprop[:, biORuni, :]
        B = modelprop[:, biORuni, :]

    avg_data = np.mean(A)
    var_of_data = np.sum((A - avg_data) ** 2)
    avg_diff = np.mean(A - B)
    var_of_diff = np.sum(((A - B - avg_diff) ** 2))

    print("Explainable variance, r2:")
    r_square = 1 - var_of_diff / var_of_data
    print(r_square)
   
    '''
    #Plot
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    axs[0, 0].imshow(dataprop[0, :, :])
    axs[0, 0].set_title('Experimental Data, cond V resp')

    axs[0, 1].imshow(dataprop[1, :, :])
    axs[0, 1].set_title('Experimental Data, cond A resp')

    axs[1, 0].imshow(modelprop[0, :, :])
    axs[1, 0].set_title('Model, cond V resp')

    axs[1, 1].imshow(modelprop[1, :, :])
    axs[1, 1].set_title('Model, cond A resp')
    
    plt.savefig("output2.png") 
    plt.figure()
    
    plt.figure()
            
    plt.show()
    '''
    
    #print (loglike)
    
    if fitType == 'mll':
        error = minus_sll
    elif fitType == 'mr2':
        error = mr2
    else:
        error = sse
    
    error += 10000000 * int((sigmaU + sigmaD) < 0)
    
    np.random.set_state(s1)
    
    #np.savetxt('array10.txt', dataprop[1, :, :], fmt='%.6f')
    #np.savetxt('array13.txt', trialType, fmt='%.6f')
    
   
    return error, modelprop, dataprop, responsesSim, r_square #, real, sigma_like, noisy, PCDU, Sc

def plotKonrads(data, model=None, save_path=None):
    a = data.shape
    condi = int((a[1] + 1) ** 0.5)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)  

    for i in range(a[1]):
        if condi < 11:
            plt.subplot(condi, condi, i + 2)
        else:
            plt.figure(int(i/condi) + 1)
            plt.subplot(condi, 1, (i % condi) + 1)
        
        plt.plot(data[0, i, :], 'b')
     
        plt.plot(data[1, i, :], 'r')
        plt.axis([0, a[2]-1, 0, 1])

    if model is not None:
        for i in range(a[1]):
            if condi < 11:
                plt.subplot(condi, condi, i+2)
            else:
                plt.figure(int(i/condi) + 1)
                plt.subplot(condi, 1, (i % condi) + 1)
            
           
            plt.plot(model[0, i, :], 'b-.')
            plt.plot(model[1, i, :], 'r-.')
            plt.axis([0, a[2]-1, 0, 1])
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()

def fit(n_parameters, n_Simulation, Behavior_Data, n_seeds = 1, 
        bounds = [(0, 1),(0.1, 3),(0.1, 3),(0.1,3),(0, 3.5)],
        es_para=[1,1,1,1,1,1,1], fixvalue=[0.5,0.4,0.2,4000,2,0,0],
        biOnly=0, Strategies = ['ave'], FitType = 'mll'):
    result_M = None
    start_time = time.time()
    random_seeds = [13, 1089, 681, 304, 118, 817, 82, 736, 295, 424, 247, 732, 243, 366, 483, 415, 747, 926, 335, 394, 653, 968, 746, 944, 197, 871, 694, 466, 958, 42, 276, 45, 419, 43, 985, 190, 405, 35, 388, 523, 796, 239, 124, 291, 924, 491, 417, 482, 57, 861, 405, 226, 292, 501, 904, 920, 199, 287, 433, 531, 797, 686, 459, 812, 152, 243, 330, 306, 583, 397, 660, 729, 480, 925, 437, 831, 452, 506, 388, 988, 160, 169, 854, 353, 93, 872, 195, 556, 266, 752, 104, 64, 902, 103, 185, 216, 527, 802, 221, 778]
    data = Behavior_Data
    best_error = float('inf')
    best_result = None
    best_strategy = None
    best_r2 = None

    strategies = []
    if Strategies[0] == 'ave':
        strategies.append(('ave', 'Averaging'))
    elif Strategies[0] == 'sel':
        strategies.append(('sel', 'Selection'))
    elif Strategies[0] == 'mat':
        strategies.append(('mat', 'Matching'))
    
    for strategy, strategy_name in strategies:
        for seed in range(n_seeds):
            np.random.seed(random_seeds[seed])
            x0 = np.random.rand(n_parameters)

            def minimize_callback(xk):
                estim_t = (n_Simulation * 0.0035)*n_parameters/5
                cur_t = time.time()
                # Compute progress
                progress = (cur_t - start_time )/ estim_t
                print(progress)

            start_time = time.time()
     
            result = minimize(lambda paras: simulateVV(paras, n_Simulation, data, biOnly, 
                                                       strategy=strategy, fitType=FitType,
                                                       es_para=es_para, fixvalue=fixvalue)[0], x0, 
                                                       method='Powell', bounds=bounds)
            if result.success and result.fun < best_error:
                best_error = result.fun
                best_result = result.x
                best_strategy = strategy
                bic = 2 * best_error + n_parameters * np.log(len(data))
                # calculate r2
                _, _, _, _, best_r2 = simulateVV(best_result, n_Simulation, data, biOnly, 
                                                 strategy=best_strategy, fitType=FitType,
                                                 es_para=es_para, fixvalue=fixvalue)

        
    estimated_parameters = best_result
    print(f"DEBUG: estimated_parameters = {estimated_parameters}")
    error = best_error
    strategy = best_strategy
    end_time = time.time()     
    execution_time = end_time - start_time  
    print("Running time:", execution_time, "s")

    return estimated_parameters, error, strategy_name, bic, best_r2, fixvalue
    
def simulate_condition(condition, parameters, Strategies):
    # Default Parameters
    sti1 = condition[0] # v, flash
    sti2 = condition[1] # a, beep
    pcommon = parameters[0]
    sigmaU = parameters[1] # sigma_v
    sigmaD = parameters[2] # sigma_a
    sigmaZ = parameters[3] # sigma_p
    PZ_center = parameters[4] # mu_p

    dU = abs(parameters[5]) if len(parameters) >= 6 else 0
    dD = abs(parameters[6]) if len(parameters) >= 7 else 0
    
    sU = 70000.5 
    sD = 70000.5
    sUm = 0
    sDm = 0
    
    # Set Random Seed
    s1 = np.random.get_state()
    s2 = np.random.get_state()
    np.random.seed(13)
    
    n=1000
    conditions = np.array([sti1,sti2])
    real = np.tile(conditions, (n, 1))
    
    # Create Mean of Distribution 
    # sU and sD is the 1/rate of increase away from center
    sigma_like = np.zeros((real.shape[0], 2))
    sigma_like[:, 0] = sigmaU * (1 + np.power(np.abs(sUm - real[:, 0]), 1) / sU)
    sigma_like[:, 1] = sigmaD * (1 + np.power(np.abs(sDm - real[:, 1]), 1) / sD)

    # Add noise
    noisy = np.zeros_like(real)
    noisy[:, 0] = real[:, 0] + np.random.randn(real.shape[0]) * sigma_like[:, 0] + dU
    noisy[:, 1] = real[:, 1] + np.random.randn(real.shape[0]) * sigma_like[:, 1] + dD
    
    # New sigma_like based on the added noise
    sigma_like[:, 0] = sigmaU * (1 + np.power(np.abs(sUm - np.maximum(0, noisy[:, 0])), 1) / sU)
    sigma_like[:, 1] = sigmaD * (1 + np.power(np.abs(sDm - np.maximum(0, noisy[:, 1])), 1) / sD)
    
    #sigma_like[real[:, 0] == 0, 0] = 1000
    #sigma_like[real[:, 1] == 0, 1] = 1000
    #sigma_like[real[:, 0] == 0, 1] = 0.001
    #sigma_like[real[:, 1] == 0, 0] = 0.001
    #########
    # Calculate p(C|D,U)
    #########
    
    # CalculetP(U,D|C)
    # Integral of P(U|Z)*P(D|Z)*P(Z)
    PDUC = prod3gauss(noisy[:, 0], noisy[:, 1], PZ_center, sigma_like[:, 0], sigma_like[:, 1], sigmaZ)[0]
    # Integral of P(U|Z)*P(Z) times integral of P(D|Z)*P(Z)
    PDUnC = prod_gaus(noisy[:, 0], PZ_center, sigma_like[:, 0], sigmaZ)[0] * prod_gaus(noisy[:, 1], PZ_center, sigma_like[:, 1],
                                                                                       sigmaZ)[0]

    # Posterior of Common Cause Given Signals
    PCDU = np.multiply(PDUC, pcommon) / (np.multiply(PDUC, pcommon) + np.multiply(PDUnC, 1 - pcommon))
    #print(np.shape(PCDU))
    #########
    # Calculate Sc_hat
    #########
    
    Sc = (
    (sigma_like[:, 1] ** 2 * sigmaZ ** 2 * noisy[:, 0]) +
    (sigma_like[:, 0] ** 2 * sigmaZ ** 2 * noisy[:, 1]) +
    (sigma_like[:, 0] ** 2 * sigma_like[:, 1] ** 2 * PZ_center)
    ) / (
    (sigma_like[:, 0] ** 2 * sigma_like[:, 1] ** 2) +
    (sigma_like[:, 1] ** 2 * sigmaZ ** 2) +
    (sigma_like[:, 0] ** 2 * sigmaZ ** 2)
    )
    
    Snc1 = (
    (sigmaZ ** 2 / (sigma_like[:, 0] ** 2 + sigmaZ ** 2)) * noisy[:, 0] +
    (sigma_like[:, 0] ** 2 / (sigma_like[:, 0] ** 2 + sigmaZ ** 2)) * PZ_center
    )

    Snc2 = (
    (sigmaZ ** 2 / (sigma_like[:, 1] ** 2 + sigmaZ ** 2)) * noisy[:, 1] +
    (sigma_like[:, 1] ** 2 / (sigma_like[:, 1] ** 2 + sigmaZ ** 2)) * PZ_center
    )
    
    # Mean Responses (Sim)
    responsesSim = np.zeros((Sc.shape[0], 2))
    if Strategies == ['Averaging']:
        # Averaging 
        responsesSim[:, 0] = PCDU * Sc + (1 - PCDU) * Snc1
        responsesSim[:, 1] = PCDU * Sc + (1 - PCDU) * Snc2
        
    elif Strategies == ['Selection']:
        # Selecting 
        responsesSim[:, 0] = np.where(PCDU > 0.5, Sc, Snc1)
        responsesSim[:, 1] = np.where(PCDU > 0.5, Sc, Snc2)
    
    else:
        # Matching
        p_cutoff = np.random.rand(Sc.shape[0])
        responsesSim[:, 0] = np.where(PCDU > p_cutoff, Sc, Snc1)
        responsesSim[:, 1] = np.where(PCDU > p_cutoff, Sc, Snc2)
        
    # Response Distribution
    D1 = np.maximum(np.round(responsesSim[:, 0]),0)
    D2 = np.maximum(np.round(responsesSim[:, 1]),0)
    
    values_D1, counts_D1 = np.unique(D1, return_counts=True)
    values_D2, counts_D2 = np.unique(D2, return_counts=True)

    prob_density_D1 = counts_D1 / (len(D1) * 1)
    prob_density_D2 = counts_D2 / (len(D2) * 1)

    return values_D1, values_D2, prob_density_D1, prob_density_D2, D1, D2, Snc1, Snc2