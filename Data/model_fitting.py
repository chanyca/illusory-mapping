# conda create --name bcitoolbox python=3.10

from bcitoolbox_local import fit
import pandas as pd
import numpy as np
import os
from helpers import get_subject_ids, find_best_models
import argparse

#################################################
### Command line arguments ######################
#################################################
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Prep modeling data')
parser.add_argument('-sigma_a',      type=float,      default=0.2)

args = parser.parse_args()

sigma_a = args.sigma_a


#################################################
### Prep modeling data ##########################
#################################################
os.system(f"python modeling_data.py -sigma_a {sigma_a}")

#################################################
### Prep and pre-defined variables ##############
#################################################
output_dir = f'csv/modeling/outputs/sigma_a_p{int(sigma_a*10)}'
os.makedirs(output_dir, exist_ok=True)

bci_refit = False
ff_refit  = False
fs_refit  = False
mle_refit = False

es_para_dict = {
    #       pcommon, sigma_v, sigma_a, sigma_p, mu_p, dU, dD
    'bci': [1,       1,       0,       1,       0,    0,  0],
    'ff':  [0,       1,       0,       1,       0,    0,  0], 
    'fs':  [0,       1,       0,       1,       0,    0,  0], 
    'mle': [0,       1,       0,       0,       0,    0,  0], 
}

# fix mu_p at 1.5
# Wozny, Beierholm, Shams (2008) JOV
mu_p = 1.5

# fix sigma_a at 0.2
# Chan et al. (2025) bioRxiv
# sigma_a = 0.2


fixvalue_dict = {
    'bci': [0.5,0.4,sigma_a,4000,mu_p,0,0], # Auditory parameters fixed at sigma_a (from rabbit data)
    'ff':  [1.0,0.4,sigma_a,4000,mu_p,0,0], # Forced fusion - p_common fixed at 1
    'fs':  [0.0,0.4,sigma_a,4000,mu_p,0,0], # Forced fusion - p_common fixed at 0
    'mle': [1.0,0.4,sigma_a,4000,mu_p,0,0], # MLE - p_common fixed at 1, extremely flat prior (sigma_p = 4000)
}

strategy_map = {
    'ave': 'Averaging',
    'sel': 'Selection',
    'mat': 'Matching',
}

#################################################
### fit_data function ###########################
#################################################

def fit_data(file_name, modeling_data_path='csv/modeling/data/',
             n_parameters=5, n_simulation=10000, 
             es_para=[1,1,0,1,1,1,0], fixvalue=[0.5,0.4,0.2,4000,2,0,0],
             Strategies=['ave'], FitType='mll'):
    

    data_file_path = modeling_data_path + file_name
    behavior_data = np.loadtxt(data_file_path, delimiter=',')

    # some low vision subjects have 0 visible locations, and some control subjects have 0 invisible locations
    # Need at least 10 trials to fit
    if len(behavior_data) < 10: 
        print('Not enough data. Skipped.')
        return

    bounds = [(0, 1),(0.1, 3),(0.1, 3),(0.1,3),(0, 3.5),(-10,10),(-10,10)]
    bounds_use = bounds[:n_parameters]

    estimated_parameters, error, strategy_name, bic, r2, fixvalue = fit(n_parameters, n_simulation, behavior_data,
                                                                        bounds=bounds_use, es_para=es_para, fixvalue=fixvalue,
                                                                        Strategies=Strategies, FitType=FitType)
    param_configs = [
        {'name': 'pcommon', 'fixed_idx': 0},
        {'name': 'sigma_v', 'fixed_idx': 1},
        {'name': 'sigma_a', 'fixed_idx': 2},
        {'name': 'sigma_p', 'fixed_idx': 3},
        {'name': 'mu_p',    'fixed_idx': 4},
        {'name': 'dU',      'fixed_idx': 5},
        {'name': 'dD',      'fixed_idx': 6},
    ]

    params = {}
    pa_index = 0
    for i, config in enumerate(param_configs):
        if es_para[i] == 1: # free parameter
            params[config['name']] = estimated_parameters[pa_index]
            pa_index += 1
        else: # fixed parameter
            params[config['name']] = fixvalue[i]

    pcommon, sigma_v, sigma_a, sigma_p, mu_p, dU, dD = params.values()
    full_parameters = pcommon, sigma_v, sigma_a, sigma_p, mu_p

    print("\nEstimated parameters: ")
    print(f"all estimated parameters: {estimated_parameters} | Strategy = {strategy_name}")
    print(f"Total number of estimated parameters: {len(estimated_parameters)}")
    print(f"\nError = {error} | BIC = {bic} | r-squared = {r2}")

    return full_parameters, error, bic, r2, strategy_name, FitType


#################################################
### MAIN LOOP ###################################
#################################################

subject_ids = get_subject_ids()

model_list  = ['bci', 'ff', 'fs', 'mle']
refit_ls    = [bci_refit,    ff_refit,    fs_refit,    mle_refit]
location_list = ['raw', 'visible', 'invisible', '5', '10', '15']

# Parameter names (will be truncated to match length of full_parameters)
PARAM_ORDER = ['pcommon', 'sigma_v', 'sigma_a', 'sigma_p', 'mu_p']

for model, refit in zip(model_list, refit_ls):
    es_para      = es_para_dict[model]
    fixvalue     = fixvalue_dict[model]
    n_parameters = int(np.sum(es_para))

    strategy_list = ['ave', 'sel', 'mat'] if model == 'bci' else ['ave']

    output_path = os.path.join(output_dir, f"{model}_fitting_results.csv")
    print(output_path)

    # Load existing results (if any)
    if not refit:
        try:
            existing_df = pd.read_csv(output_path)
        except FileNotFoundError:
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()

    # Existing results we will keep writing to
    fitting_results = (
        existing_df.to_dict('records') if not existing_df.empty and not refit
        else []
    )

    # Build a set of existing (subject_id, location, strategy, fit_type) keys
    if not existing_df.empty:
        # Normalize strategy column to str just in case
        if 'strategy' in existing_df.columns:
            existing_df['strategy'] = existing_df['strategy'].astype(str)
        existing_keys = set(
            zip(
                existing_df.get('subject_id', pd.Series(dtype=object)),
                existing_df.get('location', pd.Series(dtype=object)),
                existing_df.get('strategy', pd.Series(dtype=str)),
            )
        )
    else:
        existing_keys = set()

    # Iterate locations/subjects/strategies
    for location in location_list:
        for subject in subject_ids:
            for strategy in strategy_list:
                print(f"\nFitting Model: {model} | Subject: {subject} | location: {location} | Strategy: {strategy}")
                key = (subject, location, strategy_map[strategy])

                # Skip this exact combination iff it already exists
                if not refit and key in existing_keys:
                    print('Already done. Skipped.')
                    continue
                

                file_name = f"{subject}_{location}.csv"

                result = fit_data(
                    file_name,
                    n_parameters=n_parameters,
                    es_para=es_para,
                    fixvalue=fixvalue,
                    Strategies=[strategy] 
                )

                if result is None:
                    continue

                full_parameters, error, bic, r2, strategy_name, FitType = result
                realized_key = (subject, location, strategy_name, FitType)
                existing_keys.add(realized_key)

                row = {
                    'subject_id': subject,
                    'location': location,
                    'strategy': strategy_name,
                    'fit_type': FitType,
                    'error': error,
                    'bic': bic,
                    'r2': r2,
                }
                for i, pname in enumerate(PARAM_ORDER[:len(full_parameters)]):
                    row[pname] = full_parameters[i]

                fitting_results.append(row)

    results_df = pd.DataFrame(fitting_results)

    front_cols = ['subject_id', 'location', 'strategy', 'fit_type', 'error', 'bic', 'r2']
    param_cols = [c for c in PARAM_ORDER if c in results_df.columns]
    other_cols = [c for c in results_df.columns if c not in front_cols + param_cols]
    if not results_df.empty:
        results_df = results_df[front_cols + param_cols + other_cols]

    results_df.to_csv(output_path, index=False)

    # find_best_models(model)
