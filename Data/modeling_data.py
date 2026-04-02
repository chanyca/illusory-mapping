import numpy as np
import pandas as pd
import os
import re
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

print("Generating modeling data files...")
print(f"sigma_a value fixed at {sigma_a}")

#################################################
### Prep and pre-defined variables ##############
#################################################
modeling_path = 'csv/modeling/data/'
if not os.path.exists(modeling_path):
    os.makedirs(modeling_path)


csv_files = [
    f for f in os.listdir('csv/')
    if re.match(r'^(SV|LV)\d{3}\.csv$', f)
]
subject_ids = set()
for filename in csv_files:
    parts = filename.split('_')
    subject = parts[-1].replace('.csv', '')        
    subject_ids.add(subject)
# print(subject_ids)
    
del csv_files

columns_to_drop = ['sid', 'group', 'eye', 'glasses', 'rt', 'location']
ordered_columns = ['n_flash', 'n_beep', 'response', 'response_beep']
degrees = [5, 10, 15]
visibility = ['Visible', 'Invisible']

def generate_bayesian_pseudo_beep(n_beep, sigma_a=0.2):
    x_a = np.random.normal(loc=n_beep, scale=sigma_a)
    res = int(np.clip(np.round(x_a), 0, 3))
    return res

for subject_id in subject_ids:
    df = pd.read_csv(f'csv/{subject_id}.csv')
    df_subset = df.drop(columns_to_drop, axis=1)
    # use sigma_a to generate simulated response
    df_subset['response_beep'] = df_subset['n_beep'].apply(generate_bayesian_pseudo_beep, sigma_a=sigma_a)

    # save raw data, regardless of eccentricity (degree) and visibility
    df_all = df_subset.copy()
    df_all = df_all[ordered_columns]
    df_all_values = df_all.to_numpy() # remove headers
    np.savetxt(f'{modeling_path}{subject_id}_raw.csv', df_all_values, delimiter=',', fmt='%d')

    # save by degree
    for degree in degrees:
        df_deg = df_subset[df_subset['deg']==degree]
        df_deg = df_deg[ordered_columns]
        df_deg_values = df_deg.to_numpy()
        np.savetxt(f'{modeling_path}{subject_id}_{degree}.csv', df_deg_values, delimiter=',', fmt='%d')

    # save by visibility
    for vis in visibility:
        df_vis = df_subset[df_subset['visibility']==vis]
        df_vis = df_vis[ordered_columns]
        df_vis_values = df_vis.to_numpy()
        np.savetxt(f'{modeling_path}{subject_id}_{vis}.csv', df_vis_values, delimiter=',', fmt='%d')

# modeling data files for estimating sigma_a from control group SV2**

subject_ids = set()
for filename in os.listdir('csv/'):
    if filename.startswith('SV2') and 'beep' in filename:
        parts = filename.split('_')    
        subject_ids.add(parts[0])
        
subject_ids = sorted(list(subject_ids))

ordered_columns = ['n_flash', 'n_beep', 'response_flash', 'response']

for subject_id in subject_ids:
    df = pd.read_csv(f'csv/{subject_id}_beep.csv')
    # add dummy visual response
    df['response_flash'] = df['n_flash']

    # save raw data, regardless of eccentricity (degree) and visibility
    df_all = df.copy()
    df_all = df_all[ordered_columns]
    df_all_values = df_all.to_numpy() # remove headers
    np.savetxt(f'{modeling_path}{subject_id}_beep.csv', df_all_values, delimiter=',', fmt='%d')

print('Done! Generated modeling data files.')