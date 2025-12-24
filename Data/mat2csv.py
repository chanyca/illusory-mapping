# Imports
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import file_extractor
importlib.reload(file_extractor)
from file_extractor import FileExtractor
extractor = FileExtractor()

import get_df
importlib.reload(get_df)
from get_df import get_df
data_frame = get_df()

sid_list = extractor.get_sids()
mat_files = extractor.get_mat_files(beep=False)

# Helper functions to load MAT structures

from scipy.io import loadmat, matlab
def load_mat(filename):
    """
    This function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    def _check_vars(d):
        """
        Checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], matlab.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _toarray(d[key])
        return d

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, matlab.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _toarray(elem)
            else:
                d[strg] = elem
        return d

    def _toarray(ndarray):
        """
        A recursive function which constructs ndarray from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        if ndarray.dtype != 'float64':
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, matlab.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_toarray(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return np.array(elem_list)
        else:
            return ndarray

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_vars(data)


def classify_locations(df, chance=0.5):
    """
    Classify locations as visible or invisible based on response means.
    
    :param df: DataFrame containing trial data.
    :param visible_thres: Threshold for visible locations.
    :param invisible_thres: Threshold for invisible locations.
    :return: Two lists of locations (visible and invisible).
    """
    vis_loc = []
    invis_loc = []
    for loc in range(1, 25):
        loc_trials = df[df['location'] == loc]
        mean_response = loc_trials['response'].mean()
        if mean_response > chance:
            vis_loc.append(loc)
        elif mean_response <= chance:
            invis_loc.append(loc)
    return vis_loc, invis_loc

# Generate 'subject_details.csv'

details_df = pd.DataFrame()
columns=['SID', 'Age', 'Gender', 'Distance_R', 'Distance_L',
         'Near_R', 'Near_L', 'Low_R', 'Low_L',
         'Onset', 'Duration', 'Remarks']

records = pd.read_excel("subject_records.xlsx", sheet_name="Raw")

details_df[columns] = records[columns]
details_df['Group'] = np.where(details_df['SID'].str.startswith('LV'), 'Low Vision', 'Sighted Control')
details_df.sort_values(by='SID').to_csv("subject_details.csv", index=False)

# Eccentricity
five_deg_locs = list(range(1, 24, 3))
ten_deg_locs = list(range(2, 24, 3))
fifteen_deg_locs = list(range(3, 24+1, 3))

columns = ['sid', 'group', 'eye', 'glasses',  
           'n_flash', 'n_beep', 'response', 'accuracy', 'rt', 
           'location', 'deg', 'visibility']


for sid in sid_list:
    df = pd.DataFrame(columns=columns)
    sid_mat_files = [file for file in mat_files if file.startswith(sid)]
    all_trials = []
    shared_meta = {}

    for f in sid_mat_files:
        mat_contents = load_mat('mat/' + f)
        data = mat_contents['Data']

        if not shared_meta:
            shared_meta['sid'] = sid
            shared_meta['group'] = 'Low Vision' if f.startswith('LV') else 'Sighted Control'
            shared_meta['glasses'] = data['glasses']

        df_trials = pd.DataFrame()
        # trials = [location, n_beep, n_flash]
        df_trials['location']   = data['Conditions'][:, 0]
        df_trials['n_beep']     = data['Conditions'][:, 1].astype(float)
        df_trials['n_flash']    = data['Conditions'][:, 2].astype(float)
        df_trials['response']   = data['Responses']
        df_trials['rt']         = data['RT']
        df_trials['eye']        = data['Eye']

        all_trials.append(df_trials)
    
    # concatenate
    df = pd.concat(all_trials, ignore_index=True)

    # add subject metadata
    for k, v in shared_meta.items():
        df[k] = v

    # add eccentricity
    df['deg'] = np.where(df['location'].isin(five_deg_locs), 5,
                         np.where(df['location'].isin(ten_deg_locs), 10, 15))
    
    # compute visibility per eye
    df['visibility'] = 'unknown'
    for eye_val in df['eye'].unique():
        df_eye = df[(df['eye'] == eye_val) & (df['n_flash'] == 1) & (df['n_beep'] == 0)]
        vis_locs, invis_locs = classify_locations(df_eye)
        # print(vis_locs, invis_locs)
        df.loc[(df['eye'] == eye_val) & (df['location'].isin(vis_locs)), 'visibility'] = 'Visible'
        df.loc[(df['eye'] == eye_val) & (df['location'].isin(invis_locs)), 'visibility'] = 'Invisible'

    # compute accuracy
    df['accuracy'] = np.where(df['n_flash'] == df['response'], 1, 0)

    df = df[columns]
    df.sort_values(by=['n_flash', 'n_beep', 'eye'], inplace=True)
    # print(df)
    df.to_csv(f'csv/{sid}.csv', index=False)

    # Generate a csv with ALL participants
df_all = []
csv_files = extractor.get_csv_files(beep=False)
# print(csv_files)
for csv_file in csv_files:
    df_indiv = pd.read_csv(f'csv/{csv_file}')
    df_all.append(df_indiv)

df_all = pd.concat(df_all, ignore_index=True)
df_all.sort_values(by=['sid', 'eye'], inplace=True)
df_all.to_csv('csv/all_participants.csv', index=False)

# Beep responses
# 'ad' files

mat_files = extractor.get_mat_files(beep=True)


five_deg_locs = list(range(1, 24, 3))
ten_deg_locs = list(range(2, 24, 3))
fifteen_deg_locs = list(range(3, 24+1, 3))

columns = ['sid', 'group',  
           'n_flash', 'n_beep', 'response', 'accuracy', 'rt', 
           'location']


for f in mat_files:
    df = pd.DataFrame(columns=columns)
    all_trials = []
    shared_meta = {}

    mat_contents = load_mat('mat/' + f)
    data = mat_contents['Data']
    sid = data['SubjectID']

    df_trials = pd.DataFrame()
    # trials = [location, n_beep, n_flash]
    df_trials['location']   = data['Conditions'][:, 0]
    df_trials['n_beep']     = data['Conditions'][:, 1].astype(float)
    df_trials['n_flash']    = data['Conditions'][:, 2].astype(float)
    df_trials['response']   = data['Responses']
    df_trials['rt']         = data['RT']

    all_trials.append(df_trials)
    df = pd.concat(all_trials, ignore_index=True)
    df['sid'] = sid
    df['group'] = 'Low Vision' if f.startswith('LV') else 'Sighted Control'

    # compute accuracy
    df['accuracy'] = np.where(df['n_beep'] == df['response'], 1, 0)

    df = df[columns]
    df.sort_values(by=['n_flash', 'n_beep'], inplace=True)
    # print(df)
    df.to_csv(f'csv/{sid}_beep.csv', index=False)

print('Done! Converted mat to csv files.')