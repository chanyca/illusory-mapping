import os
import pandas as pd


def get_subject_ids():
    # Get unique subject IDs
    subject_ids = set()
    for filename in os.listdir('csv/modeling/data'):
        if filename.startswith('LV') or filename.startswith('SV'):
            parts = filename.split('_')     
            subject_ids.add(parts[0])
            
    return sorted(list(subject_ids))


def get_subject_ids_for_beep():
    # Get unique subject IDs
    subject_ids = set()
    for filename in os.listdir('csv/'):
        if filename.startswith('SV2') and 'beep' in filename:
            parts = filename.split('_')    
            subject_ids.add(parts[0])
            
    return sorted(list(subject_ids))

def find_best_models(model, sigma_a=0.2):
    # find best model for each subject based on BIC

    output_dir = f'csv/modeling/outputs/sigma_a_p{int(sigma_a*10)}/'

    results_df = pd.read_csv(f'{output_dir}{model}_fitting_results.csv')
    parameter_columns = ['pcommon', 'sigma_p', 'mu_p', 'sigma_v', 'bic', 'r2']

    best_models = []
    for location in ['raw', 'visible', 'invisible', '5', '10', '15']:
        for subject in get_subject_ids():
            temp_df = results_df.query(f"subject_id == '{subject}' and location == '{location}'")
            if temp_df.empty:
                continue
            best_row = temp_df.loc[temp_df['bic'].idxmin()]
            # convert Series to dict to avoid building a list of Series objects
            best_models.append(best_row.to_dict())

    # create DataFrame from collected dicts
    best_models_df = pd.DataFrame(best_models)

    # add group column if possible
    if not best_models_df.empty and 'subject_id' in best_models_df.columns:
        best_models_df['group'] = best_models_df['subject_id'].apply(lambda x: 'Low Vision' if str(x).startswith('LV') else 'Control')

    # save results
    best_models_df.to_csv(f'{output_dir}{model}_best_models.csv', index=False)

    # get summary for best models
    if not best_models_df.empty:
        best_summary_table = best_models_df.groupby(['group', 'location'])[parameter_columns].agg(['mean', 'sem'])
        best_summary_table = best_summary_table.round(2)
        print(best_summary_table)

        print(best_models_df.groupby(['group', 'location'])['strategy'].value_counts())

    return best_models_df

def get_best_models(model, sigma_a=0.2):
    output_dir = f'csv/modeling/outputs/sigma_a_p{int(sigma_a*10)}/'
    df = pd.read_csv(f'{output_dir}{model}_best_models.csv')
    return df


def print_parameters(model, sigma_a=0.2, loc='visible'):

    df = get_best_models(model, sigma_a)
    df = df[df['location']==loc].copy()

    PARAMS = ["pcommon", "sigma_v", "sigma_p"]
    
    group_cols = ['group']

    stats = df.groupby(group_cols)[PARAMS].agg(['mean', 'sem']).round(2)

    display_df = pd.DataFrame(index=stats.index)
    for p in PARAMS:
        display_df[p] = (
            stats[p]['mean'].map('{:.2f}'.format) + 
            " ± " + 
            stats[p]['sem'].map('{:.2f}'.format)
        )

    print(f'\n======== {model.upper()} Parameters (Mean ± SEM) =============')
    print(display_df)

    returned_df = stats.copy()
    returned_df.columns = [f"{col[0]}_{col[1]}" for col in returned_df.columns]
    
    return returned_df.reset_index()

def print_strategy(model, sigma_a=0.2, loc='visible'):
    df = get_best_models(model, sigma_a)
    df = df[df['location']==loc].copy()

    counts = (
        df.groupby('group')['strategy']
          .value_counts(normalize=True)
          .rename('% win')
          .mul(100)
          .reset_index()
    )

    print(counts)