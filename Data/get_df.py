# Generic imports
import os
import numpy as np
import pandas as pd

# Custom imports
from file_extractor import FileExtractor

class get_df:
    def __init__(self):
        self.extractor = FileExtractor()        

    def avg_flash_LR_combined(self, nf, nb):
        """
        Gets the average flashes reported by all subjects, left and right eyes combined.

        Parameters:
            file_suffix (str): File suffix to filter.

        Returns:
            pd.DataFrame: DataFrame with the average flash location report for all subjects.
        """
        sids = self.extractor.get_sids()

        data_frames = []
        for sid in sids:
            df = pd.read_csv(os.path.join(self.extractor.csv_dir, sid+'.csv'))
            df = df.query('n_flash == @nf and n_beep == @nb')
            response = df['response'].mean()
            if nb==0:
                response = min(response, 1)
            df = pd.DataFrame({'sid': [sid], 'avg_flash': [response]})
            data_frames.append(df)
        fin_df = pd.concat(data_frames, ignore_index=True)
        return fin_df
    
    def avg_flash_LR_separate(self, file_suffix):
        """
        Gets the average flashes reported by all subjects, left and right eyes separately.

        Parameters:
            file_suffix (str): File suffix to filter.

        Returns:
            pd.DataFrame: DataFrame with the average flash location report for all subjects.
        """
        sids = self.extractor.get_sids()
        data_frames = []
        for sid in sids:
            sid_files = [file for file in self.extractor.get_csv_files(directory=self.extractor.cond_dir) if sid in file and file_suffix in file]
            L = pd.read_csv(os.path.join(self.extractor.cond_dir, sid_files[0]))
            R = pd.read_csv(os.path.join(self.extractor.cond_dir, sid_files[1]))            
            n_flash_L = L['response'].mean()
            n_flash_R = R['response'].mean()
            df = pd.DataFrame({'sid': [sid], 'avg_flash_L': [n_flash_L], 'avg_flash_R': [n_flash_R]})
            data_frames.append(df)

        return pd.concat(data_frames, ignore_index=True)
    
    def perc_illusion_by_deg(self):
        """
        Gets % illusion by degree (eccentricity)

        Returns:
            pd.DataFrame: DataFrame with the % illusion by degree.
        """
        sids = self.extractor.get_sids()
        data_frames = []
        degs = ['5deg', '10deg', '15deg']

        for sid in sids:
            for deg in degs:
                sid_files = [file for file in self.extractor.get_csv_files(directory=self.extractor.loc_dir) if file.endswith('bdf_'+deg+'_exp.csv') and sid in file]
                L = pd.read_csv(os.path.join(self.extractor.loc_dir, sid_files[0]))
                R = pd.read_csv(os.path.join(self.extractor.loc_dir, sid_files[1]))
                
                # Calculate the percentage of 'response' >= 2, ignoring NaN values
                L_percentage = (L['response'] >= 2).mean(skipna=True) * 100
                R_percentage = (R['response'] >= 2).mean(skipna=True) * 100
                
                if pd.notna(L_percentage) and pd.notna(R_percentage):
                    percentage = L_percentage + R_percentage / 2
                elif pd.notna(L_percentage):
                    percentage = L_percentage
                elif pd.notna(R_percentage):
                    percentage = R_percentage
                else:
                    percentage = float('nan')
                
                group = 'Low Vision' if sid.startswith('LV') else 'Sighted Control'

                data_frames.append({'group': group, 'sid': sid, 'deg': deg, 'percentage': percentage})

        return pd.DataFrame(data_frames, columns=['group', 'sid', 'deg', 'percentage'])
    
    def visibility(self, inclusion_thres=22/24):
        """
        Columns:
            - sid
            - include
            - group
            - eye
            - location (all/Visible/Invisible)
            - vis_area_perc # percentage of Visible area
            - illu_perc # percentage of illusory response
            - illu_mean # mean no of flashes reported

        Returns:
            pd.DataFrame: DataFrame with the visibility.
        """
        sids = self.extractor.get_sids()
        data_frame = []
        for sid in sids:
            df = pd.read_csv(os.path.join(self.extractor.csv_dir, sid+'.csv'))
            vis_L_perc = df.query("eye=='L' and visibility=='Visible'")['location'].unique().shape[0] / 24
            vis_R_perc = df.query("eye=='R' and visibility=='Visible'")['location'].unique().shape[0] / 24

            invis_L_perc = df.query("eye=='L' and visibility=='Invisible'")['location'].unique().shape[0] / 24
            invis_R_perc = df.query("eye=='R' and visibility=='Invisible'")['location'].unique().shape[0] / 24

            eyes = ['Left Eye', 'Right Eye']
            locations = ['All', 'Visible', 'Invisible']
            vis_data = {
                'Left Eye': {'All': df.query("eye=='L' and n_beep==2"),
                             'Visible': df.query("eye=='L' and n_beep==2 and visibility=='Visible'"),
                             'Invisible': df.query("eye=='L' and n_beep==2 and visibility=='Invisible'")},
                'Right Eye': {'All': df.query("eye=='R' and n_beep==2"), 
                              'Visible': df.query("eye=='R' and n_beep==2 and visibility=='Visible'"),
                              'Invisible': df.query("eye=='R' and n_beep==2 and visibility=='Invisible'")},
            }
            
            include = False if ((vis_L_perc + vis_R_perc)/2 >= inclusion_thres) and ('LV' in sid) else True
            group = 'Low Vision' if 'LV' in sid else 'Sighted Control'

            for eye in eyes:
                for location in locations:
                    # get relevant data frame
                    data = vis_data[eye][location]

                    illu_perc = data[data['response'] > 1].shape[0] / data.shape[0] if data.shape[0] > 0 else 0
                    illu_mean = data['response'].mean(skipna=True) if data.shape[0] > 0 else 0

                    data_frame.append({
                        'sid': sid,
                        'include': include,
                        'group': group,
                        'eye': eye,
                        'location': location,
                        'vis_area_perc': vis_L_perc if eye == 'Left Eye' else vis_R_perc,
                        'invis_area_perc': invis_L_perc if eye == 'Left Eye' else invis_R_perc,
                        'illu_perc': illu_perc,
                        'illu_mean': illu_mean
                    })

        # Create DataFrame
        df = pd.DataFrame(data_frame)
        return df
    
    def get_mean_sd_df(self):
        """
        Gets the MEAN and SD of flashes reported per condition and location type.

        Returns:
            pd.DataFrame
            Columns:
                - sid
                - group (Low Vision/Sighted Control)
                - eye
                - nflash
                - nbeep
                - mean
                - sd
                - location (all/vis/invis/five/ten/fifteen)
                - ntrials
        """
        # print("it works in the very beginning!")
        loc_dict = {
            'all': '_raw.csv',
            'vis': '_vis', # by_vis
            'invis': '_invis', # by_vis
            'five': '_5deg', # by_loc
            'ten': '_10deg', # by_loc
            'fifteen': '_15deg' # by_loc
        }

        conditions = {
                    ('vf', 'catch'): (0, 0),
                    ('vf', 'exp'): (1, 0),
                    ('bdf', 'catch'): (1, 1),
                    ('bdf', 'exp'): (1, 2),
                }

        sids = self.extractor.get_sids()
        # print(sids)
        raw_csvs = self.extractor.get_csv_files()
        # print(raw_csvs)
        by_vis_csvs = self.extractor.get_csv_files(directory=self.extractor.vis_dir)
        by_loc_csvs = self.extractor.get_csv_files(directory=self.extractor.loc_dir)
        data_frame = []
        for sid in sids:
            # sid = sid
            # print("Processing subject ID:", sid) # Debugging line to check which subject is being processed
            group = 'Low Vision' if 'LV' in sid else 'Sighted Control'
            for loc, suffix in loc_dict.items():
                if loc == 'all':
                    sid_files = [file for file in raw_csvs if sid in file and suffix in file]
                    files_dir = self.extractor.csv_dir
                    for task in ['vf', 'bdf']:
                        for eye in ['_L_', '_R_']:
                            sid_file = [f for f in sid_files if task in f and eye in f]
                            if not sid_file:
                                continue
                            # print(sid_file[0]) # Debugging line to check which files are being processed
                            data = pd.read_csv(os.path.join(files_dir, sid_file[0]))

                            if task == 'vf':
                                nbeep = 0
                                catch_trials = data[data['n_flash'] == 0]
                                flash_trials = data[data['n_flash'] == 1]
                                
                                self.append_data(data_frame, sid, group, eye[1], 0, nbeep, catch_trials, loc)
                                self.append_data(data_frame, sid, group, eye[1], 1, nbeep, flash_trials, loc)
                            else: # task == 'bdf'
                                nflash = 1
                                catch_trials = data[data['n_beep'] == 1]
                                illus_trials = data[data['n_beep'] == 2]

                                self.append_data(data_frame, sid, group, eye[1], nflash, 1, catch_trials, loc)
                                self.append_data(data_frame, sid, group, eye[1], nflash, 2, illus_trials, loc)

                else:
                    if loc in ['vis', 'invis']:
                        sid_files = [file for file in by_vis_csvs if sid in file and suffix in file]
                        files_dir = self.extractor.vis_dir
                    elif loc in ['five', 'ten', 'fifteen']:
                        sid_files = [file for file in by_loc_csvs if sid in file and suffix in file]
                        files_dir = self.extractor.loc_dir
                    for task in ['vf', 'bdf']:
                        for cond in ['exp', 'catch']:
                            nflash, nbeep = conditions[(task, cond)]                            
                            for eye in ['_L_', '_R_']:
                                sid_file = [f for f in sid_files if task in f and cond in f and eye in f]
                                if not sid_file:
                                    continue
                                # print(sid_file[0])
                                data = pd.read_csv(os.path.join(files_dir, sid_file[0]))
                                self.append_data(data_frame, sid, group, eye[1], nflash, nbeep, data, loc)

                if len(sid_files) == 0:
                    continue

        df = pd.DataFrame(data_frame)                
        return df

    def append_data(self, data_frame, sid, group, eye, nflash, nbeep, trials, location):
        """
        Helper function to append data to the data_frame.
        For get_mean_sd
        """
        data_frame.append({
            'sid': sid,
            'group': group,
            'eye': eye,
            'nflash': nflash,
            'nbeep': nbeep,
            'mean': trials['response'].mean(),
            'sd': trials['response'].std(),
            'location': location,
            'ntrials': trials.shape[0]
        })

    def append_data_rabbit(self, data_frame, sid, eye, resp_type, nflash, nbeep, trials, location):
        """
        Helper function to append data to data_frame
        For get_mean_sd_rabbit
        """

        data_frame.append({
            'sid': sid,
            'eye': eye,
            'resp_type': resp_type,
            'nflash': nflash,
            'nbeep': nbeep,
            'mean': trials['response'].mean(),
            'sd': trials['response'].std(),
            'location': location,
            'ntrials': trials.shape[0]
        })

    def get_mean_sd_rabbit(self):
        """
        Data directory: csv/rabit
        Gets the MEAN and SD of flashes AND beeps reported per condition and location type.

        Returns:
            pd.DataFrame
            Columns:
                - sid
                - eye
                - resp_type (flash/beep)
                - nflash
                - nbeep
                - mean
                - sd
                - location (all/ctrl/bs)
                - ntrials
        """

        all_csvs = self.extractor.get_csv_files(directory=self.extractor.rabbit_dir)

        data_frame = []

        for csv in all_csvs:
            # print("Processing file:", csv) # Debugging line to check which files are being processed
            sid = csv.split('_')[0]
            eye = csv.split('_')[-1].split('.')[0] # fix so don't include 'csv'
            data = pd.read_csv(os.path.join(self.extractor.rabbit_dir, csv))

            for resp_type, resp_code in zip(['flash', 'beep'], [1, 2]):
                resp_data = data[data['response_type'] == resp_code]
                for f in [2,3]:
                    for b in [0,2,3]:
                        for loc in ['all', 'ctrl', 'bs']:
                            if loc == 'all':
                                trials = resp_data[(resp_data['n_flash'] == f) & (resp_data['n_beep'] == b)]
                            elif loc == 'ctrl':
                                trials = resp_data[(resp_data['blindspot'] == 0) & (resp_data['n_flash'] == f) & (resp_data['n_beep'] == b)]
                            elif loc == 'bs':
                                trials = resp_data[(resp_data['blindspot'] == 1) & (resp_data['n_flash'] == f) & (resp_data['n_beep'] == b)]
                            self.append_data_rabbit(data_frame, sid, eye, resp_type, f, b, trials, loc)

        df = pd.DataFrame(data_frame)                
        return df
    

