# Generic imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ranksums, ttest_1samp, wilcoxon
from statsmodels.stats.descriptivestats import sign_test

# Custom imports
import importlib
import file_extractor
importlib.reload(file_extractor)
from file_extractor import FileExtractor
extractor = FileExtractor()

import get_df
importlib.reload(get_df)
from get_df import get_df
data_frame = get_df()

class Plots:
    def __init__(self):
        self.theta_array = np.deg2rad(
            [270]*3 + # 1,2,3
            [90]*3 + # 4,5,6
            [0]*3 + # 7,8,9
            [180]*3 + # 10,11,12
            [315]*3 + # 13,14,15
            [135]*3 + # 16,17,18
            [45]*3 + # 19,20,21
            [225]*3 # 22,23,24
            )
        self.r_array = np.array([5, 10, 15]*8)
        self.loc_x_array = self.r_array * np.cos(self.theta_array)
        self.loc_y_array = self.r_array * np.sin(self.theta_array)

    def polar_data_for_plotting(self, n_flash, n_beep, vmax, eye, type='', sid=None, group=None, debug = False):
        """
        Create data for plotting polar plots

        Returns:
        color_values (float): list of color values (1 x 24)
        """

        if type == 'individual':
            df = pd.read_csv(os.path.join(extractor.csv_dir, sid+'.csv'))

            # sort by location, only keep rows with n_flash == 1 or n_beep == 2
            df = df.query('n_flash == @n_flash and n_beep == @n_beep and eye == @eye').sort_values(by='location')
            df = df.groupby('location').mean(numeric_only=True)
            print(df) if debug else None

            # Normalize the response values
            norm = mcolors.Normalize(vmin=0, vmax=vmax)
            cmap = cm.inferno  # Choose a colormap
            
            # Normalize the response values
            color_values = cmap(norm(df['response'].values))  # Map all response values to colors
            return color_values, df
        else:
            sids = extractor.get_sids(group=group)
            group_data = {location: [] for location in range(1,len(self.theta_array)+1)}            
            for sid in sids:
                df = pd.read_csv(os.path.join(extractor.csv_dir, sid+'.csv'))
                df = df.query('n_flash == @n_flash and n_beep == @n_beep and eye == @eye').sort_values(by='location')
                df = df.groupby('location').mean(numeric_only=True)
                print(df) if debug else None
                # Append each subject's response to group_data by location
                for location, response in df['response'].items():
                    group_data[location].append(response)
                    print(group_data) if debug else None

            # Calculate the mean response for each location
            if n_beep == 0: # flash detection task
                avg_responses = {location: np.count_nonzero(responses) / len(responses) for location, responses in group_data.items()}
            else:
                avg_responses = {location: np.mean(responses) for location, responses in group_data.items()}

            # Normalize the average response values
            norm = mcolors.Normalize(vmin=0, vmax=vmax)
            cmap = cm.inferno

            color_values = [cmap(norm(avg_responses[loc])) for loc in avg_responses]
            return color_values, avg_responses
    
    def polar(self, show_value=True, fontsize=16, marker_size=500, type='', sid=None, group=None, title=None):
     
        fig, axs = plt.subplots(2, 2, subplot_kw={'projection': 'polar'}, figsize=(10, 10))
        cmap = cm.inferno  # Choose a colormap
        i = 0
        for n_flash, n_beep, vmax in zip([1, 1], [0, 2], [1, 2]):
            for eye in ['L', 'R']:                
                # Normalize the response values
                color_values, avg_resp = self.polar_data_for_plotting(n_flash, n_beep, vmax, eye, type=type, sid=sid, group=group)
                responses = avg_resp['response'].values if type == 'individual' else avg_resp.values()
                ax = axs.flat[i]  # Use the appropriate subplot

                # Plot each point with color coding and add response values as labels for debugging
                for location, (theta, radius, color, response) in enumerate(zip(self.theta_array, self.r_array, color_values, responses)):
                    ax.scatter(theta, radius, s=marker_size, color=color, edgecolors='black', alpha=1)                
                    # Add debugging text: Display response value near each point
                    value_text_color = 'white' if response < 0.5 and n_beep == 0 else 'black'
                    value_text_color = 'white' if response < 1 and n_beep == 2 else value_text_color
                    ax.text(theta, radius, f"{response:.1f}", ha='center', va='center', fontsize=8, color=value_text_color) if show_value else None

        
                ax.set_xticklabels([])
                ax.set_ylim(0, 18)

                ## y tick labels            
                ax.set_yticklabels([])
                # Add custom annotations for the y-tick labels at 5, 10, and 15 degrees along the 0 degree axis
                radii = [5, 10, 15]
                for radius in radii:
                    ax.text(-0.18, radius+1.2, f"{radius}°", ha='left', va='top', fontsize=14, weight='bold', 
                            transform=ax.transData, clip_on=True) 
                    
                # Remove gridlines and solid circle
                ax.grid(False)
                ax.spines['polar'].set_visible(False)

                i += 1

        axs[0, 0].set_ylabel('Visual Flash Detection Task', fontsize=22)
        axs[1, 0].set_ylabel('Illusory Double Flash Task', fontsize=22)
        axs[1, 0].set_xlabel('Left Eye', fontsize=20)
        axs[1, 1].set_xlabel('Right Eye', fontsize=20)

        # Add color bars for top and bottom rows
        norm_top = mcolors.Normalize(vmin=0, vmax=1)
        norm_bottom = mcolors.Normalize(vmin=0, vmax=2)
        
        cbar_top = fig.colorbar(cm.ScalarMappable(norm=norm_top, cmap=cmap), 
                                ax=axs[0, :], orientation='vertical', 
                                fraction=0.02, pad=0.01)
        cbar_top.set_ticks([0, 1])
        cbar_top.set_ticklabels(['0%\nFlash\nDetection', 
                                 '100%\nFlash\nDetection'],
                                 fontsize=14)

        cbar_bottom = fig.colorbar(cm.ScalarMappable(norm=norm_bottom, cmap=cmap), 
                                   ax=axs[1, :], orientation='vertical', 
                                   fraction=0.02, pad=0.01)
        # cbar_bottom.set_ticks([0, 1, 2, 3, 4])
        # cbar_bottom.set_ticklabels(['0 Flash\nPerceived', 
        #                             '1 Flash\nPerceived', 
        #                             '2 Flashes\nPerceived', 
        #                             '3 Flashes\nPerceived',
        #                             '4 Flashes\nPerceived'],
        #                             fontsize=14)
        cbar_bottom.set_ticks([0, 2])
        cbar_bottom.set_ticklabels(['0%\nIllusory\nPerception', 
                                    '100%\nIllusory\nPerception'],
                                    fontsize=14)

        plt.tight_layout(rect=[0, 0, 0.85, 0.96])
        # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        fig.suptitle(title, fontsize=30)

        return fig, axs

    def polar_individual(self, sid):
        fig, axs = self.polar(type='individual', sid=sid, title=sid)
        plt.show()

        # save figure
        fig.savefig(f'{extractor.plot_dir}/polar/indiv/{sid}_polar.png', 
                    dpi=300, bbox_inches='tight')
    
    def polar_group(self):
        for title, group in zip(['Low Vision', 'Sighted Control'], ['LV', 'SV']):
            fig, axs = self.polar(type='group', group=group, title=title)
            plt.show()

            # save figure
            fig.savefig(f'{extractor.plot_dir}/polar/{title}_polar.png', 
                        dpi=300, bbox_inches='tight')

    def polar_ax(self, ax, task, eye, group, marker_size=300, show_value=True):
        if task=='vf':
            n_flash = 1
            n_beep = 0
            vmax = 1
        elif task=='bdf':
            n_flash = 1
            n_beep = 2
            vmax = 2

        color_values, avg_resp = self.polar_data_for_plotting(n_flash, n_beep, vmax, eye, group=group)
        responses = avg_resp.values()

        # Plot each point with color coding and add response values as labels for debugging
        for location, (theta, radius, color, response) in enumerate(zip(self.theta_array, self.r_array, color_values, responses)):
            ax.scatter(theta, radius, s=marker_size, color=color, edgecolors='black', alpha=1)
            ax.grid(False)
            ax.spines['polar'].set_visible(False)
            ax.set_xticklabels([])
            ax.set_ylim(0, 18)
            ax.set_yticklabels([])        

            value_text_color = 'white' if response < 0.5 and n_beep == 0 else 'black'
            value_text_color = 'white' if response < 1 and n_beep == 2 else value_text_color
            ax.text(theta, radius, f"{response:.1f}", ha='center', va='center', fontsize=6, color=value_text_color) if show_value else None

            ax.set_xlabel('Left Eye', fontsize=12) if eye=='L' and task=='bdf' else None
            ax.set_xlabel('Right Eye', fontsize=12) if eye=='R' and task=='bdf' else None

    def polar_schematics(self, ax, marker_size=100, color_code=False):
        color_map = {5: '#f7acbf', 10: '#7570b3', 15: '#1b9e77'} # Colorblind-safe palette  
        if color_code:            
            for r, color in color_map.items():
                mask = self.r_array == r
                
                ax.scatter(
                    self.theta_array[mask], 
                    self.r_array[mask], 
                    s=marker_size, 
                    c=color,
                    label=f"{r}°" # This label is used by the legend
                )
            ax.legend(title="Eccentricity", bbox_to_anchor=(0, 0.5))
        else:
            ax.scatter(self.theta_array, self.r_array, s=marker_size, color='black', edgecolors='black', alpha=1)
            # Add custom annotations for the y-tick labels at 5, 10, and 15 degrees along the 0 degree axis
            radii = [5, 10, 15]
            for radius in radii:
                ax.text(-0.18, radius+1, f"{radius}°", ha='left', va='top', fontsize=10, weight='bold', 
                        transform=ax.transData, clip_on=True) 

        ax.grid(False)
        ax.spines['polar'].set_visible(False)
        ax.set_xticklabels([])
        ax.set_ylim(0, 18)
        ax.set_yticklabels([])
        

    def condition_schematics(self, ax, n_flash, n_beep):
        vis_len = 17 # ms
        aud_len = 7  # ms
        aud_gap = 58 # ms
        height = 2

        onset_times = [5, 5 + aud_len + aud_gap]
        time = np.arange(0, aud_len*2 + aud_gap + 15)

        y_flash = np.zeros_like(time)
        y_beep = np.zeros_like(time)

        for i in range(n_flash):
            start_time = onset_times[i]
            end_time = start_time + vis_len
            flash_on = (time >= start_time) & (time < end_time)
            y_flash[flash_on] = height

        for i in range(n_beep):
            start_time = onset_times[i]
            end_time = start_time + aud_len
            beep_on = (time >= start_time) & (time < end_time)
            y_beep[beep_on] = height

        ax.step(time, y_flash + 3, label='Flash', color='#9b72cf')
        ax.step(time, y_beep, label='Beep', color='#ffc60a')
        ax.set_ylim([-1, 6])
        ax.set_yticks([])
        ax.set_xlabel('Time (ms)') if n_beep>0 else None
        ax.set_xticks(np.arange(0, 76, 25)) if n_beep>0 else ax.set_xticks([])
        # ax.set_xlim([0, 100])

    def LE_RE_visible_area(self, df, ax, exclude=False):
        labelfontsize = 9
        titlefontsize = 11
        # Here we're plotting all locations
        df = df[df['location'] == 'All']
        df = df[df['include'] == True] if exclude else df

        jitter_strength = 0.02
        df['vis_perc_jitter'] = df['vis_area_perc'] + np.random.uniform(-jitter_strength, jitter_strength, len(df))
        df['vis_perc_jitter'] = df['vis_perc_jitter'].clip(upper=1.0)
        df_L = df[df['eye'] == 'Left Eye']
        df_R = df[df['eye'] == 'Right Eye']
        df_L = df_L.sort_values('sid').reset_index(drop=True)
        df_R = df_R.sort_values('sid').reset_index(drop=True)

        group_counts = df_L['group'].value_counts().to_dict()
        df_L['group_with_N'] = df_L['group'].map(lambda g: f"{g} (N={group_counts[g]})")

        sns.scatterplot(
            x=df_R['vis_perc_jitter'],
            y=df_L['vis_perc_jitter'],
            hue=df_L['group_with_N'],
            alpha=0.7,
            ax=ax
        )
        # Add a reference line for equality
        min_val, max_val = -0.1, 1.1
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.tick_params(labelsize=labelfontsize)
        ax.plot([min_val, max_val], [min_val, max_val], color='grey', linestyle='--')

        ax.set_xlabel('Right Eye % Visible Area', fontsize=labelfontsize)
        ax.set_ylabel('Left Eye % Visible Area', fontsize=labelfontsize)
        ax.set_title('% Visible Area, \nLeft vs Right Eye', fontsize=titlefontsize)
        ax.legend(title=None, fontsize=labelfontsize)

    def illusion_strength_vs_visibility(self, df, ax, test=pearsonr, exclude=False):

        test = pearsonr
        labelfontsize = 8
        titlefontsize = 11

        df = df[df['location'] == 'All']
        df = df[df['include'] == True] if exclude else df

        df_L = df[df['eye'] == 'Left Eye']
        df_R = df[df['eye'] == 'Right Eye']
        df_L = df_L.sort_values('sid').reset_index(drop=True)
        df_R = df_R.sort_values('sid').reset_index(drop=True)

        df_L['diff_vis'] = df_R['vis_area_perc'] - df_L['vis_area_perc']
        df_L['diff_illu'] = df_R['illu_perc'] - df_L['illu_perc']

        group_counts = df_L['group'].value_counts().to_dict()
        df_L['group_with_N'] = df_L['group'].map(lambda g: f"{g} (N={group_counts[g]})")

        df = pd.DataFrame(df_L)

        # Initialize an empty dictionary to store correlation results
        correlation_results = {}

        # Loop through each group to calculate correlations
        print("Test used: Perason's correlation" if test == pearsonr else "Test used: Spearman's correlation")
        for group in df['group'].unique():
            group_data = df[df['group'] == group]
            r, p = test(group_data['diff_vis'], group_data['diff_illu'])
            correlation_results[group] = (r, p)

            # Print results for each group
            print(f"Group: {group}")
            print(f"  r = {r:.2f}")
            print(f"  p-value: {p:.3e}")

        # Plotting
        for name, group_df in df_L.groupby('group_with_N'):
            sns.regplot(
                data=group_df,
                x='diff_vis',
                y='diff_illu',
                scatter=True,
                ax=ax,
                label=name,
                scatter_kws={'alpha': 0.7},
                line_kws={'linewidth': 1.5}
            )
        ax.legend(title=None, fontsize=labelfontsize)
        # Add a diagonal reference line
        min_val, max_val = -1, 1
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.tick_params(labelsize=labelfontsize)

        # Annotate correlation results for each group
        # for group, (r, p) in correlation_results.items():
        #     plt.text(
        #         0.52, 0.2 - 0.05 * list(df['group'].unique()).index(group),  # Adjust position for each group
        #         f"{group}: $r$ = {r:.2f}, $p$ = {p:.2f}" if p >= 0.01 else f"{group}: $r$ = {r:.2f}, $p$ < 0.01",
        #         transform=plt.gca().transAxes
        #     )

        ax.axhline(0, color='black', linestyle='-', linewidth=0.7)  # Horizontal symmetry
        ax.axvline(0, color='black', linestyle='-', linewidth=0.7)  # Vertical symmetry
        ax.set_xlabel(r'LE Stronger $\leftarrow$    Visibility    $\rightarrow$ RE Stronger', 
                   fontsize=labelfontsize)
        ax.set_ylabel(r'LE Stronger $\leftarrow$    Illusion Strength   $\rightarrow$ RE Stronger', 
                   fontsize=labelfontsize)
        ax.set_title('Relationship between Visibility \nand Illusion Strength', fontsize=titlefontsize)

    def add_significance_star(self, ax, x1, x2, y, p_val, num_comparisons=4):
        """Adds a significance star annotation to the plot."""
        if p_val < 0.001/num_comparisons:
            star = '***'
        elif p_val < 0.01/num_comparisons:
            star = '**'
        elif p_val < 0.05/num_comparisons:
            star = '*'
        else:
            star = 'n.s.'

        ax.text((x1 + x2) / 2, y, star, ha='center', va='bottom', fontsize=12)

    def perc_illusion(self, df, axs, fig=None, exclude=False):
        
        labelfontsize = 10
        titlefontsize = 11

        df = df[df['include'] == True] if exclude else df
        visible_df = df[df['location'] == 'Visible']
        invisible_df = df[df['location'] == 'Invisible']
        combined_data = pd.concat([visible_df, invisible_df], ignore_index=True)
        palette = sns.color_palette('deep', n_colors=2)  # One color per condition
        # fig, axes = plt.gca(2, 2, figsize=(8,8), sharey=True)
        groups = ['Low Vision', 'Sighted Control']
        eyes = ['Left Eye', 'Right Eye']

        k=0
        for i, group in enumerate(groups):
            for j, eye in enumerate(eyes):
                plot_data = combined_data[(combined_data['group'] == group) & (combined_data['eye'] == eye)]
                ax = axs[k]
                sns.stripplot(
                    data=plot_data,
                    x='location',
                    y='illu_perc',
                    dodge=True,
                    color=palette[i],
                    ax=ax
                )

                if len(visible_df) == len(invisible_df):
                    vis_perc_data = visible_df[(visible_df['group'] == group) 
                                               & (visible_df['eye'] == eye)]['illu_perc'].tolist()
                    invis_perc_data = invisible_df[(invisible_df['group'] == group) 
                                                   & (invisible_df['eye'] == eye)]['illu_perc'].tolist()
                    stat, p_val = sign_test(vis_perc_data, invis_perc_data)
                    self.add_significance_star(ax, x1=0, x2=1, y=1, p_val=p_val)

                for sid in plot_data['sid'].unique():
                    subject_data = plot_data[plot_data['sid'] == sid]

                    if len(subject_data['location'].unique()) == 2:
                        ax.plot(
                            [.7, .3],
                            subject_data.groupby('location')['illu_perc'].mean(),
                            linestyle='-',
                            color=palette[i],
                            alpha=0.5
                        )

                ax.set_title(eye, fontsize=titlefontsize) if i == 0 else None
                ax.set_ylabel("% Illusory Flash Perception" if j == 0 else "", fontsize=labelfontsize) 
                ax.set_xticklabels(['Visible', 'Invisible'], fontsize=labelfontsize)
                ax.set_xlabel("Locations" if i == 1 else "", fontsize=labelfontsize)
                ax.tick_params(labelsize=labelfontsize)
                ax.set_ylim(-0.1, 1.1)
                k += 1
    
    def add_stat_annotation(self, ax, data, x, y, hue, pairs, test, palette):
        for pair in pairs:
            group1 = data[(data[x] == pair[0][0]) & (data[hue] == pair[0][1])][y]
            group2 = data[(data[x] == pair[1][0]) & (data[hue] == pair[1][1])][y]
            stat, p = test(group1, group2)
            print(pair, stat, p)
            # display(stat, p)
            annot = '*' if p < 0.05 else 'n.s.'
            annot = '**' if p < 0.01 else annot
            annot = '***' if p < 0.001 else annot
            col = palette[data[hue].unique().tolist().index(pair[0][1])]
            ax.plot(pair[2][0], pair[2][1], lw=1.5, c=col)
            ax.text(0.5, pair[2][1][0]+0.01, annot, ha='center', va='bottom', color=col)

    def add_stat_annotation_3_columns(self, ax, data, x, y, hue, pairs, test, palette):
        for pair in pairs:
            group1 = data[(data[x] == pair[0][0]) & (data[hue] == pair[0][1])][y]
            group2 = data[(data[x] == pair[1][0]) & (data[hue] == pair[1][1])][y]
            stat, p = test(group1, group2)
            # display(stat, p)
            annot = '*' if p < 0.05 else 'n.s.'
            annot = '**' if p < 0.01 else annot
            annot = '***' if p < 0.001 else annot
            col = palette[data['group'].unique().tolist().index(pair[0][1])]
            ax.plot(pair[2][0], pair[2][1], lw=1.5, c=col)
            ax.text(np.mean(pair[2][0]), pair[2][1][0], annot, ha='center', va='bottom', color=col)

    def by_task(self, vf_data, bdf_data, bdf_catch_data, test=sign_test, exclude=False):
        vf_data['data_type'] = 'Flash Detection Task (0 beep)'
        bdf_catch_data['data_type'] = 'Double Flash Task (1 beep)'
        bdf_data['data_type'] = 'Double Flash Task (2 beeps)'
        combined_data = pd.concat([vf_data, bdf_catch_data, bdf_data], ignore_index=True)
        combined_data['group'] = combined_data['sid'].apply(lambda x: 'Low Vision' if x.startswith('LV') else 'Sighted Control')
        combined_data = combined_data[combined_data['include'] == True] if exclude else combined_data
        
        palette = sns.color_palette('deep', n_colors=combined_data['group'].nunique())
        g = sns.catplot(
            data=combined_data,
            x='data_type',
            y='avg_flash',
            hue='group',
            kind='strip',
            palette=palette,
            height=6,
            aspect=1.2,
            zorder=1
        )
        x_positions = [ [0.2, 0.8], [1.2, 1.8] ]
        combos = [['Flash Detection Task (0 beep)', 'Double Flash Task (1 beep)'], 
                ['Double Flash Task (1 beep)', 'Double Flash Task (2 beeps)'] ]
        for x_pos, tasks in zip(x_positions, combos):
            for sid in combined_data['sid'].unique():
                subject_data = combined_data[combined_data['sid'] == sid]
                combo_data = subject_data[subject_data['data_type'].isin(tasks)]
                
                group = subject_data['group'].iloc[0]
                color = palette[combined_data['group'].unique().tolist().index(group)]
                
                plt.plot(
                    x_pos,
                    combo_data['avg_flash'],
                    linestyle='-',
                    color=color,
                    alpha=0.5
                )

        mean_data = combined_data.groupby(['group', 'data_type'])['avg_flash'].mean().reset_index()
        group_counts = combined_data.groupby('group')['sid'].nunique().to_dict()

        x_positions = [ [0.8, 0.2], [1.2, 1.8] ]
        for x_pos, tasks in zip(x_positions, combos):
            for group in mean_data['group'].unique():
                group_mean_data = mean_data[mean_data['group'] == group]
                combo_data = group_mean_data[group_mean_data['data_type'].isin(tasks)]                
                color = palette[mean_data['group'].unique().tolist().index(group)]
                plt.plot(
                    x_pos,
                    combo_data['avg_flash'],
                    linestyle='--',
                    color=color,
                    linewidth=3.5,
                    label=f'{group} Mean'
                )

        g.set_axis_labels("Task", "Average Flashes Perceived", 
                          fontsize=16)
        g.fig.suptitle("Average Number of Flashes Percevied by Task and Condition", 
                       fontsize=18)

        new_labels = [f'{label.get_text()} (N={group_counts[label.get_text()]})' for label in g._legend.texts]
        for t, label in zip(g._legend.texts, new_labels):
            t.set_text(label)
        g._legend.set_title('Group')
        g._legend.set_bbox_to_anchor((0.35, 0.75))
        # add boder to legend
        g._legend.get_frame().set_linewidth(1)
        pairs = [
            (('Flash Detection Task (0 beep)', 'Low Vision'), ('Double Flash Task (1 beep)', 'Low Vision'), ([0.4, 0.6], [1.8, 1.8])),
            (('Flash Detection Task (0 beep)', 'Sighted Control'), ('Double Flash Task (1 beep)', 'Sighted Control'), ([0.4, 0.6], [1.7, 1.7])),
            (('Double Flash Task (1 beep)', 'Low Vision'), ('Double Flash Task (2 beeps)', 'Low Vision'), ([1.4, 1.6], [2.1, 2.1])),
            (('Double Flash Task (1 beep)', 'Sighted Control'), ('Double Flash Task (2 beeps)', 'Sighted Control'), ([1.4, 1.6], [2, 2])),
            (('Flash Detection Task (0 beep)', 'Low Vision'), ('Double Flash Task (2 beeps)', 'Low Vision'), ([0.4, 1.6], [2.4, 2.4])),
            (('Flash Detection Task (0 beep)', 'Sighted Control'), ('Double Flash Task (2 beeps)', 'Sighted Control'), ([0.4, 1.6], [2.3, 2.3]))
        ]
        self.add_stat_annotation_3_columns(g.ax, 
                                           combined_data, 
                                           'data_type', 
                                           'avg_flash', 
                                           'group', 
                                           pairs, 
                                           test, 
                                           palette)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'{extractor.plot_dir}/avg_flashes_perceived.png',
                    dpi=300, bbox_inches='tight')
        plt.show()









