import os
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

class FileExtractor:
    def __init__(self, base_dir=None):
        """
        Initializes the FileExtractor with directories for data processing.

        Parameters:
            base_dir (str): Base directory. If None, uses the current working directory.
        """
        self.base_dir = base_dir or os.getcwd()
        self.csv_dir = os.path.join(self.base_dir, 'csv')
        self.mat_dir = os.path.join(self.base_dir, 'mat')
        self.parent_dir = os.path.dirname(self.base_dir)
        self.plot_dir = os.path.join(self.parent_dir, 'plots')

    def get_mat_files(self, directory=None, extension='.mat', beep=False):
        """
        Gets all MAT files with a specified extension from the mat directory.

        Parameters:
            directory (str): Directory to search for MAT files. If None, defaults to mat_dir.
            extension (str): File extension to filter (default is '.mat').

        Returns:
            list: List of MAT files.
        """
        target_dir = directory or self.mat_dir
        if not os.path.exists(target_dir):
            raise FileNotFoundError(f"Directory does not exist: {target_dir}")
        
        if beep:
            return [file for file in os.listdir(target_dir) if file.endswith(extension) and 'ad' in file]
        else:
            return [file for file in os.listdir(target_dir) if file.endswith(extension) and ('vf' in file or 'df' in file)]

    def get_csv_files(self, directory=None, extension='.csv', beep=False):
        """
        Gets all CSV files with a specified extension from the csv directory.

        Parameters:
            directory (str): Directory to search for CSV files. If None, defaults to csv_dir.
            extension (str): File extension to filter (default is '.csv').

        Returns:
            list: List of CSV files.
        """
        target_dir = directory or self.csv_dir
        if not os.path.exists(target_dir):
            raise FileNotFoundError(f"Directory does not exist: {target_dir}")
        
        if beep:
            return [file for file in os.listdir(target_dir) if file.endswith(extension) and 'beep' in file and file != 'all_participants.csv']
        else:
            return [file for file in os.listdir(target_dir) if file.endswith(extension) and 'beep' not in file and file != 'all_participants.csv']
            
    def get_sids(self, group=''):
        """
        Gets all subject IDs from the data files.

        Parameters:
            group (str): Group to filter (default is '').

        Returns:
            list: List of subject IDs.
        """
        mat_files = self.get_mat_files()
        # print(f"CSV files found: {csv_files}")
        sids = list(set(
            [file.split('_')[0] for file in mat_files if group in file]
        ))
        return sorted(sids, key=lambda x: int(x[2:]))
    
    def get_N(self, group = ''):
        """
        Gets the number of subjects in the data files.

        Parameters:
            group (str): Group to filter (default is '').

        Returns:
            N (int): Number of subjects.
        """
        return len(self.get_sids(group))
    