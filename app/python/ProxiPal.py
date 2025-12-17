# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 14:02:26 2023

@author: smith.j
"""
import os
import sys
from pathlib import Path
import platform
import re
import uuid
from io import StringIO
from datetime import datetime
from typing import Tuple, Optional, Literal, List, Dict, Union
# from typing import Union
# from typing import Dict
# from typing import List
import contextlib
import io
from contextlib import contextmanager
from itertools import combinations

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
from scipy.optimize import curve_fit

# from scipy.optimize import fsolve
from scipy.stats import norm
from scipy import stats
from sklearn.metrics import r2_score
import pymannkendall as mk
from outliers import smirnov_grubbs as grubbs
import math
from scipy.stats import spearmanr
from scipy.stats import pearson3

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
import seaborn as sns

import rdmlpython as rdml
# import warnings
# import qgrid

now = datetime.now()
print("OS:\t\t\t\t", platform.platform())
print("Python version:\t\t", sys.version)

# Administrators should set their own password to higher level Jupyter notebook functions
proxipal_password = 'admin'

# Find base path first
required_folders = {'data', 'python', 'samples', 'templates', 'quality'}
cwd = Path(os.getcwd())
base_path = next((p for p in cwd.parents if required_folders <= {d.name for d in p.iterdir() if d.is_dir()}), None)

# cwd  = Path(os.getcwd())
# base_path = Path(os.path.join(*cwd.parts[:cwd.parts.index('python')]))
print('Date and Time:\t\t', now.strftime('%Y-%m-%d %H:%M:%S'))
print('cwd:\t\t', cwd) 
print('base_path:\t\t', base_path)
python_folder = base_path / 'python'
print('python_folder:\t\t', python_folder)

# Define major directories
templates_folder = base_path / 'templates'
print('templates_folder:\t', templates_folder)
data_folder = base_path / 'data'
print('data_folder:\t\t', data_folder)
samples_folder = base_path / 'samples'
print('samples_folder:\t\t', samples_folder)
quality_folder = base_path / 'quality'
print('quality_folder:\t\t', quality_folder)
user_downloads = base_path / 'user_downloads'
print('user_downloads:\t\t', user_downloads)

# make the necessary folders, should they not exist already
folder_paths = [templates_folder, data_folder, samples_folder, quality_folder, user_downloads]
for folder_path in folder_paths:
    path = Path(folder_path)
    # Check if the folder exists
    if not path.exists():
        # Create the folder
        path.mkdir(parents=True, exist_ok=True)

@contextmanager
def suppress_print():
    "Used to suppress uneanted print statements"
    
    original_stdout = sys.stdout  # store the original stdout
    try:
        sys.stdout = open(os.devnull, 'w')  # redirect stdout to null
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout  # restore the original stdout

def walk_directory(path: Path, obj: bool = False) -> list:
    ''' 
    Returns a list of all folder and file paths in the specified directory and all its subdirectories.

    Parameters
    ----------
    path : Path
        A pathlib object that represents the directory path to walk.
    obj : bool, optional
        If False (default), the function returns the items as strings.
        If True, the function returns the items as pathlib objects.

    Returns
    -------
    list
        A list of folder and file paths in the specified directory and all its subdirectories.

    '''
    # Initialize an empty list to store the directory paths
    dir_list = []

    # Walk through the specified directory and all its subdirectories
    for subpath in path.rglob('*'):
        # If obj is False, append the directory paths as strings
        if obj == False:
            dir_list.append(str(subpath))
        # If obj is True, append the directory paths as pathlib objects
        elif obj == True:
            dir_list.append(subpath)
    
    # Return the list of directory paths
    return dir_list

#### USAGE:
# dir_list = walk_directory(base_path, obj = True)


def get_file_and_parents(file_path: Path, num_parents: int = 2) -> str:
    '''
    Accepts a pathlib object and returns a string representing the destination name and the parent folders up to a user-defined depth.

    Parameters
    ----------
    file_path : Path
        A pathlib object that represents the file path.
    num_parents : int, optional
        The number of parent folders to include in the output string. Default is 2.

    Returns
    -------
    str
        A string representing the destination name and the parent folders up to the specified depth.

    '''
    # Get the file name from the file path
    file_name = file_path.name
    
    # Get the parent folders up to the specified depth
    parent_folders = file_path.parts[-(num_parents + 1):-1] if len(file_path.parts) > num_parents else file_path.parts[:-1]
    
    # Join the parent folders and file name into a string
    destination = "/".join(parent_folders) + "/" + file_name
    
    # Return the destination string
    return destination


def find_matched_filenames(path: Path, native_format: str = '.eds', export_format: str = '.txt', read_export: bool = True) -> tuple:
    '''
    Identifies different file types with the same stem from a directory path and all its subdirectories.

    Parameters
    ----------
    path : Path
        A pathlib object that represents the directory path to search.
    native_format : str, optional
        A string that specifies the native file format to search for. Default is '.eds'.
    export_format : str, optional
        A string that specifies the export file format to search for. Default is '.txt'.
    read_export : bool, optional
        A boolean flag that specifies whether to read the contents of the export files. If True (default), the function returns a dictionary with file paths and contents. If False, the function returns a dummy dictionary with an 'empty' key and a message.

    Returns
    -------
    tuple
        A tuple containing two items:
        - export_file_list: a list of name-matched export files as pathlib objects
        - export_dict: a dictionary with export file paths and contents (if read_export is True) or a dummy dictionary with a message (if read_export is False)

    '''
    # Get a list of all the native files in the directory and its subdirectories
    native_file_list = []
    for f in Path(path).glob('**/*' + native_format):
        native_file_list.append(f)

    # Get a list of all the export files with the same stem as the native files
    export_file_list = [p.with_suffix(export_format) for p in native_file_list]

    # Read the contents of the export files and store them in a dictionary
    export_dict = {}
    if read_export == True:
        for file_path in export_file_list:
            try:
                # Check if the file exists
                if not file_path.is_file():
                    print(f"File not found: {file_path}")
                    continue
                
                # Reading the file based on its format
                if export_format == '.txt':
                    with open(file_path, 'r') as file:
                        file_contents = file.read()
                        export_dict[get_file_and_parents(file_path)] = file_contents
                        
                elif export_format == '.csv':
                    export_dict[get_file_and_parents(file_path)] = pd.read_csv(file_path, header=None)

            except PermissionError:
                print(f"Insufficient permissions to read file: {file_path}")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    # If read_export is False, return a dummy dictionary with a message
    elif read_export == False:
        export_dict['empty'] = 'User has specified "read_export = False" when calling function find_matched_filenames().'
    
    # Return the export file list and dictionary
    return export_file_list, export_dict


def review_matched_filenames(eds2txt_match_dict, eds2csv_match_dict):
    
    file_paths = list(eds2txt_match_dict.keys()) + list(eds2csv_match_dict.keys())
    
    # Split the file paths into base and extension
    base_ext_pairs = [(fp.rsplit('.', 1)[0], fp.rsplit('.', 1)[1]) for fp in file_paths]
    
    # Create a DataFrame from the base-ext pairs
    df = pd.DataFrame(base_ext_pairs, columns=['Base', 'Ext'])
    
    # Pivot the DataFrame so that 'txt' and 'csv' are separate columns
    df_pivot = df.pivot(index='Base', columns='Ext', values='Ext')
    
    # Reset index to convert 'Base' into a column
    df_pivot.reset_index(inplace=True)
    
    # Split the 'Base' column into 3 new columns
    df_pivot[['experiment', 'analysis', 'eds filename']] = df_pivot['Base'].str.split('/', expand=True)
    
    # Convert 'txt' and 'csv' columns into boolean
    df_pivot['txt'] = df_pivot['txt'].notna()
    df_pivot['csv'] = df_pivot['csv'].notna()
    
    # Drop the 'Base' column
    df_pivot.drop('Base', axis=1, inplace=True)
    
    # Add a new column 'path_key' that joins 'experiment', 'analysis' and 'eds filename' columns with "/"
    df_pivot['path_key'] = df_pivot[['experiment', 'analysis', 'eds filename']].apply(lambda x: '/'.join(x), axis=1)
    
    # Reorder columns
    df_review = df_pivot[['experiment', 'analysis', 'eds filename', 'txt', 'csv', 'path_key']]
    
    return df_review


#### USAGE:
# eds2txt_match_list, eds2txt_match_dict = find_matched_filenames(base_path, read_export = True)
# eds2csv_match_list, eds2csv_match_dict = find_matched_filenames(base_path, native_format = '.eds', export_format = '.csv', read_export = True)


def extract_instr_params(string_value: str) -> pd.Series:
    '''  
    Extracts instrument parameters from exported EDS TXT files.

    Parameters
    ----------
    string_value : str
        A string that represents the contents of an EDS TXT file.

    Returns
    -------
    pd.Series
        A pandas Series that contains the instrument parameters extracted from the EDS TXT file. The index of the Series is the instrument parameter (string) and the values are the instrument values (string or datetime.datetime object where appropriate).

    '''
    
    data_dict = {}

    # Define regular expression patterns to match datetime formats
    pattern_1 = r'^\d{2}-\d{2}-\d{4}$'
    pattern_2 = r'^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}) (AM|PM) [A-Z]{2,5}$'

    # Iterate over each line in the EDS TXT file
    for line in string_value.splitlines():
        # Check whether the line starts with "*"
        if line.startswith('*'):
            # Extract the key and value from each line
            key = line.split('* ')[1].split(' =')[0].replace(' ', '')
            value = line.split('= ')[1].strip()

            # Check if value matches datetime formats and convert it to a datetime.datetime object
            if re.match(pattern_1, value):
                value = datetime.strptime(value, '%m-%d-%Y')
            elif re.search(pattern_2, value):     
                match = re.match(pattern_2, value)
                value = match.group(1)            
                value = datetime.strptime(value, '%m-%d-%Y %H:%M:%S')
            elif 'Comparative C' in value:
                value = 'Comparative Ct'

            # Add the key-value pair to the dictionary
            data_dict[key] = value

    # Convert the dictionary to a pandas Series and return it
    return pd.Series(data_dict)

# USAGE
# eds_txt = eds2txt_match_dict['230112_e24_kruti/230410/2023_01_12 Expt24 endoNfL_ICPLA_36cap_79SLC171SLC2_LPS_NoLPS_DhP_RMHLot2.txt']
# instr_params = extract_instr_params(eds_txt)


def extract_usr_vals(df: pd.DataFrame) -> pd.Series:
    '''
    Extracts usr designated values and manual std curve calculations from a development csv table

    Parameters
    ----------
    df : DataFrame
        The csv file saved from an excel template used for experimental planning

    Returns
    -------
    usr_series : series
        All user designated values as a series

    '''
    # Extract user-defined values
    header = df.loc[0:3, 0].tolist()
    header_val = df.loc[0:3, 2].tolist()
    
    # Extract var descriptions
    vars_list = df.loc[24:27, 22].tolist()
    vars_val = [str(i).replace('nan','none') for i in  df.loc[24:27, 24].tolist()]
    
    # Extract manual linear fit
    usr_lin_var1 = ['usr_grad_std1', 'usr_y-int_std1', 'usr_R2_std1', 'usr_PCReff_std1']
    usr_lin_var1_val = df.loc[52:55, 24].tolist()
    usr_lin_var2 = ['usr_grad_std2', 'usr_y-int_std2', 'usr_R2_std2', 'usr_PCReff_std2']
    usr_lin_var2_val = df.loc[52:55, 25].tolist()
    usr_lin_var3 = ['usr_grad_std3', 'usr_y-int_std3', 'usr_R2_std3', 'usr_PCReff_std3']
    usr_lin_var3_val = df.loc[52:55, 26].tolist()
    usr_lin_var4 = ['usr_grad_std4', 'usr_y-int_std4', 'usr_R2_std4', 'usr_PCReff_std4']
    usr_lin_var4_val = df.loc[52:55, 27].tolist()
    
    # extract experiment plan
    # find the last row with a string entry in column 36
    last_row = df.loc[df[36].apply(lambda x: isinstance(x, str))].index[-1]
    # extract the string entry from each row and concatenate into a single string with '\n'
    expt_plan = '\n'.join(df.loc[4:last_row, 36].apply(lambda x: str(x)).tolist())
    
    # Build usr series
    series_val = header_val + vars_val + usr_lin_var1_val + usr_lin_var2_val + usr_lin_var3_val + usr_lin_var4_val + [expt_plan]
    series_index = header + vars_list + usr_lin_var1 + usr_lin_var2 + usr_lin_var3 + usr_lin_var4 + ['experiment_plan']
    usr_series = pd.Series(series_val, index = series_index)

    return usr_series

# USAGE
# df = eds2csv_match_dict['230112_e24_kruti/230410/2023_01_12 Expt24 endoNfL_ICPLA_36cap_79SLC171SLC2_LPS_NoLPS_DhP_RMHLot2.csv']
# usr_vals = extract_usr_vals(df)


def extract_csv_subset(df: pd.DataFrame, iloc: list[int] = [4,101,0,21]) -> pd.DataFrame:
    '''
    Accepts the developer csv template and extracts the main table as a smaller dataframe
    
    Parameters
    ----------
    df : pd.DataFrame
        The csv file saved from an excel template used for experimental planning
    iloc : list[int], optional
        The slice positions to extract the table, by default [4,101,0,21]
    
    Returns
    -------
    pd.DataFrame
        A smaller dataframe extracted from the input dataframe
    
    '''
    # Extract the csv subset
    csv_subset = df.iloc[iloc[0]:iloc[1], iloc[2]:iloc[3]]
    csv_subset.columns = [i.replace('\n','') for i in csv_subset.iloc[0].astype(str).tolist()]
    
    # Drop the first row
    csv_subset = csv_subset.drop(csv_subset.index[0])
    if 'ct' in csv_subset.columns.tolist():
        csv_subset['ct'] = csv_subset['ct'].replace("Undetermined", np.nan)
    
    # ADDED 23/4/24 List of column names to convert to numeric
    columns_to_convert = ['well', 'ct', 'tm', 'threshold', 'usr_raw_ng/L', 'usr_mean_ng/L', 'usr_recovery', 'usr_std', 'usr_ignore']

    # Check if any csv_subset is is from a user qPCR csv template 
    if any(column in csv_subset.columns for column in columns_to_convert):
        # Convert columns to numeric, handling non-convertible values by setting them to NaN
        for column in columns_to_convert:
            if column in csv_subset.columns:
                csv_subset[column] = pd.to_numeric(csv_subset[column], errors='coerce')
    
    # if any value in csv_subset.columns.tolist() is also in columns_to_convert:
    #     # ADDED 23/4/24  Convert columns to numeric, handling non-convertible values by setting them to NaN
    #     for column in columns_to_convert:
    #         csv_subset[column] = pd.to_numeric(csv_subset[column], errors='coerce')
         
    # # Reset the index
    csv_subset = csv_subset.reset_index(drop=True)

    return csv_subset

#### USAGE:
# csv_subset = extract_csv_subset(df)
    

# def create_data_metatable(eds2txt_match_dict, eds2csv_match_dict, path_key):
    
def create_data_metatable(eds2txt_match_dict: dict[str, str], eds2csv_match_dict: dict[str, pd.DataFrame], path_key: str) -> list[Union[pd.DataFrame, pd.Series]]:

    '''
    Checks an experiment for a eds-exported txt file and a developer csv file with a matching name.
    All instrument parameters, user-defined design values, user-defined calculations experimental data are put into a single longform dataframe, meta_list[0]
    Proxipal directory structure is recorded too with respect to the experiment and analysis parent folders
    Row values of this longform dataframe are given by experimental data`
    The meta_list elements are all written to the path_key export directory

    Parameters
    ----------
    eds2txt_match_dict : Dictionary of strings
        The "eds2txt_match_dict" kwarg can be derived using the function "find_matched_filenames()"
        Such a dictionary contains keys = file paths, and values = eds-exported txt files
    eds2csv_match_dict : Dictionary of dataframes
        The "eds2csv_match_dict" kwarg can be derived using the function "find_matched_filenames()"
        Such a dictionary contains keys = file paths, and values = dataframes with eds-matched names
    path_key : String
        The path_key is a path string pointing to a specific experiment starting from the base folder
        (example: '230112_e24/230410/2023_01_12 Expt24 endoNfL_ICPLA_36cap_79SLC171SLC2_LPS_NoLPS_DhP_RMHLot2')
        The suffix for the filename in this string is optional; the function will check for the presence of both 
        a csv and a txt variant to confirm that eds parameters and deverloper csv values are available.
        
        
    Returns
    -------
    meta_list : List, [meta_table, csv_subset, usr_vals, instr_params]
        The "meta_list" provides 4 elements that are also written to the path_key export directory
        meta_table = a dataframe of experimental data with all user-defined design values, user-defined calculations and instrument parameters
        csv_subset = a dataframe of experimental data only
        usr_vals = a series of user-defined design values, user-defined calculations
        instr_params = a series of instrument parameters only
        
        meta_list[1:] are mostly generated for record keeping
        meta_list[0] is used for generating megatables as shown in the function "create_data_megatable()"

    '''
    # Remove path_key suffixes if present
    if '.' in path_key:
        path_key_noext = path_key.split('.', 1)[0]
    else:
        path_key_noext = path_key
        
    # Check path_key present for txt and csv files
    if path_key_noext + '.txt' not in eds2txt_match_dict:
        print (f"{path_key} does not exist, please create it by exporting your eds to txt")
    if path_key_noext + '.csv' not in eds2csv_match_dict:
        print (f"{path_key} does not exist, please create it by creating a csv from your eds analysis file")    
    
    # Create instr_params
    path_key_text = path_key_noext + '.txt'
    instr_params = extract_instr_params(eds2txt_match_dict[path_key_text])
    
    # Add path elements to instr_params
    instr_params_paths_dict = {}
    instr_params_paths_dict['filepath_txt'] = path_key_text
    instr_params_paths_dict['expt_folder_txt'] =  str(path_key_text).split('/')[0]
    instr_params_paths_dict['analysis_folder_txt'] = str(path_key_text).split('/')[1]
    instr_params_paths_dict['filename_txt'] = str(path_key_text).split('/')[2]    
    instr_params_paths_series = pd.Series(instr_params_paths_dict)
    
    # Combine Series
    instr_params = pd.concat([instr_params_paths_series, instr_params], axis=0)
    
    # Create the export path and exports folder
    
    exports_folder = base_path / 'data' / instr_params_paths_dict['expt_folder_txt'] / instr_params_paths_dict['analysis_folder_txt'] / 'exports'
    
    if not exports_folder.exists():
        exports_folder.mkdir(parents=True)
    
    # Write instr_params to the exports folder
    with open(exports_folder / 'instr_params.txt', 'w') as f:
        for key, value in instr_params.items():
            f.write(f"{key}: {value}\n")
            
    # Create usr_vals
    path_key_csv = path_key_noext + '.csv'
    usr_vals = extract_usr_vals(eds2csv_match_dict[path_key_csv])
    
    # Add path elements to usr_vals
    usr_vals_paths_dict = {}
    usr_vals_paths_dict['filepath_csv'] = path_key_csv
    usr_vals_paths_dict['expt_folder_csv'] =  str(path_key_csv).split('/')[0]
    usr_vals_paths_dict['analysis_folder_csv'] = str(path_key_csv).split('/')[1]
    usr_vals_paths_dict['filename_csv'] = str(path_key_csv).split('/')[2]    
    usr_vals_paths_series = pd.Series(usr_vals_paths_dict)
    
    # Combine Series
    usr_vals = pd.concat([usr_vals_paths_series, usr_vals], axis=0)
    
    # Write usr_vals to the exports folder
    with open(exports_folder / 'usr_vals.txt', 'w', encoding='utf-8') as f:
        for key, value in usr_vals.items():
            f.write(f"{key}: {value}\n")
    
    # Write csv subset to the exports folder
    csv_subset = extract_csv_subset(eds2csv_match_dict[path_key_csv])
    csv_subset.to_csv(exports_folder / 'csv_subset.csv', index=False)
    
    # create a concatenated table of all values
    usr_vals_df = pd.concat([pd.DataFrame(usr_vals).T]*len(csv_subset), ignore_index=True)
    instr_params_df = pd.concat([pd.DataFrame(instr_params).T]*len(csv_subset), ignore_index=True)
    meta_table = pd.concat([csv_subset, usr_vals_df, instr_params_df], axis=1)
    
    meta_table = pd.concat([csv_subset, usr_vals_df, instr_params_df], axis=1)
    
    # 
    pattern_std = r"(std\d+)\[(.*?)\]_"
    pattern_prefix = r"(std\d+)"
    
    # Ensure 'sample_id' is string type
    meta_table['sample_id'] = meta_table['sample_id'].astype(str)
    
    # Create 'usr_std_list' without modifying 'usr_std' in metatable
    usr_std_list = meta_table['usr_std'].apply(lambda x: f'std{x}_')
    
    # Initialize 'calibrator' column
    meta_table['calibrator'] = "n/a"
    
    standards = meta_table['sample_id'].str.contains(pattern_std)
    matched = meta_table['sample_id'][standards]
    
    # Replace matched strings with the desired format
    standards_noConc = matched.apply(lambda x: re.sub(pattern_std, r"\1_", x)).unique()
    standards_noConc_list = standards_noConc.tolist()
    
    # Iterate over 'usr_std_list' and 'calibrator' simultaneously
    for idx, usr_std in enumerate(usr_std_list):
        for s in standards_noConc_list:
            if usr_std in s:
                # Replace 'stdx_' prefix with an empty string
                meta_table.at[idx, 'calibrator'] = re.sub(pattern_prefix + "_", "", s)
    
    # Assign a random data_id to each row; this id can be used to check that data is not being duplicated 
    meta_table['analysis_uuid'] = [str(uuid.uuid4()) for _ in range(len(meta_table))]

    # Write meta_table to the exports folder
    meta_table.to_csv(exports_folder / 'metatable.csv', index=False)
    
    meta_list = [meta_table, csv_subset, usr_vals, instr_params]

    return meta_list

# USAGE
# path_key = '230112_e24_kruti/230410/2023_01_12 Expt24 endoNfL_ICPLA_36cap_79SLC171SLC2_LPS_NoLPS_DhP_RMHLot2'
# meta_list = create_data_metatable(eds2txt_match_dict, eds2csv_match_dict, path_key)
    

def create_py_metatable(metatable: pd.DataFrame, rdml_check: bool = True, threshold_type = 'ct', export = True):

    """
    Creates a processed metatable (py_metatable) from input data, handling both standard text files and RDML files.
    This function processes raw data tables by cleaning Excel-generated error strings, setting up export directories,
    and calculating Simple Linear Regression (SLR) metrics for PCR analysis.

    The function can process two types of input data:
    1. Standard text files (when rdml_check=False): Processes only the ct threshold type
    2. RDML files (when rdml_check=True): Processes both ct and rdml_Cq thresholds with statistical efficiency

    Parameters
    ----------
    metatable : pd.DataFrame
        Input DataFrame containing PCR data with required columns:
        - filepath_txt: Path to source text file
        - Other columns needed for SLR calculations
    rdml_check : bool, optional (default=True)
        If True, checks for and processes accompanying .rdml file
        If False, processes only the text-based metatable
    threshold_type : str, optional (default='ct')
        Type of threshold to use for calculations
        Common values: 'ct', 'rdml_log2N0 (mean eff) - no plateau - stat efficiency'
    export : bool, optional (default=True)
        If True, exports the processed metatable to a CSV file in the exports folder

    Returns
    -------
    pd.DataFrame
        Processed metatable (py_metatable) with additional calculated columns including:
        - SLR metrics (gradient, y-intercept, R-squared)
        - PCR efficiency calculations
        - Concentration calculations
        - Recovery metrics

    Notes
    -----
    - The function creates an 'exports' folder in the same directory as the input file
    - For RDML processing, expects .rdml file with same base name as .txt file
    - Excel error strings (#DIV/0!, #N/A) are converted to NaN
    """

    # Error strings created by excel should be removed
    metatable = metatable.replace('#DIV/0!', np.nan)
    metatable = metatable.replace('#N/A', np.nan)
    
    # Create exports folder
    if len(metatable['filepath_txt'].unique().tolist()) == 1:
        proxipal_path = Path(metatable['filepath_txt'].unique().tolist()[0])
        exports_folder = data_folder / proxipal_path.parent / 'exports'
    
        if not exports_folder.exists():
            exports_folder.mkdir(parents=True)
        
    elif len(metatable['filepath_txt'].unique().tolist()) != 1:
        print ('''Unable to write py_metatable to a single /export folder because either 
                \n1) input metatable has no filepath_txt value or 
                \n2) multiple filepath_txt values were found, i.e. using a megatable or mastertable''')

    # Added function to accommodate .rdml analysis
    if rdml_check == True:
        rdml_path = Path(data_folder / str(proxipal_path).replace('.txt','.rdml'))
        
        if rdml_path.exists():
            rdml_metatable = rdml_linreg_analyse(rdml_path)
            py_metatable = calc_metatable_SLR(rdml_metatable, threshold_type = 'ct', export = export)
            # py_metatable = calc_metatable_SLR(py_metatable.copy(), threshold_type = 'rdml_Cq (mean eff) - no plateau - stat efficiency', export = export)
            py_metatable = calc_metatable_SLR(py_metatable.copy(), threshold_type = 'rdml_log2N0 (mean eff) - no plateau - stat efficiency', export = export)
        
        if not rdml_path.exists():
            raise FileNotFoundError(
                f"RDML file not found at: {rdml_path}\n"
                "Either:\n"
                "1. Supply the corresponding .rdml file, or\n"
                "2. Set rdml_check=False to process without RDML data"
            )
    
    elif rdml_check == False:
        py_metatable = calc_metatable_SLR(metatable, threshold_type = 'ct', export = export)

    return py_metatable

## USAGE
# py_metatable = create_py_metatable(metatable, threshold_type = 'ct', rdml_check = True, export = True)


def filter_metatable_paths(base_path: Union[str, Path]) -> pd.DataFrame:
    """
    Traverses the directory structure starting from the `base_path` and returns a DataFrame containing 
    metatable file information (values; parent, filename, and path). If a single folder contains both 
    py_metatable and metatable files, only the py_metatable file is kept.

    Parameters
    ----------
    base_path : Union[str, Path]
        The base directory path to start the search, either as a string or pathlib object.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing metatable file information, including parent folder, filename, and file path.
    """
    # Convert base_path to a pathlib object if necessary
    if isinstance(base_path, str):
        base_path = Path(base_path)

    # Get list of all file paths that meet the filter criteria
    meta_paths = [p for p in base_path.rglob('*/*metatable*.csv') if 'exports' in str(p.parent)]

    # Create list of dictionaries with "parent" and "filename" keys
    path_list = []
    for p in meta_paths:
        path_dict = {"parent": p.parent, "filename": p.name, "path": p}
        path_list.append(path_dict)

    # Create DataFrame from list of dictionaries
    path_df = pd.DataFrame(path_list)

    # df rows with py_metatable
    path_df_pymetatable = path_df[path_df['filename'].str.contains('py_metatable')]
    # remove path_df rows in py_metable parent list (also removes metatable rows)
    path_df = path_df[~path_df['parent'].isin(path_df_pymetatable['parent'].tolist())]
    # add the py_metatable rows back in
    concatenated_df = pd.concat([path_df, path_df_pymetatable], axis=0)

    return concatenated_df.sort_index()


def create_data_megatable(data_folder: Path, export: bool = True) -> pd.DataFrame:
    '''
    Synthesizes all metatables into a single table and exports it to the exports folder within the input path.
    If a folder contains both metatable.csv and py_metatable.csv files, only the latter will be used.

    Parameters
    ----------
    data_folder : str or Path
        The path to the base folder that contains the metatables to be synthesized.
    export : bool, optional
        Specifies whether to export the synthesized table to the exports folder within the input path, by default True.

    Returns
    -------
    pd.DataFrame
        The synthesized megatable that contains all the data from the metatables in the base folder.
    '''
    # Find all the metatable paths in the base folder
    df_metapaths = filter_metatable_paths(data_folder)
    
    # Read each metatable into a DataFrame and store in a list
    df_meta_list = []
    for p in df_metapaths['path']:
        pathlib = Path(p)
        df_meta_list.append(pd.read_csv(pathlib))
    
    # Concatenate the list of DataFrames into a single DataFrame
    mega_table = pd.concat(df_meta_list, ignore_index=True)
    
    # Drop rows that have no values in columns 2 or greater
    mega_table = mega_table.dropna(subset=mega_table.columns[2:], how='all')
    
    # Reset the index of the DataFrame after dropping rows
    mega_table = mega_table.reset_index(drop=True)
    
    # Export the megatable to the exports folder if export=True
    if export:
        exports_folder = data_folder / 'exports'
        if not exports_folder.exists():
            exports_folder.mkdir(parents=True)   

        curr_time = datetime.now().strftime("%Y%m%d %H-%M")[2:].replace(' ' ,'_T')
        
        filename = curr_time + ' data_megatable.csv'
        mega_table.to_csv(exports_folder / filename, index=False)

    # Return the synthesized megatable
    return mega_table


def extract_submission_vals(df: pd.DataFrame) -> pd.Series:
    '''
    Processes a samples table file and extracts global submission form values to be used in downstream analysis.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame that contains the sample submission form data.

    Returns
    -------
    pd.Series
        A pandas series that contains the extracted user-defined values from the sample submission form.

    '''
    # Check if the form type is supported and extract the user-defined values
    if df.loc[0, 0] == 'Sample Submission Form B_v1':
        contact_headers = df.loc[1:8, 0].tolist()
        header_vals = df.loc[1:7, 1].tolist() + [' '.join([str(i) for i in df.loc[9:14, 0].tolist() if str(i) != 'nan'])]
        submission_headers = df.loc[1:6, 7].tolist()
        submission_vals = df.loc[1:5, 9].tolist() + df.loc[7:7, 7].tolist()

        # Build a pandas series from the extracted values
        sample_series = pd.Series(header_vals + submission_vals, index=contact_headers + submission_headers)
        return sample_series
    
    # If the form type is not supported, print a warning message and return None
    else:
        print('Sample submission forms currently supported: Sample Submission Form B_v1')
        print('Please check you have submitted the the right form or that it has not been altered')
        return None

# # USAGE
# # Read in sumple submission form
# samples_table = pd.read_csv(samples_folder / '230422_Joe Bloggs_1.csv', header = None)
# # Extract the submission information as a series
# submission_series = extract_submission_vals(samples_table)

def process_samples_table(path: Path) -> pd.DataFrame:
    '''
    Processes a samples table file and extracts a subset of the data to be used in downstream analysis.

    Parameters
    ----------
    path : Path
        The file path to the samples table file.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame that contains the subset of data extracted from the samples table.

    '''
    try:
        # Attempt to read the CSV file with the default UTF-8 encoding
        samples_table = pd.read_csv(path, header=None)
    except UnicodeDecodeError as e:
        # If a UnicodeDecodeError occurs, print the error and file path, and re-raise the exception
        print(f"UnicodeDecodeError while reading {path}: {e}")
        raise
    except Exception as e:
        # Handle other exceptions if necessary
        print(f"An error occurred while processing {path}: {e}")
        raise

    # Extract the submission values from the samples table
    submission_series = extract_submission_vals(samples_table)
    
    # Extract a subset of the samples table using the extract_csv_subset function
    samples_subset = extract_csv_subset(samples_table, iloc=[21, len(samples_table), 0, 14])
    
    # Replace missing values in the samples subset with the submission values
    samples_subset.loc[(samples_subset['sample_id'].notnull()) | (samples_subset['tube_id'].notnull()), submission_series.index] = submission_series.values

    # Return the extracted samples subset as a DataFrame
    return samples_subset

# USAGE 
# samples_table = process_samples_table(samples_folder / '230422_Joe Bloggs_1.csv')


def process_samples_folder(samples_folder: Union[str, Path], export: bool = True) -> dict[str, pd.DataFrame]:
    """
    Processes all CSV files in the specified samples folder, applying the 'process_samples_table' function
    to each file. Optionally exports the processed data to a subfolder named 'exports' within the samples folder.

    Parameters
    ----------
    samples_folder : Union[str, Path]
        The path to the folder containing the sample CSV files to be processed, either as a string or pathlib object.
    export : bool, optional
        If True (default), processed data is saved as CSV files in the 'exports' subfolder within the samples folder.

    Returns
    -------
    samples_table_dict : Dict[str, pd.DataFrame]
        A dictionary containing the processed data for each sample CSV file, with the keys being the file path
        and the values being the corresponding processed DataFrames.
    """

    # Convert samples_folder to a pathlib object if necessary
    if isinstance(samples_folder, str):
        samples_folder = Path(samples_folder)

    # Initialize the samples_table_dict
    samples_table_dict = {}

    # Process each CSV file in the samples folder
    for f in samples_folder.glob('*'):
        if f.suffix == '.csv':
            samples_subset = process_samples_table(f)
            samples_table_dict[get_file_and_parents(f, num_parents=2)] = samples_subset

    # Export the processed data to CSV files in the 'exports' subfolder if export is True
    if export:
        exports_folder = samples_folder / 'exports'
        if not exports_folder.exists():
            exports_folder.mkdir(parents=True)

        for key, value in samples_table_dict.items():
            filename = 'processed_' + key.split('/')[-1]
            value.to_csv(exports_folder / filename, index=False)

    return samples_table_dict

# Usage
# samples_table_dict = process_samples_folder(samples_folder)


def create_samples_megatable(samples_folder: Union[str, Path], export: bool = True) -> pd.DataFrame:
    """
    Processes all sample CSV files in the specified folder and concatenates them into a single "megatable".
    Optionally exports the megatable as a CSV file to a subfolder named 'exports' within the samples folder.

    Parameters
    ----------
    samples_folder : Union[str, Path]
        The path to the folder containing the sample CSV files to be processed, either as a string or pathlib object.
    export : bool, optional
        If True (default), the megatable is saved as a CSV file in the 'exports' subfolder within the samples folder.

    Returns
    -------
    mega_table : pd.DataFrame
        A DataFrame containing the combined data from all processed sample CSV files.
    """

    # Convert samples_folder to a pathlib object if necessary
    if isinstance(samples_folder, str):
        samples_folder = Path(samples_folder)

    # Process each sample CSV file and store the resulting DataFrames in a list
    df_meta_list = []
    samples_table_dict = process_samples_folder(samples_folder)
    for key, value in samples_table_dict.items():
        df_meta_list.append(value)

    # Concatenate the DataFrames in the list to create the megatable
    mega_table = pd.concat(df_meta_list, ignore_index=True)
    mega_table = mega_table.dropna(subset=mega_table.columns[1:], how='all')
    mega_table = mega_table.reset_index(drop=True)

    # Export the megatable to a CSV file in the 'exports' subfolder if export is True
    if export:
        exports_folder = samples_folder / 'exports'

        if not exports_folder.exists():
            exports_folder.mkdir(parents=True)

        curr_time = datetime.now().strftime("%Y%m%d %H-%M")[2:].replace(' ', '_T')

        filename = curr_time + ' samples_megatable.csv'
        mega_table.to_csv(exports_folder / filename, index=False)

    return mega_table

# USAGE
# samples_megatable = create_samples_megatable(samples_folder, export = True)


def calc_conc_from_linear_regression(ct: float, dilution: float, y_intercept: float, gradient: float) -> float:
    """
    Calculates the concentration of a sample from its Ct value, dilution factor, and the parameters of a linear
    regression model (y-intercept and gradient) representing the relationship between Ct values and the log of
    the sample concentration.

    Parameters
    ----------
    ct : float
        The Ct value of the sample.
    dilution : float
        The dilution factor applied to the sample.
    y_intercept : float
        The y-intercept of the linear regression model.
    gradient : float
        The gradient (slope) of the linear regression model.

    Returns
    -------
    float
        The calculated concentration of the sample, rounded to 4 decimal places.
    """

    # Calculate the log concentration using the linear regression model
    log_concentration = (ct - y_intercept) / gradient

    # Calculate the actual concentration using the dilution factor
    concentration = 10 ** log_concentration * dilution

    # Round the calculated concentration to 4 decimal places
    return round(concentration, 4)


# Metrics suffix dictionary for calc_metatable_SLR(); simple linear regression (SLR)
SLR_dict = {'gradient': 'grad', 
            'y intercept': 'y_int',
            'equation': 'equation',
            'R-squared': 'R2',
            'PCR efficiency': 'PCReff'}


def calc_metatable_SLR(py_metatable, threshold_type = 'rdml_Cq (mean eff)', std0_status = 'exc_std0',  export = True):
    """
    [Previous docstring remains the same]
    """    
    # Fixed parameters as prerequisites
    transform_x = 'linear(x)' if '_N0' in threshold_type else 'log10(x)'

    # Avoid copy warning
    SLR_metatable = py_metatable.copy()

    # Error strings created by excel should be removed
    SLR_metatable = SLR_metatable.replace('#DIV/0!', np.nan)
    SLR_metatable = SLR_metatable.replace('#N/A', np.nan)

    # Check for column py_known_conc. if not present, create it.
    SLR_metatable = add_py_known_conc(SLR_metatable.copy())
        
    # Assign column name for threshold mean   
    threshold_mean = threshold_type + '; mean'

    # Where "usr_ignore" = 1 the threshold_mean is not calculated, it is assigned the raw threshold_type.
    SLR_metatable.loc[SLR_metatable["usr_ignore"] == 1, threshold_mean] = SLR_metatable.loc[SLR_metatable["usr_ignore"] == 1, threshold_type]

    # Filter the dataframe for "usr_ignore" /= 1 and calculate the mean for the threshold_type, based on rep_id grouping.
    filtered_df = SLR_metatable[SLR_metatable["usr_ignore"] != 1].copy()
    filtered_df.loc[:, threshold_mean] = filtered_df.groupby("rep_id")[threshold_type].transform("mean")

    # Identify how many standards are in use via the usr_std column
    stds_used_list = filtered_df['usr_std'].unique().tolist()

    # For each standard present create a subset dataframe, std_df
    for i in stds_used_list:
        std = 'std' + str(i)
        pattern = r"{}\[.*?\]_".format(std)

        # create metatable subset on pattern
        std_df = filtered_df[filtered_df['sample_id'].str.contains(pattern, regex=True)]
        
        # Choose appropriate x values based on threshold_type
        x_column = 'py_known_conc' if '_N0' in threshold_type else 'py_known_conc_log10'
        
        # Filter std0 based on both threshold_type and std0_status
        if '_N0' in threshold_type and std0_status == 'inc_std0':
            # Keep all rows including std0 for N0 with inc_std0
            valid_data = std_df.dropna(subset=[x_column, threshold_mean])
        else:
            # For all other cases (non-N0 or exc_std0), exclude rows where py_known_conc = 0
            valid_data = std_df[std_df['py_known_conc'] != 0].dropna(subset=[x_column, threshold_mean])

        # means should be calculated for replicates prior to fitting
        valid_data_means = valid_data.groupby('rep_id')[[x_column, threshold_mean]].mean()

        if len(valid_data_means) < 2:
            print(f"Warning: Not enough data points for {std} to perform regression")
            continue
            
        y = valid_data_means[threshold_mean].values.reshape(-1, 1)
        x = valid_data_means[x_column].values.reshape(-1, 1)

        model = LinearRegression()
        model.fit(x, y)
        gradient = model.coef_[0][0]
        y_intercept = model.intercept_[0]
        line_eq = f"y = {gradient:.4e}x + {y_intercept:.4e}"  # Using scientific notation

        y_pred = model.predict(x)
        r_square = r2_score(y, y_pred)
        
        # Store each metric in its appropriate column
        metrics = {
            'gradient': gradient,
            'y intercept': y_intercept,
            'equation': line_eq,
            'R-squared': r_square,
            'PCR efficiency': None
        }
        
        for key, value in metrics.items():
            column_name = f"SLR; {transform_x}; {std0_status}; {threshold_type}; {SLR_dict[key]}"
            if value is not None:
                if key in ['equation']:
                    SLR_metatable.loc[SLR_metatable['usr_std'] == i, column_name] = value
                else:
                    # Store full precision for numerical values
                    SLR_metatable.loc[SLR_metatable['usr_std'] == i, column_name] = value

    # Update metatable with calculations
    mask = SLR_metatable["usr_ignore"] != 1
    
    # Mean threshold calculation remains the same
    SLR_metatable.loc[mask, threshold_mean] = SLR_metatable.loc[mask].groupby("rep_id")[threshold_type].transform("mean")
    
    # Update PCR efficiency calculation using new column name format
    grad_col = f"SLR; {transform_x}; {std0_status}; {threshold_type}; {SLR_dict['gradient']}"
    pcr_eff_col = f"SLR; {transform_x}; {std0_status}; {threshold_type}; {SLR_dict['PCR efficiency']}"

    if '_N0' not in threshold_type:
        SLR_metatable.loc[mask, pcr_eff_col] = SLR_metatable.loc[mask].apply(
            lambda row: round((10**(-1/row[grad_col]) - 1) * 100, 2), axis=1)
    else:
        SLR_metatable.loc[mask, pcr_eff_col] = np.nan
    
    # Update concentration calculations using new column names and handle _N0 case
    y_int_col = f"SLR; {transform_x}; {std0_status}; {threshold_type}; {SLR_dict['y intercept']}"
    
    def calc_conc(row):
        if '_N0' in threshold_type:
            return (row[threshold_type] - row[y_int_col]) / row[grad_col] * row['dilution']
        else:
            return calc_conc_from_linear_regression(row[threshold_type], row['dilution'], row[y_int_col], row[grad_col])
            
    def calc_conc_mean(row):
        if '_N0' in threshold_type:
            return (row[threshold_mean] - row[y_int_col]) / row[grad_col] * row['dilution']
        else:
            return calc_conc_from_linear_regression(row[threshold_mean], row['dilution'], row[y_int_col], row[grad_col])
    
    SLR_metatable.loc[mask, f"SLR; {transform_x}; {std0_status}; {threshold_type}; raw_ng/L"] = \
        SLR_metatable.loc[mask].apply(calc_conc, axis=1)
        
    SLR_metatable.loc[mask, f"SLR; {transform_x}; {std0_status}; {threshold_type}; mean_ng/L"] = \
        SLR_metatable.loc[mask].apply(calc_conc_mean, axis=1)
    
    SLR_metatable.loc[mask, f"SLR; {transform_x}; {std0_status}; {threshold_type}; raw_recovery"] = SLR_metatable.loc[mask].apply(
        lambda row: round(row[f"SLR; {transform_x}; {std0_status}; {threshold_type}; raw_ng/L"] / row['py_known_conc'], 2) if row['py_known_conc'] != 0 else np.nan, axis=1)
    
    SLR_metatable.loc[mask, f"SLR; {transform_x}; {std0_status}; {threshold_type}; mean_recovery"] = SLR_metatable.loc[mask].apply(
        lambda row: round(row[f"SLR; {transform_x}; {std0_status}; {threshold_type}; mean_ng/L"] / row['py_known_conc'], 2) if row['py_known_conc'] != 0 else np.nan, axis=1)

    if export == True: 
        # Create exports folder
        if len(py_metatable['filepath_txt'].unique().tolist()) == 1:
            proxipal_path = Path(py_metatable['filepath_txt'].unique().tolist()[0])
            exports_folder = data_folder / proxipal_path.parent / 'exports'
        
            if not exports_folder.exists():
                print ('No local /exports folder found. Creating one.')
                exports_folder.mkdir(parents=True)
                
        SLR_metatable.to_csv(exports_folder / 'py_metatable.csv', index=False)
    
    return SLR_metatable

### USAGE
# SLR_metatable = calc_metatable_SLR(py_metatable, threshold_type = 'rdml_Cq (mean eff)', export = False)


def plot_slr_standards(metatable, threshold_type='rdml_Cq (mean eff)', std0_status = 'exc_std0', figsize=(10, 6), separate_plots=False):
    """
    [Previous docstring remains the same]
    """
    # [Previous code remains the same up to the plotting functions]
    # Fixed parameters for SLR
    transform_x = 'linear(x)' if '_N0' in threshold_type else 'log10(x)'
    
    # Get unique standards with non-zero usr_std
    standards = metatable[(metatable['usr_std'].notna()) & (metatable['usr_std'] > 0)]['usr_std'].unique()
    
    # Color map for different standards
    colors = plt.cm.tab10(np.linspace(0, 1, len(standards)))
    
    threshold_mean = f"{threshold_type}; mean"
    
    # Determine boxout position based on threshold_type
    is_n0 = '_N0' in threshold_type
    x_position = 0.01 if is_n0 else 0.78  # Move closer to left/right edges
    
    if separate_plots:
        plots_dict = {}
        
        for idx, std in enumerate(standards):
            fig, ax = plt.subplots(figsize=figsize)
            color = colors[idx]
            std_name = f'Standard {std}'
            
            plot_single_slr_standard(ax, metatable, std, color, threshold_type, threshold_mean, 
                                   transform_x, std0_status, std_name, single_plot=True,
                                   x_position=x_position, is_n0=is_n0)
            
            format_slr_axes(ax, metatable, threshold_type, is_n0=is_n0)
            plt.tight_layout()
            
            plots_dict[f'std{std}'] = {'fig': fig, 'ax': ax}
            
        return plots_dict
    
    else:
        fig, ax = plt.subplots(figsize=figsize)
        
        equations_text = []
        for idx, std in enumerate(standards):
            color = colors[idx]
            std_name = f'Standard {std}'
            eq_text = plot_single_slr_standard(ax, metatable, std, color, threshold_type, 
                                             threshold_mean, transform_x, std0_status, std_name,
                                             is_n0=is_n0)
            equations_text.append((eq_text, color))
        
        # Add equations to plot with colored boxes
        box_height = 0.12  # Approximate height of each box
        start_y = 0.97  # Move closer to top edge
        
        for idx, (text, color) in enumerate(equations_text):
            vertical_position = start_y - (box_height * idx)
            ax.text(x_position, vertical_position, text,
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, boxstyle='round,pad=0.5'),
                    verticalalignment='top')
        
        format_slr_axes(ax, metatable, threshold_type, is_n0=is_n0)
        plt.tight_layout()
        
        return {'fig': fig, 'ax': ax}
    
def plot_single_slr_standard(ax, metatable, std, color, threshold_type, threshold_mean, 
                           transform_x, std0_status, std_name, single_plot=False, 
                           x_position=0.01, is_n0=False):
    """Helper function to plot a single standard curve with linear regression"""
    # Filter data for this standard and valid values
    mask = (metatable['usr_std'] == std) & (metatable['usr_ignore'] != 1)
    std_data = metatable[mask].copy()
    
    if len(std_data) == 0:
        print(f"No data found for standard {std}")
        return None
    
    # Get mean values for each replicate group
    grouped = std_data.groupby('rep_id').agg({
        'py_known_conc': 'mean',
        threshold_mean: 'mean' if threshold_mean in std_data.columns else threshold_type
    }).sort_values('py_known_conc')
    
    # Plot scatter points for mean values
    ax.scatter(grouped['py_known_conc'], 
              grouped[threshold_mean if threshold_mean in grouped.columns else threshold_type], 
              color=color, alpha=0.7, s=50)
    
    # Get SLR parameters using updated column naming
    param_prefix = f"SLR; {transform_x}; {std0_status}; {threshold_type}"
    gradient = std_data[f"{param_prefix}; {SLR_dict['gradient']}"].iloc[0]
    y_int = std_data[f"{param_prefix}; {SLR_dict['y intercept']}"].iloc[0]
    R2 = std_data[f"{param_prefix}; {SLR_dict['R-squared']}"].iloc[0]
    PCReff = std_data[f"{param_prefix}; {SLR_dict['PCR efficiency']}"].iloc[0]
    
    # Create smooth line for the fit
    if is_n0:
        x_min = max(grouped['py_known_conc'].min() * 0.8, 0)
        x_max = grouped['py_known_conc'].max() * 1.2
        x_smooth = np.linspace(x_min, x_max, 100)
        y_smooth = gradient * x_smooth + y_int
    else:
        x_min = max(grouped['py_known_conc'].min() * 0.8, 0.005)
        x_max = grouped['py_known_conc'].max() * 1.2
        x_smooth = np.logspace(np.log10(x_min), np.log10(x_max), 100)
        y_smooth = gradient * np.log10(x_smooth) + y_int
    
    # Plot the regression line
    ax.plot(x_smooth, y_smooth, '-', color=color)
    
    # Create equation text
    text = f"{std_name}:"
    if is_n0:
        # Use scientific notation for _N0 cases
        text += f"\ny = {gradient:.2e}x + {y_int:.2e}"
    else:
        text += f"\ny = {gradient:.4f}x + {y_int:.4f}"
    text += f"\nR² = {R2:.4f}"
    if not is_n0:
        text += f"\nPCR eff = {PCReff:.1f}%"
    
    # If single plot, add equation directly
    if single_plot:
        vertical_position = 0.97
        ax.text(x_position, vertical_position, text,
                transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, boxstyle='round,pad=0.5'),
                verticalalignment='top')
    
    return text

def format_slr_axes(ax, metatable, threshold_type, is_n0=False):
    """Helper function to format plot axes for SLR"""
    ax.set_xlabel('Concentration (ng/L)')
    ax.set_ylabel(threshold_type)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    
    # Set reasonable y-axis limits
    ydata = metatable[metatable['usr_std'].notna()][threshold_type]
    ymin, ymax = ydata.min(), ydata.max()
    margin = (ymax - ymin) * 0.1
    ax.set_ylim(ymin - margin, ymax + margin)
    
    if is_n0:
        # Linear scale for N0 threshold types
        ax.grid(True, which='major', linestyle='--', alpha=0.3)
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        
        # Set reasonable x-axis limits
        xdata = metatable[metatable['usr_std'].notna()]['py_known_conc']
        xmin, xmax = xdata.min(), xdata.max()
        margin = (xmax - xmin) * 0.1
        ax.set_xlim(max(0, xmin - margin), xmax + margin)
    else:
        # Log scale for non-N0 threshold types
        ax.set_xscale('log')
        ax.grid(True, which='minor', linestyle=':', alpha=0.2)
        
        # Format tick labels for log scale
        def format_func(x, p):
            if x < 1:
                return f"{x:.3f}"
            elif x < 10:
                return f"{x:.1f}"
            return f"{int(x)}"
        
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_func))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0))
        ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1)))
    
    # Set tick parameters
    ax.tick_params(axis='x', which='major', labelsize=8, rotation=0)
    ax.tick_params(axis='y', which='major', labelsize=8)

## USAGE: For all standards on one plot
# result = plot_slr_standards(metatable=SLR_metatable, threshold_type='rdml_Cq (mean eff)', std0_status = 'exc_std0', figsize=(8, 6), separate_plots=False)


def load_most_recent_megatable(folder: Path):
    '''
    Load the most recent CSV file with a "_megatable.csv" pattern from the given folder.
    
    Parameters
    ----------
    folder : Path
        The folder containing the CSV files.
    
    Raises
    ------
    ValueError
        If no "_megatable.csv" files are found in the folder.
    
    Returns
    -------
    most_recent_df : DataFrame
        A pandas DataFrame containing the data from the most recent CSV file.
    '''      
    
    # List all files in the folder and filter only the files with the correct pattern
    files = [f for f in folder.iterdir() if "_megatable.csv" in f.name]
    
    if not files:
        raise ValueError("No megatable.csv files found in " + str(folder))
    
    # Parse timestamps from the filenames and find the most recent file
    most_recent_file = None
    most_recent_time = None
    
    for file in files:
        timestamp = file.name.split(" ")[0]
        date_str, time_str = timestamp.split("_")
        date_obj = datetime.strptime(date_str + time_str, "%y%m%dT%H-%M")
    
        if most_recent_time is None or date_obj > most_recent_time:
            most_recent_time = date_obj
            most_recent_file = file
    
    # Read the most recent file using pandas
    most_recent_df = pd.read_csv(most_recent_file)

    return most_recent_df


def load_most_recent_mastertable(folder: Path, instrument_data: bool = False):
    '''
    Load the most recent CSV file with a "mastertable.csv" or "mastertable_wInstrumentData.csv" pattern from the given folder.

    Parameters
    ----------
    folder : Path
        The folder containing the CSV files.
    instrument_data : bool, optional
        Whether to load files with "mastertable_wInstrumentData.csv" pattern. Default is False.
    
    Raises
    ------
    ValueError
        If no matching CSV files are found in the folder.
    
    Returns
    -------
    most_recent_df : DataFrame
        A pandas DataFrame containing the data from the most recent CSV file.
    most_recent_file : Path
        The path of the most recent CSV file.
    '''      

    # Determine the pattern based on the instrument_data argument
    pattern = "mastertable_wInstrumentData.csv" if instrument_data else "mastertable.csv"

    # List all files in the folder and filter only the files with the correct pattern
    files = [f for f in folder.iterdir() if pattern in f.name]
    
    if not files:
        raise ValueError(f"No {pattern} files found in {folder}")

    # Parse timestamps from the filenames and find the most recent file
    most_recent_file = None
    most_recent_time = None
    
    for file in files:
        timestamp = file.name.split(" ")[0]
        date_str, time_str = timestamp.split("_")
        date_obj = datetime.strptime(date_str + time_str, "%y%m%dT%H-%M")
    
        if most_recent_time is None or date_obj > most_recent_time:
            most_recent_time = date_obj
            most_recent_file = file
    
    # Read the most recent file using pandas
    most_recent_df = pd.read_csv(most_recent_file, low_memory=False)

    return most_recent_df, most_recent_file

# Example usage:
# folder_path = Path('path_to_your_folder')
# df, file = load_most_recent_mastertable(folder_path, instrument_data=True)


def create_master_table(match_type: str = 'TS') -> dict:
    """
    Creates a master table by merging two data tables: data_df and samples_df. The merge is performed based on 
    the provided match_type parameter. The master table contains all data with matched sample and quality 
    information, where possible. The function also returns orphan records. Orphan records are records that 
    have experimental data but no submission details, or records that are available but have not been tested.
    
    Parameters
    ----------
    match_type : str, optional
        Determines the merge strategy for the two tables. If 'TS', the function will merge the tables based on 
        both "tube_id" and "sample_id" columns. If 'S', the function will merge the tables based on the 
        "sample_id" column only. The default is 'TS'.
        
    Returns
    -------
    dict
        Dictionary containing the master DataFrame after merge operation ('master_df_merge') and a DataFrame 
        with orphan records ('master_df_orphan').
        
    Side Effects
    ------------
    - Prints out lists of sample_ids for experimental data with no submission record and sample submissions that 
      have no data (or have not been matched to any data).
    - Exports the final merged DataFrame as a .csv file if match_type is 'TS'.
    """

    # To load from previous master tables; if wanting to use new data_df and samples_df tables, then precede with
    # create_data_megatable() and/or create_samples_megatable()
    data_df = load_most_recent_megatable(data_folder / 'exports')
    samples_df = load_most_recent_megatable(samples_folder / 'exports')
    # quality_megatable = load_most_recent_megatable(quality_folder / 'exports')
    
    # Ensure sample_id and tube_id values are string
    data_df['sample_id'] = data_df['sample_id'].astype(str)
    data_df['tube_id'] = data_df['tube_id'].astype(str)
    samples_df['sample_id'] = samples_df['sample_id'].astype(str)
    samples_df['tube_id'] = samples_df['tube_id'].astype(str)
    
    # Create index column in each df
    data_df['data_df_index'] = data_df.index
    samples_df['samples_df_index'] = samples_df.index 
    
    if match_type == 'TS':
        # Merge both tube id and sample id
        TS_merge = data_df.merge(samples_df, on=["tube_id", "sample_id"], how="left")
    
    elif match_type == 'S':
        TS_merge = data_df.merge(samples_df, on=["sample_id"], how="left")

    # New master_match column indicates the Data and Sample information was matched by both tube id and sample id
    TS_merge['match_type'] = match_type
    
    # data_df_orphan represents samples we have experimental data for, but no submission details; should be corrected
    data_df_orphan = TS_merge[pd.isna(TS_merge['Submission ID (Office use only)'])]
    # data_list_orphan = list(set(data_df_orphan['sample_id'].tolist()))
    data_list_orphan = list(set([f"{row['sample_id']} (tube_id: {row['tube_id']})" for _, row in data_df_orphan.iterrows()]))

    # samples_df_orphan represents samples we have available but have not tested
    unique_values = TS_merge['samples_df_index'].dropna().unique().tolist()
    unique_values = [int(x) for x in unique_values if isinstance(x, (int, float)) and x == int(x)]
    samples_df_orphan = samples_df.drop(unique_values, axis = 0)
    # samples_list_orphan = list(set(samples_df_orphan['sample_id'].tolist()))
    samples_list_orphan = list(set([f"{row['sample_id']} (tube_id: {row['tube_id']})" for _, row in samples_df_orphan.iterrows()]))

    # Determine the length of the longest list
    max_length = max(len(data_list_orphan), len(samples_list_orphan))
    
    # Extend both lists to the length of the longest list, filling with None
    data_list_orphan.extend([None]*(max_length - len(data_list_orphan)))
    samples_list_orphan.extend([None]*(max_length - len(samples_list_orphan)))
    
    # Now you can create your DataFrame
    data_dict = {'data(+) submission(-)': data_list_orphan, 'data(-) submission(+)': samples_list_orphan}
    df_orphan = pd.DataFrame(data_dict)

    # Find common entries between both columns
    # Data AND Submission orphans shouldn't occur and would indicate something is wrong with the matching rules.
    common_entries = set(df_orphan['data(+) submission(-)']).intersection(df_orphan['data(-) submission(+)'])
    
    # Separate entries with square brackets and without square brackets
    common_brackets = sorted([entry for entry in common_entries if '[' in str(entry)])
    common_no_brackets = sorted([entry for entry in common_entries if '[' not in str(entry)])
    
    # Combine lists, with entries containing brackets last
    common_sorted = common_no_brackets + common_brackets
    
    # Create DataFrame for common entries
    df_common = pd.DataFrame({'data(+) submission(-)': common_sorted, 'data(-) submission(+)': common_sorted})
    
    # Remove common entries from original DataFrame and drop na
    df_unique = df_orphan[~df_orphan['data(+) submission(-)'].isin(common_entries) & ~df_orphan['data(-) submission(+)'].isin(common_entries)].dropna(how='all')
    
    # Sort unique entries based on 'no data found' column
    df_unique_sorted = df_unique.sort_values(by='data(+) submission(-)')
    
    # Concatenate common and unique entries, reset index
    df_orphan_combined = pd.concat([df_common, df_unique_sorted]).reset_index(drop=True)

    # Drop the index columns
    TS_merge = TS_merge.drop(columns=['data_df_index', 'samples_df_index'])
    
    # Esport only strict matches
    if match_type == 'TS':
        exports_folder = base_path / 'exports'

        if not exports_folder.exists():
            exports_folder.mkdir(parents=True)
        
        curr_time = datetime.now().strftime("%Y%m%d %H-%M")[2:].replace(' ', '_T')

        filename = curr_time + ' mastertable.csv'
        TS_merge.to_csv(base_path / exports_folder / filename, index=False)
    
    master_dict = {}    
    master_dict['master_df_merge'] = TS_merge
    master_dict['master_df_orphan'] = df_orphan_combined
    
    return master_dict

# def batch_py_metatables(eds2txt_match_dict: dict, eds2csv_match_dict: dict) -> None:
#     """
#     Given two dictionaries that match EDS files to txt and csv files respectively, this function
#     processes each matched file set in a batch.

#     :param eds2txt_match_dict: dictionary that maps EDS files to txt files.
#     :param eds2csv_match_dict: dictionary that maps EDS files to csv files.

#     The function iterates over each file set, checks if the txt and csv files exist for the path_key,
#     and if they do, it tries to create a data metatable and calculate standard deviations using linear regression.
#     If the files don't exist or if the processing fails due to FileNotFoundError, an appropriate message is printed.
#     """
    
#     # Generate pivot dataframe from the matched filenames
#     df_pivot = review_matched_filenames(eds2txt_match_dict, eds2csv_match_dict)
    
#     # Iterate over the 'path_key' list
#     for path in df_pivot['path_key'].tolist():
#         # Check if both 'txt' and 'csv' files exist for the given 'path_key'
#         if (df_pivot.loc[df_pivot['path_key'] == path, ['txt', 'csv']].all(axis=1)).any():
#             try:
#                 # If they exist, try to create a data metatable and calculate standard deviations
#                 create_data_metatable(eds2txt_match_dict, eds2csv_match_dict, path)                
#                 path_csv = data_folder / (path + '.csv')
#                 path_metatable = path_csv.parent / 'exports/metatable.csv'
#                 metatable = pd.read_csv(path_metatable, low_memory = False)
#                 calc_py_metatable_all_models(metatable, rdml_check=True, export=True)
#                 print(path, 'processed')
#             except FileNotFoundError:
#                 # If processing fails, print an error message
#                 print(path, 'create_data_metatable() or calc_metatable_std_lin_reg() failed')
#         else:
#             # If 'txt' and 'csv' files do not exist, print an error message
#             print(path, 'is missing, or mislabeled, an eds, txt, or csv file and cannot be processed')
            
#     return None

# def batch_py_metatables(eds2txt_match_dict: dict, eds2csv_match_dict: dict) -> None:
#     """
#     Given two dictionaries that match EDS files to txt and csv files respectively, this function
#     processes each matched file set in a batch.
#     :param eds2txt_match_dict: dictionary that maps EDS files to txt files.
#     :param eds2csv_match_dict: dictionary that maps EDS files to csv files.
#     The function iterates over each file set, checks if the txt and csv files exist for the path_key,
#     and if they do, it tries to create a data metatable and calculate standard deviations using linear regression.
#     If the files don't exist or if the processing fails due to FileNotFoundError, an appropriate message is printed.
#     """
    
#     # Generate pivot dataframe from the matched filenames
#     df_pivot = review_matched_filenames(eds2txt_match_dict, eds2csv_match_dict)
    
#     # Iterate over the 'path_key' list
#     for path in df_pivot['path_key'].tolist():
#         # Check if both 'txt' and 'csv' files exist for the given 'path_key'
#         if (df_pivot.loc[df_pivot['path_key'] == path, ['txt', 'csv']].all(axis=1)).any():
#             try:
#                 # If they exist, try to create a data metatable and calculate standard deviations
#                 create_data_metatable(eds2txt_match_dict, eds2csv_match_dict, path)                
#                 path_csv = data_folder / (path + '.csv')
#                 path_metatable = path_csv.parent / 'exports/metatable.csv'
#                 metatable = pd.read_csv(path_metatable, low_memory = False)
                
#                 # Suppress print statements from calc_py_metatable_all_models
#                 with contextlib.redirect_stdout(io.StringIO()):
#                     calc_py_metatable_all_models(metatable, rdml_check=True, export=True)
#                 print(path, 'processed')
#             except FileNotFoundError:
#                 # If processing fails, print an error message
#                 print(path, 'create_data_metatable() or calc_py_metatable_all_models() failed')
#         else:
#             # If 'txt' and 'csv' files do not exist, print an error message
#             print(path, 'is missing, or mislabeled, an eds, txt, or csv file and cannot be processed')
            
#     return None

def batch_py_metatables(eds2txt_match_dict: dict, eds2csv_match_dict: dict, 
                        df_pivot_slice: str = None, calc_all_models: bool = True) -> None:
    """
    Given two dictionaries that match EDS files to txt and csv files respectively, this function
    processes each matched file set in a batch.
    eds2txt_match_dict: dictionary that maps EDS files to txt files.
    eds2csv_match_dict: dictionary that maps EDS files to csv files.
    df_pivot_slice: optional string to specify which slice of df_pivot to process.
                         Format: 'start:end' or 'start:' (e.g., '15:18' or '15:')
                         If None, processes all rows.
    calc_all_models: Will fit all models from calc_py_metatable_all_models() against many conditions. 
    WARNING. These calculations can lead to vey long processing times ~10min per experiment.
    
    The function iterates over each file set, checks if the txt and csv files exist for the path_key,
    and if they do, it tries to create a data metatable and calculate standard deviations using linear regression.
    If the files don't exist or if the processing fails due to FileNotFoundError, an appropriate message is printed.
    """
    
    # Generate pivot dataframe from the matched filenames
    df_pivot = review_matched_filenames(eds2txt_match_dict, eds2csv_match_dict)
    
    # Get list of paths based on df_pivot_slice parameter
    if df_pivot_slice is not None:
        try:
            # Parse the slice string
            slice_parts = df_pivot_slice.split(':')
            start = int(slice_parts[0]) if slice_parts[0] else None
            end = int(slice_parts[1]) if slice_parts[1] else None
            paths = df_pivot['path_key'].tolist()[start:end]
        except (ValueError, IndexError):
            print(f"Invalid df_pivot_slice format: {df_pivot_slice}. Processing all rows.")
            paths = df_pivot['path_key'].tolist()
    else:
        paths = df_pivot['path_key'].tolist()
    
    # Iterate over the selected paths
    for path in paths:
        # Check if both 'txt' and 'csv' files exist for the given 'path_key'
        if (df_pivot.loc[df_pivot['path_key'] == path, ['txt', 'csv']].all(axis=1)).any():
            try:
                # If they exist, try to create a data metatable and calculate standard deviations
                create_data_metatable(eds2txt_match_dict, eds2csv_match_dict, path)                
                path_csv = data_folder / (path + '.csv')
                path_metatable = path_csv.parent / 'exports/metatable.csv'
                metatable = pd.read_csv(path_metatable, low_memory = False)
                
                if calc_all_models:
                    # Suppress print statements from calc_py_metatable_all_models
                    with contextlib.redirect_stdout(io.StringIO()):
                        calc_py_metatable_all_models(metatable, rdml_check=True, export=True)
                        
                print(path, 'processed')
            except FileNotFoundError:
                # If processing fails, print an error message
                print(path, 'create_data_metatable() or calc_py_metatable_all_models() failed')
        else:
            # If 'txt' and 'csv' files do not exist, print an error message
            print(path, 'is missing, or mislabeled, an eds, txt, or csv file and cannot be processed')
            
    return None

# # Process all rows (original behavior)
# batch_py_metatables(eds2txt_match_dict, eds2csv_match_dict)

# # Process only indexes 15-17
# batch_py_metatables(eds2txt_match_dict, eds2csv_match_dict, df_pivot_slice='15:18')

# # Process from index 15 to the end
# batch_py_metatables(eds2txt_match_dict, eds2csv_match_dict, df_pivot_slice='15:')


def extract_instr_tables(path: Path):
    '''  
    Extracts the following tables from QuantStudio .eds files exported as .txt:
    'Sample Setup', 'Raw Data', 'Amplification Data', 'Multicomponent Data', 
    'Technical Analysis Result', 'BioGroup Analysis Result', 'Results', 'Melt Curve Raw Data'
    
    Returns a dictionary where each key corresponds to one of these tables, and each value is a DataFrame representing the table.
    
    Parameters
    ----------
    path : Path
        The path to the .txt file exported from a QuantStudio .eds file. 

    Returns
    -------
    dict
        A dictionary where keys are the names of the tables in the .eds file, and each value is a DataFrame representing the corresponding table.
    '''
    
    # Open the file and read its lines
    with path.open('r') as file:
        lines = file.readlines()

    # Helper function to check if a string can be interpreted as a decimal number, with or without commas
    def is_numeric(s):
        return bool(re.fullmatch(r"(\d{1,3}(,\d{3})*(\.\d+)?)", s))

    # Helper function to convert a string to a float or int, depending on whether it has a decimal point
    def convert_to_numeric(s):
        s = s.replace(',', '')  # Remove commas
        return int(s) if s.isdigit() else float(s)
        
    # Prepare to collect the tables
    tables = {}
    current_table = []
    current_key = None
    
    # Process each line in the file
    for line in lines:
        if line.startswith('['):  # Beginning of a new table
            # If we've already started a table, save it
            if current_key is not None:
                # Prepare the table for conversion to a DataFrame
                # Ensure all rows have the same length as the header
                max_cols = len(current_table[0])
                for i, row in enumerate(current_table):
                    length = len(row)
                    if length < max_cols:
                        current_table[i] += [None] * (max_cols - length)
                    elif length > max_cols:
                        current_table[i] = row[:max_cols]
                
                # Create DataFrame with lowercase column headers
                df = pd.DataFrame(current_table[1:], columns=[col.lower() for col in current_table[0]])

                # Drop rows with non-numeric 'Well', if 'Well' column exists
                if 'well' in df.columns:
                    df = df[df['well'].apply(is_numeric)]
                
                # Convert strings that are actually numbers
                for col in df.columns:
                    df[col] = df[col].apply(lambda x: convert_to_numeric(x) if is_numeric(str(x)) else x)
                
                # Save the DataFrame to the dictionary of tables
                tables[current_key] = df
            
            # Start a new table
            current_key = line.strip()[1:-1]
            current_table = []
        else:  # Continuation of a table
            current_table.append(line.strip().split('\t'))

    # After going through all lines, save the last table if there is one
    if current_key is not None and current_table:
        # As above, prepare the table for conversion to a DataFrame
        max_cols = len(current_table[0])
        for i, row in enumerate(current_table):
            length = len(row)
            if length < max_cols:
                current_table[i] += [None] * (max_cols - length)
            elif length > max_cols:
                current_table[i] = row[:max_cols]
        
        # Create DataFrame with lowercase column headers
        df = pd.DataFrame(current_table[1:], columns=[col.lower() for col in current_table[0]])

        # Drop rows with non-numeric 'Well', if 'Well' column exists
        if 'well' in df.columns:
            df = df[df['well'].apply(is_numeric)]
        
        # Convert strings that are actually numbers
        for col in df.columns:
            df[col] = df[col].apply(lambda x: convert_to_numeric(x) if is_numeric(str(x)) else x)
        
        # Save the DataFrame to the dictionary of tables
        tables[current_key] = df

    return tables

# Usage
# MRFF2_eds = data_folder / '230324_MRFF_e2_kruti' / '230410' / 'Plate 2_NfL_MRFF_Dhama Plasma 21s to 40s.txt'
# MRFF2_dict = extract_instr_tables(MRFF2_eds)

# def build_instr_df(instr_tables_dict: dict):
#     '''
#     Manipulates a dictionary object created in extract_instr_tables()
#     From the object the dataframes for ['Raw Data', 'Amplification Data', 'Multicomponent Data', 'Melt Curve Raw Data']
#     are manipulated into a single wideform flat dataframe where 1 well = 1 row. Cycling data is collapsed into lists on a per cell basis.
    
#     This function will typically be used for merging with the mastertable so that raw values on a per reaction basis can be compared across experiments.
    
#     Parameters
#     ----------
#     instr_tables_dict : dict
#         An obect created by extract_instr_tables()
        
#     Returns
#     ----------    
#     instr_df : pd.DataFrame
#         A single dataframe contain all values from tables ['Raw Data', 'Amplification Data', 'Multicomponent Data', 'Melt Curve Raw Data']
#     '''
#     # Select the relevant tables
#     relevant_keys = ['Raw Data', 'Amplification Data', 'Multicomponent Data', 'Melt Curve Raw Data']
#     relevant_tables = {key: instr_tables_dict[key] for key in relevant_keys}
    
#     # Prepare a new DataFrame to hold the flattened information
#     instr_df = pd.DataFrame()

#     for key, table in relevant_tables.items():
#         # Flatten each column into a list, or a single value if all values are identical
#         for col in table.columns:
#             instr_df[f'{col} (eds; {key.lower()})'] = table.groupby('well')[col].apply(lambda x: x.iloc[0] if x.nunique() == 1 else x.tolist())

#     # Reset the index to merge on 'well'
#     instr_df.reset_index(inplace=True)

#     return instr_df

def build_instr_df(instr_tables_dict: dict):
    '''
    Manipulates a dictionary object created in extract_instr_tables()
    From the object the dataframes for ['Raw Data', 'Amplification Data', 'Multicomponent Data', 'Melt Curve Raw Data']
    are manipulated into a single wideform flat dataframe where 1 well = 1 row. Cycling data is collapsed into lists on a per cell basis.
    
    This function will typically be used for merging with the mastertable so that raw values on a per reaction basis can be compared across experiments.
    
    Parameters
    ----------
    instr_tables_dict : dict
        An object created by extract_instr_tables()
        
    Returns
    ----------    
    instr_df : pd.DataFrame
        A single dataframe containing all values from tables
        ['Raw Data', 'Amplification Data', 'Multicomponent Data', 'Melt Curve Raw Data'] (where present).
    '''
    # REQUIRED tables
    required_keys = ['Raw Data', 'Amplification Data', 'Multicomponent Data']
    # OPTIONAL tables
    optional_keys = ['Melt Curve Raw Data']

    # Fail only if a required table is missing
    missing_required = [k for k in required_keys if k not in instr_tables_dict]
    if missing_required:
        raise KeyError(f"Missing required keys {', '.join(missing_required)}")

    # Only include optional tables that actually exist
    present_optional = [k for k in optional_keys if k in instr_tables_dict]

    # Select the tables we actually have
    relevant_keys = required_keys + present_optional
    relevant_tables = {key: instr_tables_dict[key] for key in relevant_keys}
    
    # Prepare a new DataFrame to hold the flattened information
    instr_df = pd.DataFrame()

    for key, table in relevant_tables.items():
        # Flatten each column into a list, or a single value if all values are identical
        for col in table.columns:
            instr_df[f'{col} (eds; {key.lower()})'] = (
                table.groupby('well')[col]
                     .apply(lambda x: x.iloc[0] if x.nunique() == 1 else x.tolist())
            )

    # Reset the index to merge on 'well'
    instr_df.reset_index(inplace=True)

    return instr_df

# Usage
# Build the instr_df with selected tables
# MRFF2_df = build_instr_df(MRFF2_dict)
# Add in an identifier column if planning to merge on mastertable
# MRFF2_df['filepath_txt'] = '230324_MRFF_e2_kruti/230410/Plate 2_NfL_MRFF_Dhama Plasma 21s to 40s.txt'


# def build_master_instr_df(df_pivot: pd.DataFrame, data_folder: Path):
#     '''
#     Accepts a pd.DataFrame reporting the file integrity status of experiments in the data_folder.
#     The dataframe is produced by the review_matched_filenames() function.
    
#     When executed, build_master_instr_df() will iterate on every experiment with all requisite files and 
#     extract all Quantstudio .eds exported data from the tables ['Raw Data', 'Amplification Data', 
#     'Multicomponent Data', 'Melt Curve Raw Data'] of each experiment.
    
#     Parameters
#     ----------
#     df_pivot : pd.DataFrame
#         A dataframe produced by review_matched_filenames()
    
#     data_folder : Path
#         A pathlib object pointing to ProxiPal's /data folder
        
#     Returns
#     ----------
#     all_instr_dfs : pd.DataFrame
#         Consolidates all .eds exported raw data from ProxiPal experiments into a single concatenated dataframe.
#         Adds column "filepath_txt" for each experiment so that, with "well" all instrument data can be merged onto the mastertable.
#     '''
    
#     all_instr_dfs = pd.DataFrame()

#     for idx, row in df_pivot.iterrows():
#         if row['txt'] == True and row['csv'] == True:
#             path_key = data_folder / (row['path_key'] + '.txt')
#             instr_dict = extract_instr_tables(path_key)
            
#             if len(instr_dict) != 8:
#                 print('Warning: Not all instrument tables were exported. Open the QuantStudio .eds and re-export the .txt file below. This function will not complete if an eds export is incomplete.')
#                 print(path_key)
                
#             # Select the relevant tables
#             relevant_keys = ['Raw Data', 'Amplification Data', 'Multicomponent Data', 'Melt Curve Raw Data']
            
#             missing_keys = [key for key in relevant_keys if key not in instr_dict]
            
#             if missing_keys:
#                 raise KeyError(f"Missing keys {', '.join(missing_keys)} in file {path_key}")
                
#             instr_df = build_instr_df(instr_dict)
#             instr_df['filepath_txt'] = str(row['path_key'] + '.txt')
#             all_instr_dfs = pd.concat([all_instr_dfs, instr_df], axis=0)

#     return all_instr_dfs

def build_master_instr_df(df_pivot: pd.DataFrame, data_folder: Path):
    '''
    Accepts a pd.DataFrame reporting the file integrity status of experiments in the data_folder.
    The dataframe is produced by the review_matched_filenames() function.
    
    When executed, build_master_instr_df() will iterate on every experiment with all requisite files and 
    extract all Quantstudio .eds exported data from the tables ['Raw Data', 'Amplification Data', 
    'Multicomponent Data', 'Melt Curve Raw Data'] of each experiment.
    
    Parameters
    ----------
    df_pivot : pd.DataFrame
        A dataframe produced by review_matched_filenames()
    
    data_folder : Path
        A pathlib object pointing to ProxiPal's /data folder
        
    Returns
    ----------
    all_instr_dfs : pd.DataFrame
        Consolidates all .eds exported raw data from ProxiPal experiments into a single concatenated dataframe.
        Adds column "filepath_txt" for each experiment so that, with "well" all instrument data can be merged onto the mastertable.
    '''
    
    all_instr_dfs = pd.DataFrame()

    for idx, row in df_pivot.iterrows():
        if row['txt'] == True and row['csv'] == True:
            path_key = data_folder / (row['path_key'] + '.txt')
            instr_dict = extract_instr_tables(path_key)
            
            # Keep the existing sanity check
            if len(instr_dict) != 8:
                print('Warning: Not all instrument tables were exported. Open the QuantStudio .eds and re-export the .txt file below. This function will not complete if an eds export is incomplete.')
                print(path_key)
            
            # REQUIRED tables
            required_keys = ['Raw Data', 'Amplification Data', 'Multicomponent Data']
            # OPTIONAL tables
            optional_keys = ['Melt Curve Raw Data']
            
            missing_required = [key for key in required_keys if key not in instr_dict]
            if missing_required:
                raise KeyError(f"Missing required keys {', '.join(missing_required)} in file {path_key}")
            
            # Log but do not fail if optional tables are missing
            missing_optional = [key for key in optional_keys if key not in instr_dict]
            if missing_optional:
                print(f"Warning: missing optional keys {', '.join(missing_optional)} in file {path_key}")
            
            instr_df = build_instr_df(instr_dict)
            instr_df['filepath_txt'] = str(row['path_key'] + '.txt')
            all_instr_dfs = pd.concat([all_instr_dfs, instr_df], axis=0)

    return all_instr_dfs


# Usage
# master_instr_df = build_master_instr_df(df_pivot, data_folder)
# To merge on mastertable
# mastertable_updated = pd.merge(mastertable, master_instr_df, how='inner', on=['well', 'filepath_txt'])

def extract_residuals(df: pd.DataFrame, x_concentration: str, y_residuals: str) -> pd.DataFrame:
    '''
    __GPT4 request:__

    Example:
    def extract_residuals(df, x_concentration, y_residuals):
        * code here  

    Where:
    df is a pandas table (example: the one submitted with this request)
    x_concentration, y_residuals are columns not labeled as such in the table; but will be specified by the user when calling the function

    The function should 
    * perform a log 10 of the x_concentration values prior to plotting and use the x axis for these values, labeled as Log10[column name]
    * put the y_residuals on the y axis, labeled as per the [column name]
    * a new column y_residuals_mean should be calculated wherein experimental replicates are identified and the mean calculated to derive the y_residual_mean for the group

    Rows are deemed experimental replicates belonging to the same group if
    "filepath_txt" is identical
    "rep_id" is identical
    "target" is identical
    "sample_id" is identical

    Then function should return a table with the columns
    "sample_id", "x_concentration", "Log10[x_concentration]", "y_residuals", "y_residuals_mean" "calibrator", "target", "rep_id", "var_1", "var_2", "var_3", "var_4", "var_1 (desc)", "var_2 (desc)", "var_3 (desc)", "var_4 (desc)", "analysis_uuid", "filepath_txt"

    Provide full comments and a method signature to your supplied function. 

    How to use:
    residual_table = extract_residuals(mastertable, "py_known_conc", "py_raw_recovery")
    '''
    df = df.copy()
    
    # Perform a log10 transformation of the x_concentration column
    # Only apply np.log10 to positive values to avoid the division by zero warning
    df['Log10[{}]'.format(x_concentration)] = df[x_concentration].apply(lambda x: np.log10(x) if x > 0 else np.nan)

    # Identify experimental replicates and calculate the mean of y_residuals
    df['y_residuals_mean'] = df.groupby(['filepath_txt', 'rep_id', 'target', 'sample_id'])[y_residuals].transform('mean')

    # Filter the DataFrame to contain only the necessary columns
    df = df[['sample_id', x_concentration, 'Log10[{}]'.format(x_concentration), y_residuals, 'y_residuals_mean',
             'calibrator', 'target', 'rep_id', 'var_1', 'var_2', 'var_3', 'var_4',
             'var_1 (desc)', 'var_2 (desc)', 'var_3 (desc)', 'var_4 (desc)',
             'analysis_uuid', 'filepath_txt']]

    return df

# Usage
# residuals_table_normal_progen = extract_residuals(master_progen_normal, "py_known_conc", "py_raw_recovery")
# residuals_table_normal_progen.head(1)

def plot_residuals(df: pd.DataFrame, x_concentration: str, y_residuals: str, 
                   remove_duplicates: bool = True, y_intercept: float = 1, 
                   addtrace: bool = True) -> pd.DataFrame:
    """
    This function creates an interactive plot using plotly. It plots the x_concentration against the y_residuals and
    draws a horizontal line at y_intercept. 

    __GPT4 request:__

    Example:
    def plot_residuals(df, x_concentration, y_residuals, remove_duplicates = TRUE, y_intercept = 1):
            * code here  

    Where:
    df is a pandas table (example: the one submitted with this request)
    x_concentration, y_residuals are columns not labeled as such in the table; but will be specified by the user when calling the function
    y_intercept is a black horizontal line drawn thorugh the given value

    if remove_duplicates = TRUE, duplicate rows will be collapsed into one row prior to plotting. Rows are deemed duplicates if:
    "x_concentration" is identical
    "y_residuals" is identical
    "filepath_txt" is identical
    "rep_id" is identical
    "target" is identical
    "sample_id" is identical
    "calibrator" is identical

    The function should:
    * Use plotly to draw an interactive dot plot and table of x_concentration vs y_residuals.
    * Return a dataframe of the plotted values

    How to use:
    residuals_fig, residuals_figdf = plot_residuals(residuals_table_normal_progen, "Log10[py_known_conc]", "y_residuals_mean")
    """
    
    # Make a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()

    # Remove duplicates if required
    if remove_duplicates:
        df = df.drop_duplicates(subset=[x_concentration, y_residuals, 'filepath_txt', 'rep_id', 'target', 'sample_id', 'calibrator'])

    # Create a figure
    fig = go.Figure()

    # Determine the mode based on whether lines should be added or not
    mode = 'lines+markers' if addtrace else 'markers'

    # Create a new column with the first 4 characters of 'sample_id'
    df['sample_id_start'] = df['sample_id'].str[:4]

    # Group by "filepath_csv" and "sample_id_start" and plot each group separately
    for (filepath_csv, sample_id_start), group in df.groupby(['filepath_txt', 'sample_id_start']):
        fig.add_trace(go.Scatter(
            x=group[x_concentration], y=group[y_residuals], mode=mode, marker=dict(size=5), name=f'{filepath_csv}_{sample_id_start}'))

    # Add the intercept line if y_intercept is a float or an integer
    if isinstance(y_intercept, (float, int)):
        fig.add_trace(go.Scatter(
            x=[df[x_concentration].min(), df[x_concentration].max()], y=[y_intercept, y_intercept], mode='lines', name='Intercept Line', line=dict(color='black')))

    # Update layout to move legend below plot
    fig.update_layout(
        xaxis_title=x_concentration,
        yaxis_title=y_residuals,
        height=650,
        legend=dict(y=-0.35, orientation='h')
    )

    return fig, df

# Usage
# recovery_residuals_progen_normal_fig, recovery_residuals_progen_normal_figdf = plot_residuals(residuals_table_normal_progen, "Log10[py_known_conc]", "y_residuals_mean")


def filter_expt_var(dataframe: pd.DataFrame, dictionary: dict) -> pd.DataFrame:
    """
    This function accepts a DataFrame and a dictionary. The function filters the DataFrame based on the rules specified 
    in the dictionary. For each key-value pair in the dictionary, the function drops rows from the DataFrame where the 
    'filepath_txt' matches the 'filepath_txt' of the row indicated by the key, and where the corresponding variable(s) 
    (var_2, var_3, var_4) have a value of 1.
    
    When working with modified dataframes ensure that your index designations match that of the dictionary.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame to filter.
    dictionary (dict): The dictionary specifying the filtering rules. The dictionary keys correspond to DataFrame row 
    indices, and the values are lists of integers (1, 2, 3, 4) corresponding to the variable(s) that should have a value 
    of 1 for the row to be dropped.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """

    # Create a copy of the dataframe to prevent changes to the original dataframe
    df = dataframe.copy()

    # Iterate over the dictionary items
    for key, values in dictionary.items():
        # Get the 'filepath_txt' of the row indicated by the key
        filepath_txt = df.loc[key, 'filepath_txt']

        # Iterate over the values (the variables)
        for val in values:
            # Form the variable name
            var = f'var_{val}'

            # Drop the rows from the dataframe where 'filepath_txt' matches the 'filepath_txt' of the row indicated by the key, 
            # and where the corresponding variable has a value of 1
            df = df.drop(df[(df['filepath_txt'] == filepath_txt) & (df[var] == 1)].index)
    
    # Return the filtered dataframe
    return df


def rdml_linreg_analyse(filepath, export = False):
    """
    Analyze RDML file using linear regression and annotate the metatable with the analysis results.
    
    Parameters:
    - filepath: Path object pointing to the location of the RDML file.
    
    Returns:
    - A pandas DataFrame containing the annotated metatable.
    """
    
    # Read in the RDML file
    file = rdml.Rdml(str(filepath))

    # Validate RDML version
    cli_resValidate = rdml.Rdml.validate(file)
    if 'RDML file is valid' in cli_resValidate.split('\t')[-1]:
        print(filepath, ': ', cli_resValidate)
        version = cli_resValidate.split('\t')[2][:3]

        if version in ['1.1', '1.2', '1.3']:
            cli_linRegPCR = file
            cli_exp = cli_linRegPCR.experiments()[0]
            cli_run = cli_exp.runs()[0]
    
            with suppress_print():
                cli_result = cli_run.linRegPCR(updateRDML=True, saveResultsCSV=True, timeRun=True, verbose=True)
                
            cli_linRegPCR.save(str(filepath.with_name(filepath.stem + '_rdmlLinReg.rdml')))

            # Read results into pandas
            data = StringIO(cli_result["resultsCSV"])
            df = pd.read_csv(data, delimiter='\t')
            
            # Save a copy
            df.to_csv(filepath.parent / 'exports' / 'rdml_output.csv', index=False)
            
            # Prepare df for merge on metatable
            df.columns = 'rdml_' + df.columns
            df.rename(columns={'rdml_id': 'well', 
                               'rdml_well': 'position'}, inplace=True)
                
            # Find columns containing 'rdml_N0' and create log2 columns
            n0_columns = [col for col in df.columns if 'rdml_N0' in col]
            for col in n0_columns:
                new_col = col.replace('rdml_N0', 'rdml_log2N0')
                df[new_col] = -1*np.log2(df[col])

            # Convert well column to string for consistency during merging
            df['well'] = df['well'].astype(str)
            
        elif version == '1.0':
            print('rdmlpython does not support rdml v1.0; convert files to v1.1 or 1.2 then try again using https://www.gear-genomics.com/rdml-tools/edit.html')
            return
    else:
        print('Unable to confirm if ' + str(filepath) + 'is a valid file')
        return
    
    # Read in data metatable
    if Path(filepath.parent / 'exports' / 'metatable.csv').exists():
        curr_metatable = pd.read_csv(filepath.parent / 'exports' / 'metatable.csv')
    
    # Convert well column to string for consistency during merging
    curr_metatable['well'] = curr_metatable['well'].astype(str)
    
    # Check if the metatable already has rdml annotations
    cols_to_drop = [col for col in curr_metatable.columns if 'rdml_' in col]
    if cols_to_drop:
        print('It looks like the py_metatable has already been annotated for rdml calculations\nDropping existing rdml annotations and recalculating')
        curr_metatable.drop(columns=cols_to_drop, inplace=True)

    # Left join df on metatable using 'well' and 'position' columns and save
    rdml_metatable = curr_metatable.merge(df, on=['well', 'position'], how='left')

    if export == True:
        rdml_metatable.to_csv(filepath.parent / 'exports' / 'py_metatable.csv', index=False)
        
    return rdml_metatable


# The functions below were written to avoid float > str conversions throwing up long decimal place formats due to character encoding. 
# The function should now respect the float format when converting to str 
def format_number(val):
    if isinstance(val, float):
        return format_float(val)
    else:
        return format(val)


def format_float(val):
    if val.is_integer():
        return "{:.0f}".format(val)  # No decimal places for integer values
    else:
        return "{:.12g}".format(val)  # Use general format to preserve original decimal places


def join_unique(x):
    formatted_vals = []

    for val in x.drop_duplicates():
        if isinstance(val, (int, float)):
            formatted_val = format_number(val)  # Format numeric values
        else:
            formatted_val = str(val)  # Convert non-numeric values to string

        formatted_vals.append(formatted_val)

    return ' / '.join(formatted_vals)


# Below add_py_known_calc differs from the below in that py_known_calc == 0 is not replaced with np.nan
def add_py_known_conc(df):
    """
    Add concentration-related columns to the metadata DataFrame.

    This version:
    - Extracts concentrations from sample_id strings like 'std1[100]_...'
    - Prints a warning for non-string or missing sample_id values, reporting their index and value
    - Suggests that users assign text-based sample_id values (not numeric or NaN)
    - Keeps py_known_conc == 0 (does NOT replace with NaN)
    - Only replaces zeros with NaN when computing log10 (to avoid -inf)
    """
    df = df.copy()

    # Add py_known_conc column if not present
    if 'py_known_conc' not in df.columns:
        conc_list = []
        pattern_std = r"std\d+\[(.*?)\]_"

        for idx, s in enumerate(df['sample_id']):
            # Skip NaN or non-string sample_id values, report them
            if pd.isna(s) or not isinstance(s, str):
                print(f"⚠️  Warning: Row {idx} has invalid sample_id = {s!r}. "
                      f"Each sample_id should be a text label, not numeric or N/A.")
                conc_list.append(np.nan)
                continue

            # Try to extract concentration value from string pattern
            match = re.search(pattern_std, s)
            if match:
                try:
                    conc_list.append(float(match.group(1)))
                except ValueError:
                    print(f"⚠️  Warning: Row {idx} sample_id = {s!r} "
                          f"contains a non-numeric concentration inside brackets.")
                    conc_list.append(np.nan)
            else:
                conc_list.append(np.nan)

        df['py_known_conc'] = conc_list

    # Add py_known_conc_log10 column if not present
    if 'py_known_conc_log10' not in df.columns:
        temp_conc = df['py_known_conc'].copy()
        temp_conc = temp_conc.replace({0: np.nan})  # avoid log10(0)
        log10_py_known_conc = np.log10(temp_conc)
        df['py_known_conc_log10'] = np.where(np.isinf(log10_py_known_conc),
                                             np.nan, log10_py_known_conc)

    return df


# def add_py_known_conc(df):
#     """
#     Add concentration-related columns to the metadata DataFrame.
#     """
#     df = df.copy()
    
#     # Add py_known_conc column if not present
#     if 'py_known_conc' not in df.columns:
#         conc_list = []
#         pattern_std = r"std\d+\[(.*?)\]_"
#         for s in df['sample_id'].tolist():
#             match = re.search(pattern_std, s)
#             if match:
#                 conc_list.append(float(match.group(1)))
#             else:
#                 conc_list.append(np.nan)
        
#         df['py_known_conc'] = conc_list
    
#     # Add py_known_conc_log10 column if not present    
#     if 'py_known_conc_log10' not in df.columns:
#         # Create a temporary series for log transformation, keeping original py_known_conc intact
#         temp_conc = df['py_known_conc'].copy()
#         temp_conc = temp_conc.replace({0: np.nan})  # Only replace zeros for log calculation
#         log10_py_known_conc = np.log10(temp_conc)
#         df['py_known_conc_log10'] = np.where(np.isinf(log10_py_known_conc), np.nan, log10_py_known_conc)

#     return df


# def extract_experiment_tables(df, filepath_csv, quant_model='SLR', threshold_type='ct', 
#                             transform_x='log10(x)', std0_status='exc_std0', 
#                             simple_headers=False, sample_type='standards'):
#     """
#     Extract experiment tables from the metatable with support for standards, samples, and wells.
    
#     Parameters
#     ----------
#     df : pandas.DataFrame
#         Input metatable
#     filepath_csv : str
#         Path to CSV file
#     quant_model : str, optional
#         Quantification model (e.g., 'SLR', '4PL', or '5PL'). Default is 'SLR'
#     threshold_type : str, optional
#         Type of threshold (e.g., 'Ct'). Default is 'Ct'
#     transform_x : str, optional
#         Type of x transformation (e.g., 'log10(x)', 'linear(x)')
#     std0_status : str, optional
#         Status of std0 (e.g., 'exc_std0', 'inc_std0')
#     simple_headers : bool, optional
#         Whether to simplify column headers. Default is False
#     sample_type : str, optional
#         Type of data to extract ('standards', 'samples', or 'wells'). Default is 'standards'
    
#     Returns
#     -------
#     dict
#         Dictionary containing calculation, plot, and report tables for standards or samples,
#         or a single wells table when sample_type='wells'
#     """
#     if sample_type not in ['standards', 'samples', 'wells']:
#         raise ValueError("sample_type must be either 'standards', 'samples', or 'wells'")

#     # Define relevant columns
#     threshold_mean = threshold_type + '; mean'

#     # Handle column name pattern based on model and parameters
#     if quant_model == 'SLR':
#         if '_N0' not in threshold_type:
#             transform_x = 'log10(x)'
#             std0_status='exc_std0'
#         elif '_N0' in threshold_type:
#             transform_x = 'linear(x)'
        
#     col_pattern = f"{quant_model}; {transform_x}; {std0_status}; {threshold_type}"

#     # Filter columns based on pattern
#     col_list = [col for col in df.columns if col_pattern in col]
    
#     if not col_list and sample_type != 'wells':
#         raise ValueError(f"No columns found matching pattern: {col_pattern}")

#     # Filter relevant df
#     exp_df = df[df['filepath_csv']==filepath_csv]

#     if sample_type == 'wells':
#         # Create wells table with specific columns
#         wells_columns = ['position', 'target', 'usr_ignore', 
#                          threshold_type, f"{threshold_type}; mean",
#                          'tm', 'rep_id', 'dilution', 'usr_raw_ng/L',
#                          f"{quant_model}; {transform_x}; {std0_status}; {threshold_type}; raw_ng/L",
#                          f"{quant_model}; {transform_x}; {std0_status}; {threshold_type}; mean_ng/L",
#                          "calibrator"]
        
#         # Check for missing columns
#         missing_cols = set(wells_columns) - set(exp_df.columns)
#         if missing_cols:
#             raise ValueError(f"Missing columns for wells table: {missing_cols}")
        
#         wells_table = exp_df[wells_columns].copy()
        
#         # Round numeric columns to 2 decimal places
#         float_cols = wells_table.select_dtypes(include=['float64']).columns
#         wells_table[float_cols] = wells_table[float_cols].round(2)
        
#         # Store in dictionary with a single key
#         table_dict = {'wells_table': wells_table}

#     elif sample_type == 'samples':
#         # Create sample table with specific columns as in calc_metatable_SLR
#         sample_table = exp_df[['sample_id', 'tube_id', 'position', 'target', 
#                               'usr_ignore', 
#                               threshold_type, 
#                               f"{threshold_type}; mean", 
#                               'usr_mean_ng/L', 
#                               f"{quant_model}; {transform_x}; {std0_status}; {threshold_type}; mean_ng/L",
#                               'calibrator']]

#         # Round all float columns to 2 decimal places
#         float_cols = sample_table.select_dtypes(include=['float64']).columns
#         sample_table[float_cols] = sample_table[float_cols].round(2)

#         # Remove standard samples
#         std_pattern = r"std\d+\[(.*?)\]_"
#         sample_table = sample_table[~sample_table['sample_id'].str.contains(std_pattern)]

#         # Convert IDs to strings for grouping
#         sample_table['tube_id'] = sample_table['tube_id'].astype(str)
#         sample_table['sample_id'] = sample_table['sample_id'].astype(str)

#         # Group and collapse duplicate entries
#         grouped = sample_table.groupby(['sample_id', 'tube_id'])
#         collapsed_df = grouped.agg(join_unique).reset_index()

#         # Store in dictionary with a single key
#         table_dict = {'sample_report_table': collapsed_df}

#     else:  # sample_type == 'standards'
#         # Include extra relevant cols
#         col_list = ['rdml_indiv PCR eff', 'rdml_mean PCR eff', 'py_known_conc', 
#                     'py_known_conc_log10', threshold_mean] + col_list

#         # Check col_list in df
#         missing_cols = set(col_list) - set(exp_df.columns)
#         if missing_cols:
#             raise ValueError(f"""Missing columns: {missing_cols}\n
#                              Columns for your selected quant_model and threshold_type have not been generated. 
#                              Run calc_metatable() for your chosen metrics first.""")

#         # Initialize dictionary for standards
#         table_dict = {}

#         # Process standards
#         stds_used_list = exp_df['usr_std'].unique().tolist()
        
#         for i in stds_used_list:
#             std = 'std' + str(i)
#             pattern = r"{}\[.*?\]_".format(std)

#             # Create metatable subset on pattern
#             std_df = exp_df[exp_df['sample_id'].str.contains(pattern, regex=True)]

#             # Limit std_df to columns of interest
#             std_df = std_df.loc[:, ['position', 'eds_id', 'target', threshold_type, 'tm', 
#                                    'sample_id', 'tube_id', 'rep_id', "dilution", 
#                                    "usr_std", 'usr_ignore'] + col_list]
            
#             # Handle formatting
#             if '_N0' in threshold_type:
#                 n0_cols = [col for col in std_df.columns if 'rdml_N0' in col]
                
#                 for col in n0_cols:
#                     std_df[col] = pd.to_numeric(std_df[col], errors='coerce')
                    
#                     if 'ng/L' not in col and 'recovery' not in col:
#                         std_df[col] = std_df[col].apply(
#                             lambda x: '{:.2e}'.format(float(x)) if pd.notnull(x) else x
#                         )
#                     else:
#                         std_df[col] = std_df[col].round(2)

#                 numeric_cols = std_df.select_dtypes(include=['float64', 'int64']).columns
#                 other_cols = [col for col in numeric_cols if col not in n0_cols]
#                 std_df[other_cols] = std_df[other_cols].round(2)
#             else:
#                 std_df = std_df.round(2)

#             # Process trend and report dataframes
#             trend_df = std_df.groupby('sample_id').agg(join_unique).reset_index()
#             trend_df['py_known_conc'] = pd.to_numeric(trend_df['py_known_conc'])
#             trend_df = trend_df.sort_values('py_known_conc', ascending=True)

#             mean_ngl_cols = [col for col in trend_df.columns if 'mean_ng/L' in col and col_pattern in col]
#             mean_recovery_cols = [col for col in trend_df.columns if 'mean_recovery' in col and col_pattern in col]

#             if not mean_ngl_cols or not mean_recovery_cols:
#                 print(f"Available columns: {trend_df.columns.tolist()}")
#                 raise ValueError(f"Could not find required columns matching pattern: {col_pattern}")
                
#             mean_ngl_col = mean_ngl_cols[0]
#             mean_recovery_col = mean_recovery_cols[0]
#             raw_ngl_col = [col for col in trend_df.columns if 'raw_ng/L' in col and col_pattern in col][0]

#             report_df = trend_df[['sample_id', 'position', threshold_type, threshold_mean, 
#                                  'py_known_conc', raw_ngl_col, mean_ngl_col, mean_recovery_col]]

#             table_dict[std + '_calc_table'] = std_df
#             table_dict[std + '_plot_table'] = trend_df
#             table_dict[std + '_report_table'] = report_df

#     # Handle simple headers if requested
#     if simple_headers:
#         print(f"""User setting simple_headers = True. Therefore the parameters: f"quant_model = {quant_model}; transform_x = {transform_x}; std0_status = {std0_status}; threshold_type = {threshold_type}; have been removed from column headers""")
        
#         for key, value in table_dict.items():
#             if value.columns.str.contains('; ').any():
#                 new_columns = {col: col.split('; ')[-1] for col in value.columns}
#                 table_dict[key] = value.rename(columns=new_columns)

#     return table_dict

def extract_experiment_tables(df, filepath_csv, quant_model='SLR', threshold_type='ct', 
                            transform_x='log10(x)', std0_status='exc_std0', 
                            simple_headers=False, sample_type='standards',
                            custom_columns=None):
    """
    Extract experiment tables from the metatable with support for standards, samples, and wells.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input metatable
    filepath_csv : str
        Path to CSV file
    quant_model : str, optional
        Quantification model (e.g., 'SLR', '4PL', or '5PL'). Default is 'SLR'
    threshold_type : str, optional
        Type of threshold (e.g., 'Ct'). Default is 'Ct'
    transform_x : str, optional
        Type of x transformation (e.g., 'log10(x)', 'linear(x)')
    std0_status : str, optional
        Status of std0 (e.g., 'exc_std0', 'inc_std0')
    simple_headers : bool, optional
        Whether to simplify column headers. Default is False
    sample_type : str, optional
        Type of data to extract ('standards', 'samples', or 'wells'). Default is 'standards'
    custom_columns : list, optional
        List of column names to use instead of default columns. Default is None.
    
    Returns
    -------
    dict
        Dictionary containing calculation, plot, and report tables for standards or samples,
        or a single wells table when sample_type='wells'
    """
    if sample_type not in ['standards', 'samples', 'wells']:
        raise ValueError("sample_type must be either 'standards', 'samples', or 'wells'")

    # Define relevant columns
    threshold_mean = threshold_type + '; mean'

    # Handle column name pattern based on model and parameters
    if quant_model == 'SLR':
        if '_N0' not in threshold_type:
            transform_x = 'log10(x)'
            std0_status='exc_std0'
        elif '_N0' in threshold_type:
            transform_x = 'linear(x)'
        
    col_pattern = f"{quant_model}; {transform_x}; {std0_status}; {threshold_type}"

    # Filter columns based on pattern
    col_list = [col for col in df.columns if col_pattern in col]
    
    if not col_list and sample_type != 'wells':
        raise ValueError(f"No columns found matching pattern: {col_pattern}")

    # Filter relevant df
    exp_df = df[df['filepath_csv']==filepath_csv].copy()

    if custom_columns:
        # Validate custom columns exist in the dataframe
        missing_cols = set(custom_columns) - set(exp_df.columns)
        if missing_cols:
            raise ValueError(f"Custom columns not found in dataframe: {missing_cols}")

    if sample_type == 'wells':
        # Use custom columns if provided, otherwise use default columns
        wells_columns = custom_columns if custom_columns else [
            'position', 'target', 'usr_ignore', 
            threshold_type, f"{threshold_type}; mean",
            'tm', 'rep_id', 'dilution', 'usr_raw_ng/L',
            f"{quant_model}; {transform_x}; {std0_status}; {threshold_type}; raw_ng/L",
            f"{quant_model}; {transform_x}; {std0_status}; {threshold_type}; mean_ng/L",
            "calibrator"
        ]
        
        # Check for missing columns
        missing_cols = set(wells_columns) - set(exp_df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns for wells table: {missing_cols}")
        
        wells_table = exp_df[wells_columns].copy()
        
        # Round numeric columns to 2 decimal places
        float_cols = wells_table.select_dtypes(include=['float64']).columns
        wells_table[float_cols] = wells_table[float_cols].round(2)
        
        table_dict = {'wells_table': wells_table}

    elif sample_type == 'samples':
        # Use custom columns if provided, otherwise use default columns
        sample_columns = custom_columns if custom_columns else [
            'sample_id', 'tube_id', 'position', 'target', 
            'usr_ignore', threshold_type, 
            f"{threshold_type}; mean", 
            'usr_mean_ng/L',
            f"{quant_model}; {transform_x}; {std0_status}; {threshold_type}; mean_ng/L",
            'calibrator'
        ]
        
        # Ensure sample_id is in columns if we need to filter standards and custom_columns is provided
        if custom_columns and 'sample_id' not in sample_columns:
            sample_columns = sample_columns + ['sample_id']
            
        sample_table = exp_df[sample_columns].copy()

        # Round all float columns to 2 decimal places
        float_cols = sample_table.select_dtypes(include=['float64']).columns
        sample_table[float_cols] = sample_table[float_cols].round(2)

        # Remove standard samples using pandas string operations
        std_pattern = r'std\d+\[.*?\]_'
        mask = ~sample_table['sample_id'].str.match(std_pattern, na=False)
        sample_table = sample_table[mask]
        
        # If sample_id was added only for filtering, remove it now
        if custom_columns and 'sample_id' not in custom_columns:
            sample_table = sample_table[custom_columns]

        # Convert IDs to strings for grouping only if needed
        if 'tube_id' in sample_columns:
            sample_table['tube_id'] = sample_table['tube_id'].astype(str)
        if 'sample_id' in sample_columns:
            sample_table['sample_id'] = sample_table['sample_id'].astype(str)

        # Group and collapse duplicate entries if sample_id and tube_id are present
        if 'sample_id' in sample_columns and 'tube_id' in sample_columns:
            grouped = sample_table.groupby(['sample_id', 'tube_id'])
            collapsed_df = grouped.agg(join_unique).reset_index()
        else:
            collapsed_df = sample_table

        table_dict = {'sample_report_table': collapsed_df}

    else:  # sample_type == 'standards'
        if custom_columns:
            col_list = custom_columns
        else:
            # Include extra relevant cols
            col_list = ['rdml_indiv PCR eff', 'rdml_mean PCR eff', 'py_known_conc', 
                        'py_known_conc_log10', threshold_mean] + col_list

        # Check col_list in df
        missing_cols = set(col_list) - set(exp_df.columns)
        if missing_cols:
            raise ValueError(f"""Missing columns: {missing_cols}\n
                             Columns for your selected quant_model and threshold_type have not been generated. 
                             Run calc_metatable() for your chosen metrics first.""")

        # Initialize dictionary for standards
        table_dict = {}

        # Process standards
        stds_used_list = exp_df['usr_std'].unique().tolist()
        
        for i in stds_used_list:
            std = 'std' + str(i)
            pattern = r"{}\[.*?\]_".format(std)

            # Create metatable subset on pattern
            std_df = exp_df[exp_df['sample_id'].str.contains(pattern, regex=True)].copy()

            # Use custom columns if provided, otherwise use default columns
            std_columns = custom_columns if custom_columns else [
                'position', 'eds_id', 'target', threshold_type, 'tm', 
                'sample_id', 'tube_id', 'rep_id', "dilution", 
                "usr_std", 'usr_ignore'
            ] + col_list
            
            # Limit std_df to columns of interest
            std_df = std_df.loc[:, std_columns]
            
            # Handle formatting
            if '_N0' in threshold_type:
                n0_cols = [col for col in std_df.columns if 'rdml_N0' in col]
                
                for col in n0_cols:
                    std_df[col] = pd.to_numeric(std_df[col], errors='coerce')
                    
                    if 'ng/L' not in col and 'recovery' not in col:
                        std_df[col] = std_df[col].apply(
                            lambda x: '{:.2e}'.format(float(x)) if pd.notnull(x) else x
                        )
                    else:
                        std_df[col] = std_df[col].round(2)

                numeric_cols = std_df.select_dtypes(include=['float64', 'int64']).columns
                other_cols = [col for col in numeric_cols if col not in n0_cols]
                std_df[other_cols] = std_df[other_cols].round(2)
            else:
                std_df = std_df.round(2)

            # Process trend and report dataframes
            if 'sample_id' in std_columns:
                trend_df = std_df.groupby('sample_id').agg(join_unique).reset_index()
                if 'py_known_conc' in trend_df.columns:
                    trend_df['py_known_conc'] = pd.to_numeric(trend_df['py_known_conc'])
                    trend_df = trend_df.sort_values('py_known_conc', ascending=True)

                # Only create report_df if all required columns are present
                required_report_cols = ['sample_id', 'position', threshold_type, threshold_mean,
                                      'py_known_conc']
                if all(col in trend_df.columns for col in required_report_cols):
                    mean_ngl_cols = [col for col in trend_df.columns if 'mean_ng/L' in col and col_pattern in col]
                    mean_recovery_cols = [col for col in trend_df.columns if 'mean_recovery' in col and col_pattern in col]
                    raw_ngl_cols = [col for col in trend_df.columns if 'raw_ng/L' in col and col_pattern in col]

                    if mean_ngl_cols and mean_recovery_cols and raw_ngl_cols:
                        report_df = trend_df[required_report_cols + [
                            raw_ngl_cols[0], mean_ngl_cols[0], mean_recovery_cols[0]
                        ]]
                    else:
                        report_df = trend_df[required_report_cols]
                else:
                    report_df = trend_df

            table_dict[std + '_calc_table'] = std_df
            if 'sample_id' in std_columns:
                table_dict[std + '_plot_table'] = trend_df
                table_dict[std + '_report_table'] = report_df

    # Handle simple headers if requested
    if simple_headers:
        print(f"""User setting simple_headers = True. Therefore the parameters: f"quant_model = {quant_model}; transform_x = {transform_x}; std0_status = {std0_status}; threshold_type = {threshold_type}; have been removed from column headers""")
        
        for key, value in table_dict.items():
            if value.columns.str.contains('; ').any():
                new_columns = {col: col.split('; ')[-1] for col in value.columns}
                table_dict[key] = value.rename(columns=new_columns)

    return table_dict

## USAGE
# SLR_experiment_tables = extract_experiment_tables(df = SLR_metatable, filepath_csv = '240905_e82_kruti/240917/240905_NfL-e82_kruti.csv', quant_model = 'SLR', threshold_type = 'rdml_Cq (mean eff)', transform_x='log10(x)', std0_status='exc_std0', sample_type = 'standards', simple_headers=False)


def analyse_sample_linearity(df: pd.DataFrame, 
                             gradient_column: str = 'SLR; log10(x); exc_std0; ct; grad', 
                             concentration_column: str = 'SLR; log10(x); exc_std0; ct; mean_ng/L', 
                             checklist: list = None) -> list:
    """
    Identifies sample dilution series within the dataset and performs linear regression analysis
    for assessing linearity and parallelism.

    Samples are grouped based on shared metadata defined in the checklist. For each group with
    at least 3 unique dilution values (excluding standards), the function computes regression
    statistics using log10(inverse dilution) vs mean Ct.

    Parameters:
    - df (pd.DataFrame): The input dataframe containing dilution series data.
    - gradient_column (str): Name of the column containing externally computed gradients to include in output.
    - checklist (list[str], optional): List of column names used to identify unique dilution series groups.
                                       Defaults to a predefined set of metadata fields.

    Returns:
    - list[pd.DataFrame]: A list of DataFrames, each corresponding to a sample's dilution series,
                          annotated with regression statistics (slope, intercept, R², efficiency).
    """

    
    # Default checklist if none is provided
    if checklist is None:
        checklist = ['filepath_csv', 'sample_id', 'tube_id', 
                     'var_1', 'var_2', 'var_3', 'var_4', 
                     'var_1 (desc)', 'var_2 (desc)', 'var_3 (desc)', 'var_4 (desc)']
    
    # Copy the input dataframe and fill NA values in the checklist columns
    data_lintest = df.copy()
    data_lintest[checklist] = data_lintest[checklist].fillna('na')
    
    # Group data by the checklist columns
    grouped = data_lintest.groupby(checklist)
    filter_groups = [grouped.get_group(label) for label in grouped.groups.keys()]

    lin_list = []
    for df_group in filter_groups:
        # Filter groups based on the unique dilution values
        if len(df_group['dilution'].unique()) > 2:
            sub_group = df_group[['filename_csv', 'expt_folder_txt', gradient_column, 'sample_id', concentration_column, 
                                  'dilution', 'ct', 'specimen_type', 'collection_type']].copy()

            # Fill unavailable data for certain columns
            sub_group['specimen_type'] = sub_group['specimen_type'].fillna('unavailable')
            sub_group['collection_type'] = sub_group['collection_type'].fillna('unavailable')

            # Convert 'ct' column to numeric and compute mean by dilution
            sub_group['ct'] = pd.to_numeric(sub_group['ct'], errors='coerce')
            sub_group['ct_mean_y'] = sub_group.groupby('dilution')['ct'].transform('mean')
            sub_group.drop(columns=['ct'], inplace=True)
            sub_group.drop_duplicates(inplace=True)
            lin_list.append(sub_group)

    for df_lin in lin_list:
        # Compute inverse dilution and its log10 value
        df_lin['inv_dil'] = (1/df_lin['dilution'])*100
        df_lin['log10_inv_dil'] = np.log10(df_lin['inv_dil'])

        # Linear regression on log10 inverse dilution vs mean ct value
        slope, intercept, r_value, _, _ = linregress(df_lin['log10_inv_dil'], df_lin['ct_mean_y'])

        # Calculate efficiency based on the slope
        efficiency = (10**(-1/slope)) - 1

        # Update the dataframe with regression results
        df_lin['m'] = slope
        df_lin['c'] = intercept
        df_lin['efficiency'] = efficiency
        df_lin['R2'] = r_value**2  # R-squared value

    return lin_list


def sns_linearity_plot(df, gradient_column: str = 'SLR; log10(x); exc_std0; ct; grad', 
                       origin='[x2,y2]', inv_y=False, inv_x=False, xlab=None, ylab=None, ax=None, xtick="log10_inv_fil"):
    """
    Take a dataframe produced by analyse_sample_linearity() and visualise sample dilution with a lineplot.
    Overlaid with the lineplot is a representation of the standard curve that was applied to the same sample 
    in the same experiment (red); and a ideal model fit region for a PCR-efficiency of 0.85 - 1.05 across the same 
    dilution range (yellow). Both the latter are calculated from the gradient but obviously will be shifted in the
    x,y dimensions for the overlay to apply.
    
    Various axis manipulations are supplied so that plots can be made more interpretable to a non-PCR audience.
    
    Parameters:
    - df (DataFrame): Input data containing 'log10_inv_dil', 'ct_mean_y', and other relevant columns.
    - gradient_column (str): Name of the column containing externally computed gradients to include in output.
    - origin (str): Determines the starting point of the red line. Options are '[x1,y1]' or '[x2,y2]'.
    - inv_y (bool): If True, the y-axis will be inverted.
    - inv_x (bool): If True, the x-axis will be inverted.
    - xlab (str): Label for the x-axis.
    - ylab (str): Label for the y-axis.
    - ax (matplotlib Axis): Optional axis on which to plot. If None, a new figure and axis will be created.
    - xtick (str): Determines the labeling of x-axis ticks. Options are 'log10_inv_fil' or 'dilution'.
    
    Returns:
    None
    """
    
    # Create a new plot if no axis is provided
    if ax is None:
        _, ax = plt.subplots(figsize=(10/3, 6/3))
        
    # Extract data for plotting
    x1 = min(df['log10_inv_dil'])
    x2 = max(df['log10_inv_dil'])
    m = df[gradient_column].iloc[0]
    
    # Define model gradients
    m1_model = -3.3208
    m2_model = -3.448
    
    # Calculate y-values based on the specified origin
    if origin == '[x1,y1]':
        y1 = max(df['ct_mean_y'])
        y2 = m * (x2 - x1) + y1
        y2_m1_model = m1_model * (x2 - x1) + y1
        y2_m2_model = m2_model * (x2 - x1) + y1
        ax.fill_between([x1, x2], [y1, y2_m1_model], [y1, y2_m2_model], color='orange', alpha=0.5)
        
    elif origin == '[x2,y2]':
        y2 = min(df['ct_mean_y'])
        y1 = m * x1 + (y2 - m * x2)
        y1_m1_model = y2 - m1_model * (x2 - x1)
        y1_m2_model = y2 - m2_model * (x2 - x1)
        ax.fill_between([x1, x2], [y1_m1_model, y2], [y1_m2_model, y2], color='orange', alpha=0.5)
        
    else:
        raise ValueError("Invalid value for 'origin'. Expected '[x1,y1]' or '[x2,y2]'")
    
    # Plot data and regression line
    sns.lineplot(data=df, x='log10_inv_dil', y='ct_mean_y', marker='X', markersize=9, color='black', ax=ax)
    ax.plot([x1, x2], [y1, y2], color='red', linewidth=1)
    
    # Adjust plot appearance
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    y_min, y_max = ax.get_ylim()
    x_min, x_max = ax.get_xlim()
    ax.set_yticks(np.arange(np.floor(y_min), np.ceil(y_max) + 1, 1))
    
    # Handle x-axis ticks and labels based on xtick parameter
    if xtick == "dilution":
        # The *100 modifier below is applied because the 'inv_dil' calculation made by analyse_linearity() 
        # uses a *100 modifier to yield a more readable arbitrary log10_inv_dil value.
        dilution_values = (1 / (10 ** df['log10_inv_dil'].values))*100
        ax.set_xticks(df['log10_inv_dil'])
        ax.set_xticklabels(['{:.0f}'.format(dil) for dil in dilution_values])

    else:
        ax.set_xticks(np.arange(np.floor(x_min), np.ceil(x_max) + 0.5, 0.5))
    
    ax.tick_params(axis='both', labelsize=9)
    title_part1 = df.iloc[0]['sample_id']
    title_part2 = df.iloc[0]['expt_folder_txt'].split('_k')[0]
    title = f"{title_part1}\n{title_part2}" if len(title_part1 + title_part2) > 30 else f"{title_part1}; {title_part2}"
    ax.set_title(title, fontsize=11)

    # Set x and y axis labels
    if xlab:
        ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)
    
    # Adjust text display based on inversion settings
    if inv_y and inv_x:
        text_to_display = df.iloc[0]['specimen_type'] + '\n' + df.iloc[0]['collection_type']
        ax.text(x_max - .1, y_max, text_to_display, fontsize=9, verticalalignment='bottom', horizontalalignment='left')
    elif inv_y and not inv_x:
        text_to_display = df.iloc[0]['specimen_type'] + '\n' + df.iloc[0]['collection_type']
        ax.text(x_max - 0.1, y_max, text_to_display, fontsize=9, verticalalignment='bottom', horizontalalignment='right')
    elif not inv_y and inv_x:
        text_to_display = df.iloc[0]['specimen_type'] + '\n' + df.iloc[0]['collection_type']
        ax.text(x_min + 0.1, y_min, text_to_display, fontsize=9, verticalalignment='bottom', horizontalalignment='right')
    elif not inv_y and not inv_x:
        text_to_display = df.iloc[0]['specimen_type'] + '\n' + df.iloc[0]['collection_type']
        ax.text(x_min + 0.1, y_min, text_to_display, fontsize=9, verticalalignment='bottom', horizontalalignment='left')
    
    # Invert y-axis and/or x-axis based on flags
    if inv_y:
        ax.invert_yaxis()
    if inv_x:
        ax.invert_xaxis()
        
    # Display the plot if no axis was provided
    if ax is None:
        plt.show()

# USAGE
# sns_linearity_plot(lin_list[0], inv_y = True, inv_x = True, origin = '[x2,y2]', xlab = 'Incr. dilution', ylab = 'Signal\n(lower is brighter)', xtick="dilution")

def calculate_replicate_delta(master_df: pd.DataFrame, header: pd.Series = 'ct') -> pd.DataFrame:
    """
    Calculates the replicate delta for each group in the master dataframe.
    
    From the master_df performs groupby 'rep_id' and 'filepath_csv'. For each group "replicate_delta" is calculated: 
    1. Where len(group) ==2, 'replicate_delta' is the the absolute difference between the two 'Ct' values
    2. Where len(group) ==1, 'replicate_delta' is np.nan
    3. Where len(group) >2, 'replicate delta' is the mean of all absolute differences calculated from all pairwise combinations of 'Ct' values.
    
    After merging 'replicate_delta' back to a copy of master_df the 'rep_%inc' is calculated thus:
    
    rep_%inc = ((2^(replicate_delta))-1)*100
    
    Because 'replicate delta' is always calculated as an absolute the 2^(replicate_delta) is always calculated in the % increase direction which applies the highest penalty to replicate deviation (0.3785 Cts = 30% difference)
    
    Parameters:
    - master_path (Path): Path object pointing to the master CSV file.
    
    Returns:
    - pd.DataFrame: A modified master dataframe with two extra columns.
        - 'replicate_delta; header', the Ct difference between replicates
        - 'rep_%inc; header' the equivalent % concentration increase between replicates of different Ct values 
    
    """
    
    # Function to calculate replicate_delta for each group
    def compute_delta(group):
        ct_values = group[header].values
        if len(ct_values) == 1:
            return np.nan
        elif len(ct_values) == 2:
            return abs(ct_values[0] - ct_values[1])
        else:
            # Compute all pairwise differences and return the mean
            pairwise_diffs = [abs(a-b) for a, b in combinations(ct_values, 2)]
            return np.mean(pairwise_diffs)
     
    # Calculate replicate_delta for each group
    deltas = master_df.groupby(['rep_id', 'filepath_csv']).apply(compute_delta).reset_index()
    
    # Drop any pre-existing columns
    if 'rep_delta' + '; ' + header in deltas.columns:
        deltas.drop(columns=['rep_delta' + '; ' + header], inplace=True)
    
    deltas.rename(columns={0: 'rep_delta' + '; ' + header}, inplace=True)
    
    # Merge the calculated deltas with the master dataframe
    master_copy = pd.merge(master_df, deltas, on=['rep_id', 'filepath_csv'], how='left')
    
    # Calculate the % effective difference
    master_copy['rep_%inc; ' + header] = (((2**(master_copy['rep_delta' + '; ' + header]))-1)*100).round(1)

    return master_copy


def generate_shewhart_table(df: pd.DataFrame, y_column: str, xrange: tuple = None, include_col: list = None) -> pd.DataFrame:
    """
    Generates a table of values for creating a Shewhart chart from the provided DataFrame.
    A Shewhart chart shows how values change over time. Time is extracted from the 
    'ExperimentRunStartTime' exported as .txt from the .eds file. We use this value to 
    differentiate between experiments that might be the same but have different analysis folders.
    The y_column defines the values to be plotted, and a +/- 2 stdev range is calculated.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    y_column (str): The name of the column to be used for the y-axis.
    xrange (tuple): A tuple of two strings representing the start and end dates for the x-axis range 
                    in the format ('YYYY-MM-DD', 'YYYY-MM-DD'). Default is None.
    include_col (list): A list of additional column names to be included in the final table. Default is None.

    Returns:
    pd.DataFrame: A DataFrame containing the relevant data for the Shewhart chart.
    """
    
    # Ensure working on a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Convert 'ExperimentRunStartTime' column to datetime and extract date only
    df['ExperimentRunStartTime'] = pd.to_datetime(df['ExperimentRunStartTime'], format='%Y-%m-%d %H:%M:%S')
    df['ExperimentRunDate'] = df['ExperimentRunStartTime'].dt.date

    # Filter data based on the provided xrange if specified
    if xrange:
        start_date = datetime.strptime(xrange[0], '%Y-%m-%d').date()
        end_date = datetime.strptime(xrange[1], '%Y-%m-%d').date()
        df = df[(df['ExperimentRunDate'] >= start_date) & (df['ExperimentRunDate'] <= end_date)]

    # Calculate mean and standard deviation of the specified y_column
    mean_y = df[y_column].mean()
    std_y = df[y_column].std()

    # Default columns to include
    default_columns = ['ExperimentRunDate', y_column, 'calibrator', 'filepath_csv', 'analysis_folder_csv', 'sample_id', 'dilution']
    
    # If additional columns are specified, include them
    if include_col:
        default_columns.extend(include_col)
        # Remove duplicates in case some columns are in both default and include_col
        default_columns = list(dict.fromkeys(default_columns))

    # Create a new DataFrame to store the values and statistics for plotting
    shewhart_table = df[default_columns].copy()
    shewhart_table['Mean'] = mean_y
    shewhart_table['+2 SD'] = mean_y + 2 * std_y
    shewhart_table['-2 SD'] = mean_y - 2 * std_y

    return shewhart_table

## Usage
## shewhart_table  = generate_shewhart_table(progen2000_df, y_column = 'ct', xrange=('2024-01-01', '2024-12-31'))


def plotly_shewhart_chart(shewhart_table: pd.DataFrame, y_column: str, title: str = 'Shewhart Chart', xrange: tuple = None, legend_col: str = 'calibrator') -> None:
    """
    Visualizes the Shewhart table using Plotly.

    Parameters:
    shewhart_table (pd.DataFrame): The DataFrame containing the Shewhart chart data.
    y_column (str): The name of the column to be used for the y-axis.
    title (str): The title of the chart. Default is 'Shewhart Chart'.
    xrange (tuple): A tuple of two strings representing the start and end dates for the x-axis range 
                    in the format ('YYYY-MM-DD', 'YYYY-MM-DD'). Default is None.

    Returns:
    None: Displays the Shewhart chart.
    """
    
    # Filter data based on the provided xrange if specified
    if xrange:
        start_date = datetime.strptime(xrange[0], '%Y-%m-%d').date()
        end_date = datetime.strptime(xrange[1], '%Y-%m-%d').date()
        shewhart_table = shewhart_table[(shewhart_table['ExperimentRunDate'] >= start_date) & (shewhart_table['ExperimentRunDate'] <= end_date)]

    # Create a scatter plot for the Shewhart chart
    fig = px.scatter(
        shewhart_table, 
        x='ExperimentRunDate', 
        y=y_column, 
        color=legend_col, 
        hover_data=['filepath_csv', 'sample_id', 'dilution'],
        title=title
    )

    # Add a line for the mean value of the specified y_column
    fig.add_trace(
        go.Scatter(
            x=shewhart_table['ExperimentRunDate'],
            y=shewhart_table['Mean'],
            mode='lines',
            name='Mean',
            line=dict(color='gray', dash='dash')
        )
    )

    # Add a line for +2 standard deviations from the mean
    fig.add_trace(
        go.Scatter(
            x=shewhart_table['ExperimentRunDate'],
            y=shewhart_table['+2 SD'],
            mode='lines',
            name='+2 SD',
            line=dict(color='red', dash='dot')
        )
    )

    # Add a line for -2 standard deviations from the mean
    fig.add_trace(
        go.Scatter(
            x=shewhart_table['ExperimentRunDate'],
            y=shewhart_table['-2 SD'],
            mode='lines',
            name='-2 SD',
            line=dict(color='red', dash='dot')
        )
    )

    # Customize the layout of the chart
    fig.update_layout(
        xaxis_title='Experiment Run Date',
        yaxis_title=y_column,
        legend_title=legend_col,
        paper_bgcolor='white',  # Set the background color of the entire chart
        plot_bgcolor='white',   # Set the background color of the plot area
        xaxis=dict(
            showgrid=True,      # Show gridlines on the x-axis
            gridcolor='lightgrey',  # Set the color of the x-axis gridlines
            showline=True,      # Show the x-axis line
            linecolor='black',  # Set the color of the x-axis line
        ),
        yaxis=dict(
            showgrid=True,      # Show gridlines on the y-axis
            gridcolor='lightgrey',  # Set the color of the y-axis gridlines
            showline=True,      # Show the y-axis line
            linecolor='black',  # Set the color of the y-axis line
        )
    )

    # Display the figure
    fig.show()
    
    
## USAGE
## plotly_shewhart_chart(shewhart_table, title = 'asdfa', y_column = 'ct')


def mann_kendall_test(shewhart_table: pd.DataFrame, y_column: str):
    """
    Performs the Mann-Kendall test on the y_column data from the Shewhart table using pymannkendall.
    Shewhart tables can be produced with generate_shewhart_table()
    
    ProxiPal implements the test as described by https://pypi.org/project/pymannkendall/

    Parameters:
    shewhart_table (pd.DataFrame): The Shewhart table containing the data.
    y_column (str): The name of the column to be tested.

    Returns:
    result: The result of the Mann-Kendall test.
    """
    # Extract the y_column data
    y_data = shewhart_table[y_column].values

    # Perform the Mann-Kendall test
    result = mk.original_test(y_data)

    return result

## USAGE
## mk_result = mann_kendall_test(shewhart_table, y_column='ct')


def extract_interassay_records(df: pd.DataFrame, search: [str, float, int], column: str, record: str) -> pd.DataFrame:
    """   
    Extracts and processes records from a conventional ProxiPal /data DataFrame, handling duplicates 
    and pivoting the data into a wideform format for interassay analysis. The function will analyse all submitted
    values so dataframes should be prefiltered for specific experimental conditions and calibration curves.

    This function filters the DataFrame for rows where a given search string is a substring 
    of the specified column. It handles duplicates by calculating the mean for the specified 
    record and returns a pivoted DataFrame with one row per 'filepath_csv' and columns 
    for each unique 'sample_id_(tube_id)'. The columns are sorted by the number of 
    non-NA values in descending order.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    search (str, float, int): The substring to search for in the specified column.
    column (str): The column name to search within.
    record (str): The record column for which the mean is calculated if duplicates are found.

    Returns:
    pd.DataFrame: A pivoted DataFrame with sorted columns based on the number of non-NA values and the filepath_csv date
    
    NB: Be very careful to prefilter the input dataframe for the specific variables you wish to test. 
    For instance, if you have 1 sample tested multiple times under several experimental variables only the results from one variable will be kept.
    The rule for keeping this value is on first occurrence only (see the line: # Keep only the first occurrence of each duplicate).
    In the wide format we cannot document all the non-routine variables thus, once produced, the opportunity for more subtle filtering is lost.
    Dilutional variables are tolerated and recorded in the 'sample_id_(tube_id)_dil' header
    """
    print("This function requires the user to be very careful to select the input dataframe for the specific experiment variables being tested.\
    For instance, if you have 1 sample tested multiple times under several experimental variables only the results from one variable will be kept.\
    The rule for keeping this value is on first occurrence only (see the line: # Keep only the first occurrence of each duplicate).\
    In the wide format we cannot document all the non-routine variables thus, once produced, the opportunity for more subtle filtering is lost.\
    Dilutional variables are tolerated and recorded in the 'sample_id_(tube_id)_dil' header.")
    
    # Convert search to string
    search_str = str(search)
    
    # Convert the relevant column to string and filter dataframe for rows where search is a substring
    filtered_df = df[df[column].astype(str).str.contains(search_str, na=False)].copy()
    
    # Fill NaN values in 'tube_id' with an empty string
    filtered_df.loc[:, 'tube_id'] = filtered_df['tube_id'].fillna('')
    
    # Create a new column 'sample_id_(tube_id)_dil'
    filtered_df.loc[:, 'sample_id_(tube_id)_dil'] = filtered_df['sample_id'] + '(' + filtered_df['tube_id'] + ')_dil=' + filtered_df['dilution'].astype(str)

    # Identify duplicates based on 'filepath_csv', 'sample_id_(tube_id)_dil', 'rep_id' and 'dilution'
    duplicates = filtered_df.duplicated(subset=['filepath_csv', 'sample_id_(tube_id)_dil', 'rep_id', 'dilution'], keep=False)
    
    # Calculate the mean for duplicates
    mean_df = filtered_df[duplicates].groupby(['filepath_csv', 'sample_id_(tube_id)_dil', 'rep_id','dilution'])[record].mean().reset_index()
    mean_df.rename(columns={record: record + '_mean'}, inplace=True)
    
    # Merge the mean values back to the original filtered dataframe
    filtered_df = pd.merge(filtered_df, mean_df, on=['filepath_csv', 'sample_id_(tube_id)_dil', 'rep_id', 'dilution'], how='left')
    
    # Fill NaN values in the mean column with the original record values for non-duplicates
    filtered_df.loc[:, record + '_mean'] = filtered_df[record + '_mean'].fillna(filtered_df[record])
    
    # Keep only the first occurrence of each duplicate
    filtered_df = filtered_df.drop_duplicates(subset=['filepath_csv', 'sample_id_(tube_id)_dil', 'rep_id','dilution'], keep='first')
    
    # Trim the dataframe to 'filepath_csv', 'sample_id_(tube_id)_dil', and the mean record column
    trimmed_df = filtered_df[['filepath_csv', 'sample_id_(tube_id)_dil', record + '_mean']]
    
    # Pivot the dataframe to have 'filepath_csv' as rows and 'sample_id_(tube_id)' as columns
    pivoted_df = trimmed_df.pivot_table(index='filepath_csv', columns='sample_id_(tube_id)_dil', values=record + '_mean', aggfunc='mean')
    
    # Sort columns by the number of non-NA values in descending order
    column_counts = pivoted_df.notna().sum().sort_values(ascending=False)
    sorted_df = pivoted_df[column_counts.index]
    
    # Flatten the columns and reset index
    sorted_df.columns = [str(col) for col in sorted_df.columns]
    sorted_df = sorted_df.reset_index()
    
    # Sort the DataFrame based on 'filepath_csv' column
    sorted_df = sorted_df.sort_values(by='filepath_csv')

    return sorted_df

## Usage
## record_df = extract_interassay_records(df = idt_abcam_data, search = 'progen', column = 'calibrator', record = 'ct')

def describe_interassay_drift(df: pd.DataFrame, row_labels = 'index') -> pd.DataFrame:
    """
    Accepts the wideform dataframe produced by extract_interassay_records(). Identifies samples tested more 
    than once, extracts their min-max interassay drift.     Also extracts the interassy drift for the closest 
    standards measured in the same experiments. Then compares the drift differences between sample and standard.
    Blank space is returned for repeat sample tests that do not have repeated calibrator values.

    Parameters:
    df (pd.df): The input df containing the data to describe.
    Minimum count of values required in a column to calculate standard deviation, standard error, and coefficient of variation.

    Returns:
    pd.df: The original df concatenated with the descriptive statistics.

    Description of additional metrics:
    - 'delta max-min': Difference between the maximum and minimum experimental values in each column.
    - 'sterror': Standard error of the values in each column (calculated as std / sqrt(count)).
    - 'cv': Coefficient of variation (calculated as std / mean).
    - 'relevant std_max-min': Difference between the standard concentrations closest to the minimum and maximum values.
    - 'std_id_min/max': Concatenation of the IDs of the standards closest to the min and max values.
    - 'rel_std_delta': std_id_max - std_id_min
    - 'rel_div_delta': Ratio of 'delta max-min' to 'relevant std_max-min'.
    - 'rel_delta_delta': Difference between 'delta max-min' and 'relevant std_max-min'.
    - 'sample_coordinates': the sample (min, max) used for delta calculations
    - 'std_coordinates': the nearest std concentration (min,max) to the sample (min,max) from the same experiment used for delta calculations
    - 'rel_std_name': the standards by name used for std_coordinates
    
    """
    # Compute basic statistics for each column
    description = df.describe().T   
    df.reset_index(drop=True, inplace=True)
    df.rename(columns={df.columns[0]: 'category'}, inplace=True)

    # Add additional statistics that are not part of `describe()`
    description['delta max-min'] = description['max'] - description['min']
    description['sterror'] = description['std'] / np.sqrt(description['count'])
    description['cv'] = description['std'] / description['mean']

    rel_std_min_list = []
    rel_std_max_list = []
    rel_std_delta_list = []
    rel_div_delta_list = []
    rel_delta_delta_list = []
    rel_sample_coordinates_list = []
    rel_std_coordinates_list = []
    rel_std_name_list = []
    
    # create df from available standards information
    std_df = df.loc[:, df.columns.str.contains('std|category', case=False)].fillna(np.nan)
    
    for column in df.columns[1:]:

        # we'll just treat all columns like samples
        min_value = df[column].min()
        max_value = df[column].max()
        sample_coordinates = (min_value, max_value)

        # we get the row number for min_value and max_value
        min_index = df[column].idxmin()
        row_number_min = df.index.get_loc(min_index)
        max_index = df[column].idxmax()
        row_number_max = df.index.get_loc(max_index)
          
        # filter the std_df to a dataframe with 2 rows
        filtered_df = std_df.iloc[[row_number_min,row_number_max]].reset_index()

        # We keep the index row as a dummy column of values so samples without adequate rel standard can match on it.
        filtered_df.rename(columns={'index': 'dummy_col'}, inplace=True)
        filtered_df['dummy_col'] = 3

        # Create a new row with tuples of the first and second values of each column
        new_row = {col: (filtered_df[col].iloc[0], filtered_df[col].iloc[1]) for col in filtered_df.columns}
        # Append the new row to the dataframe
        filtered_df = pd.concat([filtered_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Function to calculate Euclidean distance
        def euclidean_distance(coord1, coord2):
            return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
        
        # Initialize variables to store the minimum distance and corresponding information
        min_distance = float('inf')
        closest_tuple = None
        closest_column = None
        strings_tuple = None
        
        # Inspect the third row (excluding the first column)
        df_excluding_category = filtered_df.drop(columns=['category'])
        third_row = df_excluding_category.iloc[2]
        found_valid_tuple = False
        
        for col in df_excluding_category:
            current_tuple = third_row[col]
            
            if np.isnan(current_tuple).any():
                continue  # Skip if the current tuple contains np.nan
            
            distance = euclidean_distance(current_tuple, sample_coordinates)
            found_valid_tuple = True
            
            if distance < min_distance:
                min_distance = distance
                closest_tuple = current_tuple
                closest_column = col
                strings_tuple = filtered_df['category'][2]
        
        # Check if no valid tuple was found
        if not found_valid_tuple:
            min_distance = np.nan
            closest_tuple = (np.nan, np.nan)
            closest_column = 'none'
            strings_tuple = ('none', 'none')
        
        # Append to lists
        rel_std_min_list.append(closest_tuple[0])
        rel_std_max_list.append(closest_tuple[1])
        rel_std_delta_list.append(closest_tuple[1]-closest_tuple[0])
           
        # Check if either closest_tuple or sample_coordinates contain NaN values
        if not (np.isnan(closest_tuple).any() or np.isnan(sample_coordinates).any()):

            delta_closest_std = closest_tuple[1] - closest_tuple[0]
            delta_sample = sample_coordinates[1] - sample_coordinates[0]

            # Check if delta_closest_std or delta_sample is zero
            if delta_closest_std == 0 or delta_sample == 0:
                rel_div_delta_list.append(0)
            else:
                rel_div_delta_list.append(delta_closest_std / delta_sample)

            rel_delta_delta_list.append(delta_closest_std - delta_sample)   
        
        rel_sample_coordinates_list.append(sample_coordinates)
        rel_std_coordinates_list.append(closest_tuple)
        rel_std_name_list.append(closest_column + ' ' + str(strings_tuple))


    # Add the calculated values to the description df
    description['rel_std_min'] = rel_std_min_list
    description['rel_std_max'] = rel_std_max_list
    description['rel_std_delta']    = rel_std_delta_list
    description['rel_div_delta'] = rel_div_delta_list
    description['rel_delta_delta'] = rel_delta_delta_list
    description['sample_coordinates'] = rel_sample_coordinates_list
    description['std_coordinates'] = rel_std_coordinates_list
    description['rel_std_name'] = rel_std_name_list
    
    description = description.T
    description.reset_index(inplace=True)
    description.rename(columns={description.columns[0]: 'category'}, inplace=True)

    # Concatenate the original df with the description df
    final_df = pd.concat([df, description], axis=0)
    final_df.reset_index(drop=True, inplace=True)
    
    # Clean up final_df
    # Scan for 'dummy' in 'rel_std_name' category
    columns_with_dummy = final_df[final_df['category'] == 'rel_std_name'].apply(lambda x: x.str.contains('dummy')).any()

    # Get the column names that contain 'dummy'
    columns_to_modify = columns_with_dummy[columns_with_dummy].index

    # Categories to delete values from
    dummy_categories_to_clear = ['rel_std_min', 'rel_std_max', 'rel_std_delta','rel_div_delta', 
                                 'rel_delta_delta', 'std_coordinates','rel_std_name']

    # Delete the values in the specified columns and categories
    for category in dummy_categories_to_clear:
        final_df.loc[final_df['category'] == category, columns_to_modify] = np.nan
        
    # Categories to clear values from
    count_categories_to_clear = ['mean', 'std', 'min', '25%', '50%', '75%', 'max', 
                                 'delta max-min', 'sterror', 'cv', 'rel_std_min', 
                                 'rel_std_max', 'rel_std_delta', 'rel_div_delta', 'rel_delta_delta', 
                                 'sample_coordinates', 'std_coordinates', 'rel_std_name']

    # Identify the columns where the count is either 1 or 0
    count_condition = final_df[final_df['category'] == 'count'].iloc[0, 1:]
    columns_to_modify = count_condition[(count_condition == 1) | (count_condition == 0)].index

    # Delete the values in the specified columns and categories
    for category in count_categories_to_clear:
        final_df.loc[final_df['category'] == category, columns_to_modify] = np.nan

    # Replace 'std' with 'stdev' in the 'category' column
    final_df['category'] = final_df['category'].replace('std', 'stdev')
        
    if row_labels == 'index':
        final_df.set_index('category', inplace=True)

    return final_df

## Usage 
## final_df = describe_interassay_drift(record_df)

def summarise_interassay_drift(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Summarises interassay drift metrics from a dataframe produced by the proxipal function describe_interassay_drift().

    Args:
        df (pd.DataFrame): The DataFrame containing assay data with specific rows and columns for calculations.

    Returns:
        tuple[dict, pd.DataFrame]: A dictionary containing various calculated drift metrics and a wide-form DataFrame summarising the data.
    """

    def effective_conc_change(delta: float) -> float:
        """
        Calculates the effective concentration change based on delta value.

        Args:
            delta (float): The delta value for concentration change calculation.

        Returns:
            float: The effective concentration change as a percentage.
        """
        return ((2 ** delta) - 1) * 100

    # Filter out columns with 'std' in the header
    non_std_cols = [col for col in df.columns if 'std' not in col.lower()]

    # Calculate relevant std delta metrics
    rel_std_min = df.loc['rel_std_min'].min()
    rel_std_max = df.loc['rel_std_max'].max()
    rel_std_delta_mean = df.loc['rel_std_delta', non_std_cols].mean()
    rel_std_delta_stdev = df.loc['rel_std_delta', non_std_cols].std()
    
    # Calculate sample delta metrics
    sample_delta_min = df.loc['min'].min()
    sample_delta_max = df.loc['max'].max()
    sample_delta_mean = df.loc['delta max-min', non_std_cols].mean()
    sample_delta_stdev = df.loc['stdev', non_std_cols].std()

    def calc_ci(mean: float, std: float, multiplier: float) -> tuple[float, float]:
        """
        Calculates the confidence interval for a given mean, standard deviation, and multiplier.

        Args:
            mean (float): The mean value.
            std (float): The standard deviation.
            multiplier (float): The multiplier for the confidence interval calculation.

        Returns:
            tuple[float, float]: The lower and upper bounds of the confidence interval.
        """
        return (mean - multiplier * std, mean + multiplier * std)

    drift_dict = {
        "Relevant std delta (min)": rel_std_min,
        "Relevant std delta (max)": rel_std_max,
        "Relevant std delta (mean)": rel_std_delta_mean,
        "Relevant std delta (stdev)": rel_std_delta_stdev,
        "Relevant std delta range (68% confidence interval)": calc_ci(rel_std_delta_mean, rel_std_delta_stdev, 1),
        "Relevant std delta range (95% confidence interval)": calc_ci(rel_std_delta_mean, rel_std_delta_stdev, 2),
        "Relevant std delta range (99.7% confidence interval)": calc_ci(rel_std_delta_mean, rel_std_delta_stdev, 3),
        "sample delta (min)": sample_delta_min,
        "sample delta (max)": sample_delta_max,
        "sample delta (mean)": sample_delta_mean,
        "sample delta (stdev)": sample_delta_stdev,
        "sample delta range (68% confidence interval)": calc_ci(sample_delta_mean, sample_delta_stdev, 1),
        "sample delta range (95% confidence interval)": calc_ci(sample_delta_mean, sample_delta_stdev, 2),
        "sample delta range (99.7% confidence interval)": calc_ci(sample_delta_mean, sample_delta_stdev, 3)}

    def retrieve_row_values(df: pd.DataFrame, row_label: str) -> pd.Series:
        """
        Retrieves values for a specified row label, filtering columns without 'std' and where 'count' > 1.

        Args:
            df (pd.DataFrame): The DataFrame containing assay data.
            row_label (str): The label of the row to retrieve values from.

        Returns:
            pd.Series: The filtered values for the specified row label.
        """
        filtered_columns = [col for col in df.columns if 'std' not in col]
        filtered_df = df[filtered_columns].copy()
        filtered_df.loc['count'] = pd.to_numeric(filtered_df.loc['count'], errors='coerce')
        count_values = filtered_df.loc['count']
        filtered_columns = count_values[count_values > 1].index
        result = filtered_df.loc[row_label, filtered_columns]
        return result

    rel_std_result = retrieve_row_values(df=df, row_label='rel_std_delta')
    rel_std_delta_min = rel_std_result.min()
    rel_std_delta_max = rel_std_result.max()
    rel_std_delta_mean = rel_std_result.mean()
    rel_std_delta_stdev = rel_std_result.std()
    rel_std_delta_68range = (rel_std_delta_mean - rel_std_delta_stdev, rel_std_delta_mean, rel_std_delta_mean + rel_std_delta_stdev)
    rel_std_delta_95range = (rel_std_delta_mean - 2 * rel_std_delta_stdev, rel_std_delta_mean, rel_std_delta_mean + 2 * rel_std_delta_stdev)
    rel_std_delta_99range = (rel_std_delta_mean - 3 * rel_std_delta_stdev, rel_std_delta_mean, rel_std_delta_mean + 3 * rel_std_delta_stdev)

    sample_result = retrieve_row_values(df=df, row_label='delta max-min')
    sample_delta_min = sample_result.min()
    sample_delta_max = sample_result.max()
    sample_delta_mean = sample_result.mean()
    sample_delta_stdev = sample_result.std()
    sample_delta_68range = (sample_delta_mean - sample_delta_stdev, sample_delta_mean, sample_delta_mean + sample_delta_stdev)
    sample_delta_95range = (sample_delta_mean - 2 * sample_delta_stdev, sample_delta_mean, sample_delta_mean + 2 * sample_delta_stdev)
    sample_delta_99range = (sample_delta_mean - 3 * sample_delta_stdev, sample_delta_mean, sample_delta_mean + 3 * sample_delta_stdev)

    data_dict = {
        'category': ['min', 'max', 'mean', 'stdev', '68range', '95range', '99range'],
        'rel_std': [
            round(rel_std_delta_min, 2),
            round(rel_std_delta_max, 2),
            round(rel_std_delta_mean, 2),
            round(rel_std_delta_stdev, 2),
            tuple(round(x, 2) for x in rel_std_delta_68range),
            tuple(round(x, 2) for x in rel_std_delta_95range),
            tuple(round(x, 2) for x in rel_std_delta_99range)
        ],
        'sample': [
            round(sample_delta_min, 2),
            round(sample_delta_max, 2),
            round(sample_delta_mean, 2),
            round(sample_delta_stdev, 2),
            tuple(round(x, 2) for x in sample_delta_68range),
            tuple(round(x, 2) for x in sample_delta_95range),
            tuple(round(x, 2) for x in sample_delta_99range)
        ]
    }

    drift_table = pd.DataFrame(data_dict).set_index('category')

    return drift_dict, drift_table

## Usage
## drift_dict, drift_table = summarise_interassay_drift(final_df)

def plot_interassay_drift(drift_table):
    """
    Accepts a drift_table as produced by ProxiPal's summarise_interassay_drift( ) function.
    Generates a Plotly plot to visualize the overlay of normal distributions
    for the relative standard deviation (rel_std) and sample based on the 
    provided drift table.

    Parameters:
    drift_table (pd.DataFrame): DataFrame containing 'mean' and 'stdev' for 'rel_std' and 'sample'.
    """
    # Extract mean and standard deviation for both distributions
    mean_rel_std = drift_table.loc['mean', 'rel_std']
    std_rel_std = drift_table.loc['stdev', 'rel_std']
    mean_sample = drift_table.loc['mean', 'sample']
    std_sample = drift_table.loc['stdev', 'sample']

    # Generate x-values
    x = np.linspace(min(mean_rel_std, mean_sample) - 4 * max(std_rel_std, std_sample),
                    max(mean_rel_std, mean_sample) + 4 * max(std_rel_std, std_sample),
                    1000)

    # Calculate y-values for both distributions
    y_rel_std = norm.pdf(x, mean_rel_std, std_rel_std)
    y_sample = norm.pdf(x, mean_sample, std_sample)

    # Create the Plotly figure
    fig = go.Figure()

    # Add trace for rel_std
    fig.add_trace(go.Scatter(x=x, y=y_rel_std, mode='lines', name='rel_std',
                             line=dict(color='blue')))

    # Add trace for sample
    fig.add_trace(go.Scatter(x=x, y=y_sample, mode='lines', name='sample',
                             line=dict(color='red')))

    # Update layout
    fig.update_layout(title='Overlay of Normal Distributions',
                      xaxis_title='Value',
                      yaxis_title='Probability Density')

    # Show the plot
    fig.show()

## Usage
## plot_interassay_drift(drift_table)


def analyse_imprecision(dataframe: pd.DataFrame, columns_to_analyse: list, 
                        grubbs_test: str = 'intra/inter', grubbs_remove: bool = True) -> pd.DataFrame:
    """
    analyse_imprecision() → performs intra-assay and inter-assay CV analysis. It explicitly distinguishes within-assay (repeated measures in a single run) and 
    between-assay (averages across runs) variability. Uses grouping by sample_id, tube_id, and filepath_csv to compute separate CVs at both hierarchical levels.
    
    
    analyse_imprecision() takes a user-curated dataframe with ProxiPal data structure, i.e.
    'sample_id', 'filepath_csv', 'expt_folder_csv', 'analysis_folder_csv'. Such dataframes are typically produced as 
    megatables or metatables in ProxiPal. Curation of the input dataframe should be considered prior to using
    analyses_imprecision(), i.e. calibrator type, dilution conditions, should be selected for.
    NB: Currently tube_id is ignored, and should be written in at a later point. This is stated in the print statement when running the function.
    
    The function will count the frequency of data points for each sample_id and calculate mean, stdev, and CV. 
    These metrics will be calculated first for intra-assay and then for inter-assay values. These will be 
    influenced by the arg, grubbs_test, as follows:
        grubbs_test = intra-assay; outliers will be identified only for intra-assay groups
        grubbs_test = inter-assay; outliers will be identified only for inter-assay groups
        grubbs_test = intra/inter; outliers will be identified for intra-assay groups and then inter-assay groups
    
    The arg, grubbs_remove, lets users remove outliers and recalculate metrics. Therefore:
        grubbs_test = intra-assay, grubbs_remove = True; can directly alter pool of values used for intra assay metrics, 
        and indirectly affect the inter assay metrics.
        grubbs_test = inter-assay, grubbs_remove = True; will not affect any intra assay metrics, but can alter the pool
        of values used to calculate the inter assay metrics
        grubbs_test = intra/inter, grubbs_remove = True; will first directly affect the pool of values used for intra 
        assay metrics, and subsequently directly affect the pool of values used to calculate the inter assay metrics
        
    Wherever grubbs_remove = False, outliers will be reported but will not be removed during calculations for intra and inter-assay metrics.
     
    Summary
    analyses imprecision in the data by calculating intra- and inter-assay metrics
    and optionally performing Grubbs' test to identify and remove outliers.

    Args:
        dataframe (pd.DataFrame): Input dataframe containing the data to be analysed.
        columns_to_analyse (list): List of column names to analyse for imprecision.
        grubbs_test (str, optional): Specifies the type of Grubbs' test to perform.
            Options are 'intra-assay', 'inter-assay', or 'intra/inter'. Default is 'intra-assay'.
        grubbs_remove (bool, optional): If True, detected outliers are removed from calculations.
            Default is False.

    Returns:
        pd.DataFrame: A dataframe containing the imprecision analysis results for each sample.
    """

    # Ensure only numeric, non-NaN data in columns to analyse
    dataframe = dataframe.copy()
    for col in columns_to_analyse:
        dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    dataframe = dataframe.dropna(subset=columns_to_analyse)

    # Group the data by 'sample_id'
    # grouped = dataframe.groupby('sample_id') #deprecated in favour of grouping by both sample_id and tube_id
    dataframe['tube_id'] = dataframe['tube_id'].fillna('missing')
    grouped = dataframe.groupby(['sample_id', 'tube_id'])


    # Initialize a list to store the analysis results
    results = []

    for name, group in grouped:
        analysis_dict = {}

        for col in columns_to_analyse:
            # Group by 'filepath_csv' and concatenate unique values in the column as strings
            grouped_by_filepath = group.groupby('filepath_csv')[col].apply(lambda x: ','.join(x.astype(str).unique()))

            # Store the concatenated strings in the analysis dictionary
            analysis_dict[f'values ({col})'] = '; '.join(f'{values}' for values in grouped_by_filepath)
            
            intra_outliers = []
            intra_mean_values = []
            intra_stdev_values = []
            intra_frequency_values = []
            intra_cv_values = []

            for filepath, values in group.groupby('filepath_csv')[col]:
                outliers_removed = values  # Default to using the original values
                
                if grubbs_test in ['intra-assay', 'intra/inter']:
                    # Perform a two-sided Grubbs' test for outliers
                    detected_outliers = set(values) - set(grubbs.test(values, alpha=0.05))
                    intra_outliers.extend(detected_outliers)

                    # Optionally remove outliers for calculation
                    if grubbs_remove:
                        outliers_removed = values[~values.isin(detected_outliers)]

                # Calculate intra-assay metrics (optionally without outliers)
                mean_value = outliers_removed.mean()
                stdev_value = outliers_removed.std()
                freq_value = len(outliers_removed)
                cv_value = (stdev_value / mean_value) * 100 if mean_value != 0 else 0

                intra_mean_values.append(mean_value)
                intra_stdev_values.append(stdev_value)
                intra_frequency_values.append(freq_value)
                intra_cv_values.append(cv_value)

            # Store the intra-assay metrics in the analysis dictionary
            analysis_dict[f'intra-assay_mean ({col})'] = '; '.join(f'{round(v, 2)}' for v in intra_mean_values)
            analysis_dict[f'intra-assay_stdev ({col})'] = '; '.join(f'{round(v, 2)}' for v in intra_stdev_values)
            analysis_dict[f'intra-assay_cv ({col})'] = '; '.join(f'{round(v, 2)}' for v in intra_cv_values)
            analysis_dict[f'intra-assay_freq ({col})'] = '; '.join(f'{v}' for v in intra_frequency_values)
            
            if intra_outliers:
                analysis_dict[f'intra-assay_outliers ({col})'] = list(intra_outliers)

            # Calculate inter-assay metrics using the intra-assay means
            if grubbs_test in ['inter-assay', 'intra/inter']:
                # Perform a Grubbs' test for outliers on the intra-assay means
                inter_assay_outliers_removed = grubbs.test(intra_mean_values, alpha=0.05)
                removed_outliers = set(intra_mean_values) - set(inter_assay_outliers_removed)

                # Prepare values for metric calculation, optionally removing outliers
                inter_assay_values_for_calculation = (
                    [value for value in intra_mean_values if value not in removed_outliers]
                    if grubbs_remove else
                    intra_mean_values
                )

                # Calculate inter-assay metrics
                inter_assay_mean = pd.Series(inter_assay_values_for_calculation).mean()
                inter_assay_stdev = pd.Series(inter_assay_values_for_calculation).std()
                inter_assay_cv = (inter_assay_stdev / inter_assay_mean) * 100 if inter_assay_mean != 0 else 0
                inter_assay_freq = len(inter_assay_values_for_calculation)
                
                # Store the inter-assay metrics in the analysis dictionary
                analysis_dict[f'inter-assay_mean ({col})'] = round(inter_assay_mean, 2)
                analysis_dict[f'inter-assay_stdev ({col})'] = round(inter_assay_stdev, 2)
                analysis_dict[f'inter-assay_cv ({col})'] = round(inter_assay_cv, 2)
                analysis_dict[f'inter-assay_freq ({col})'] = inter_assay_freq
                
                if removed_outliers:
                    analysis_dict[f'inter-assay_outliers ({col})'] = list(removed_outliers)
            else:
                # Calculate inter-assay metrics without removing outliers
                inter_assay_mean = pd.Series(intra_mean_values).mean()
                inter_assay_stdev = pd.Series(intra_mean_values).std()
                inter_assay_cv = (inter_assay_stdev / inter_assay_mean) * 100 if inter_assay_mean != 0 else 0
                inter_assay_freq = len(intra_mean_values)
                
                # Store the inter-assay metrics in the analysis dictionary
                analysis_dict[f'inter-assay_mean ({col})'] = round(inter_assay_mean, 2)
                analysis_dict[f'inter-assay_stdev ({col})'] = round(inter_assay_stdev, 2)
                analysis_dict[f'inter-assay_cv ({col})'] = round(inter_assay_cv, 2)
                analysis_dict[f'inter-assay_freq ({col})'] = inter_assay_freq

            # Calculate standard error and confidence intervals
            inter_assay_se = inter_assay_stdev / np.sqrt(inter_assay_freq)
            z_value = 1.96  # for 95% confidence interval
            ci_lower = inter_assay_mean - z_value * inter_assay_se
            ci_upper = inter_assay_mean + z_value * inter_assay_se

            # Store SE and CI in the analysis dictionary
            analysis_dict[f'inter-assay_SE ({col})'] = round(inter_assay_se, 2)
            analysis_dict[f'CI_lower ({col})'] = round(ci_lower, 2)
            analysis_dict[f'CI_upper ({col})'] = round(ci_upper, 2)

        # Store unique 'expt_folder_csv' and 'analysis_folder_csv' values
        expt_analysis = '; '.join(group['expt_folder_csv'].astype(str).unique())
        analysis = '; '.join(group['analysis_folder_csv'].astype(str).unique())
        filepath_analysis = '; '.join(group['filepath_csv'].astype(str).unique())

        # Combine all results into a dictionary
        result = {
            # 'sample_id': name, #deprecated in favour of grouping by both sample_id and tube_id
            'sample_id': name[0],
            'tube_id': name[1],
            'expt_analysis': expt_analysis,
            'analysis': analysis,
            'filepath_analysis': filepath_analysis,  # Reflect unique filepath_csv values
            **analysis_dict  # Add the calculated statistics
        }
        
        results.append(result)

    # Convert the list of dictionaries into a DataFrame
    final_df = pd.DataFrame(results)
    
    print("analyse_imprecision() testing finished")
    
    return final_df

# Example usage
# data_rdml_indiv_interAssay = analyse_imprecision(
#     data_rdml_indiv_means,
#     columns_to_analyse=['rdml_mean_ng/L'],
#     grubbs_test='intra/inter',
#     grubbs_remove=False
# )


def passing_bablok(method1, method2):
    """Perform Passing-Bablok analysis.
    Implementation, https://rowannicholls.github.io/python/statistics/hypothesis_testing/passing_bablok.html"""
    #
    # Calculate the gradients of the lines between each pair of points
    #
    n_points = len(method1)
    # sv is a list of the gradients between of each pair of points
    sv = []
    # k is the number of gradients less than -1
    k = 0
    for i in range(n_points - 1):
        for j in range(i + 1, n_points):
            dy = method2[j] - method2[i]
            dx = method1[j] - method1[i]
            # Ignore gradients that are vertical (ie the x values of the points
            # are the same)
            if dx != 0:
                gradient = dy / dx
            elif dy < 0:
                gradient = -1.e+23
            elif dy > 0:
                gradient = 1.e+23
            else:
                gradient = None
            if gradient is not None:
                sv.append(gradient)
                k += (gradient < -1)
    # Sort the gradients into ascending order
    sv.sort()

    #
    # Find the estimated gradient and confidence limits
    #
    m0 = (len(sv) - 1) / 2
    if m0 == int(m0):
        # If odd
        gradient_est = sv[k + int(m0)]
    else:
        # If even
        gradient_est = 0.5 * (sv[k + int(m0 - 0.5)] + sv[k + int(m0 + 0.5)])
    # Calculate the index of the upper and lower confidence bounds
    w = 1.96
    ci = w * math.sqrt((n_points * (n_points - 1) * (2 * n_points + 5)) / 18)
    n_gradients = len(sv)
    m1 = int(round((n_gradients - ci) / 2))
    m2 = n_gradients - m1 - 1
    # Calculate the lower and upper bounds of the gradient
    (gradient_lb, gradient_ub) = (sv[k + m1], sv[k + m2])

    def calc_intercept(method1, method2, gradient):
        """Calculate intercept given points and a gradient."""
        temp = []
        for i in range(len(method1)):
            temp.append(method2[i] - gradient * method1[i])
        return np.median(temp)

    # Calculate the intercept as the median of all the intercepts of all the
    # lines connecting each pair of points
    int_est = calc_intercept(method1, method2, gradient_est)
    int_ub = calc_intercept(method1, method2, gradient_lb)
    int_lb = calc_intercept(method1, method2, gradient_ub)

    return (gradient_est, gradient_lb, gradient_ub), (int_est, int_lb, int_ub)

## Usage
## beta, alpha = passing_bablok(df['Simoa'], df['PLA'])


def plot_passing_bablok(data, col1_name, col2_name, beta, alpha, Rs=True, Title = 'Passing-Bablok Regression', figsize = (5,5)):
    """
    Plot Passing-Bablok Regression for two methods in the provided DataFrame.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    col1_name (str): Name of the column for Method 1.
    col2_name (str): Name of the column for Method 2.
    beta (tuple): Tuple containing estimated slope and confidence interval bounds.
    alpha (tuple): Tuple containing estimated intercept and confidence interval bounds.
    Rs (bool): Whether to compute and display Spearman's rank correlation coefficient.
    """
    plt.figure(figsize=figsize)  # Adjust size as needed

    ax = plt.axes()
    ax.set_title(Title)
    ax.set_xlabel(col1_name)
    ax.set_ylabel(col2_name)
    
    # Scatter plot
    ax.scatter(data[col1_name], data[col2_name], c='k', s=20, alpha=0.6, marker='o')
    
    # Get and set axis limits
    left, right = plt.xlim()
    bottom, top = plt.ylim()
    ax.set_xlim(0, right)
    ax.set_ylim(0, top)
    
    # Reference line
    ax.plot([left, right], [left, right], c='grey', ls='--')
    
    # Passing-Bablok regression line
    x = np.array([left, right])
    y = beta[0] * x + alpha[0]
    ax.plot(x, y)
    
    # Confidence intervals
    y_lb = beta[1] * x + alpha[1]
    y_ub = beta[2] * x + alpha[2]
    ax.plot(x, y_ub, c='tab:blue', alpha=0.2)
    ax.plot(x, y_lb, c='tab:blue', alpha=0.2)
    ax.fill_between(x, y_ub, y_lb, alpha=0.2)
    
    # Create legend manually
    legend_elements = [
        Line2D([0], [0], color='grey', ls='--', label='Reference line'),
        Line2D([0], [0], color='tab:blue', label=f'{beta[0]:4.2f}x + {alpha[0]:4.2f}'),
        Line2D([0], [0], color='tab:blue', alpha=0.4, label=f'Upper CI: {beta[2]:4.2f}x + {alpha[2]:4.2f}'),
        Line2D([0], [0], color='tab:blue', alpha=0.4, label=f'Lower CI: {beta[1]:4.2f}x + {alpha[1]:4.2f}')
    ]
    
    # Compute and add Spearman's rank correlation coefficient if Rs is True
    if Rs:
        spearman_coefficient, _ = spearmanr(data[col1_name], data[col2_name])
        legend_elements.append(Line2D([0], [0], color='none', label=f"Spearman's R = {spearman_coefficient:.2f}"))
    
    # Add legend
    ax.legend(handles=legend_elements, frameon=True, facecolor='white', framealpha=1)
    
    # Grid
    plt.grid(True)

    fig = plt.gcf()  #to capture the figure
    
    # Show plot
    plt.show()
    
    return fig       #so you can save the figure


## Usage
## plot_passing_bablok(df, 'Simoa','PLA',  beta, alpha, Title = 'Simoa vs PLA: Passing Bablok Regression', Rs = True, figsize = (6,5))


def create_expt_dict(dataframe: pd.DataFrame) -> dict:
    """
    Creates a dictionary where each key corresponds to a specific experiment group, 
    and the value is the subset of the dataframe related to that group. Most useful for separating
    out a ProxiPal megatable or mastertable.

    The function first checks the uniqueness of the 'expt_folder_csv' and 'analysis_folder_csv'
    columns. If there are more unique analysis folders than experiment folders, it prints 
    a warning message. Then, it groups the dataframe by the 'filepath_csv' column, and for each 
    group, it extracts a key from the 'filepath_csv' by taking the second-to-last component 
    (replacing hyphens with underscores). The function returns a dictionary with these keys 
    and their corresponding dataframe groups.

    Args:
        dataframe (pd.DataFrame): A pandas DataFrame containing the experimental data.

    Returns:
        dict: A dictionary with keys derived from 'filepath_csv' and values being the grouped 
        subsets of the dataframe.
    """
    
    # Step 1: Check the uniqueness of expt_folder_csv and analysis_folder_csv
    unique_expt = dataframe["expt_folder_csv"].nunique()
    unique_analysis = dataframe["analysis_folder_csv"].nunique()

    # Step 2: Compare and print the message if the condition is met
    if unique_analysis > unique_expt:
        print("More analysis folders detected than experiment folders: from input table delete extra analysis groups")

    # Step 3: Group by filepath_csv and create the dictionary
    result_dict = {}
    for key, group in dataframe.groupby("filepath_csv"):
        dict_key = key.split("_")[-2].replace("-", "_")
        result_dict[dict_key] = group

    # Return the result_dict
    return result_dict

## Usage
## expt_dict = create_expt_dict(data)

# def summarise_imprecision(df: pd.DataFrame, metric: str, sample_id_substring: str = 'NFL', min_inter_assay_freq: int = 3) -> pd.DataFrame:
#     """
#     Summarizes imprecision data by filtering and calculating mean values for key metrics.

#     The function filters the input dataframe based on a substring in the 'sample_id' column 
#     and a minimum value in the 'inter-assay_freq' column for a specified metric. It then 
#     calculates and prints the mean values of inter-assay metrics, including the mean, standard 
#     deviation, and coefficient of variation (CV).

#     Args:
#         df (pd.DataFrame): The input dataframe containing imprecision data.
#         metric (str): The metric to analyze (e.g., 'rdml_mean_ng/L').
#         sample_id_substring (str): The substring to filter by in the 'sample_id' column. 
#                                    Default is 'NFL'.
#         min_inter_assay_freq (int): The minimum frequency threshold for 'inter-assay_freq'. 
#                                     Default is 3.

#     Returns:
#         pd.DataFrame: A filtered dataframe based on the specified criteria.
#     """
    
#     # Filter the dataframe for rows where 'sample_id' contains the specified substring and 'inter-assay_freq' >= min_inter_assay_freq
#     filtered_df = df[(df['sample_id'].str.contains(sample_id_substring)) & (df[f'inter-assay_freq ({metric})'].astype(int) >= min_inter_assay_freq)]

#     # Report the number of rows in the filtered dataframe
#     filtered_len = len(filtered_df)
#     print(f"Number of rows in filtered dataframe: {filtered_len}")

#     # Calculate the mean for the specified metric columns
#     mean_inter_assay_mean = filtered_df[f'inter-assay_mean ({metric})'].astype(float).mean()
#     mean_inter_assay_stdev = filtered_df[f'inter-assay_stdev ({metric})'].astype(float).mean()
#     mean_inter_assay_cv = filtered_df[f'inter-assay_cv ({metric})'].astype(float).mean()

#     # Print the calculated means
#     print(f"Mean of 'inter-assay_mean ({metric})': {mean_inter_assay_mean:.2f}")
#     print(f"Mean of 'inter-assay_stdev ({metric})': {mean_inter_assay_stdev:.2f}")
#     print(f"Mean of 'inter-assay_cv ({metric})': {mean_inter_assay_cv:.2f}")
    
#     # Return the filtered dataframe
#     return filtered_df

def summarise_imprecision(df: pd.DataFrame, metric: str, sample_id_substring: str = None, min_inter_assay_freq: int = 3) -> pd.DataFrame:
    """
    Summarizes imprecision data by filtering and calculating mean values for key metrics.
    
    Args:
        df (pd.DataFrame): The input dataframe containing imprecision data.
        metric (str): The metric to analyze (e.g., 'rdml_mean_ng/L').
        sample_id_substring (str, optional): The substring to filter by in the 'sample_id' column. 
                                           If None, no filtering is applied. Default is None.
        min_inter_assay_freq (int): The minimum frequency threshold for 'inter-assay_freq'. 
                                   Default is 3.
    Returns:
        pd.DataFrame: A filtered dataframe based on the specified criteria.
    """
    
    # Create the base condition for frequency filtering
    condition = df[f'inter-assay_freq ({metric})'].astype(int) >= min_inter_assay_freq
    
    # Add substring filtering only if a substring is provided
    if sample_id_substring is not None:
        condition = condition & df['sample_id'].str.contains(sample_id_substring)
    
    # Apply the filter
    filtered_df = df[condition]
    
    # Rest of the function remains the same
    filtered_len = len(filtered_df)
    print(f"Number of rows in filtered dataframe: {filtered_len}")
    
    mean_inter_assay_mean = filtered_df[f'inter-assay_mean ({metric})'].astype(float).mean()
    mean_inter_assay_stdev = filtered_df[f'inter-assay_stdev ({metric})'].astype(float).mean()
    mean_inter_assay_cv = filtered_df[f'inter-assay_cv ({metric})'].astype(float).mean()
    
    print(f"Mean of 'inter-assay_mean ({metric})': {mean_inter_assay_mean:.2f}")
    print(f"Mean of 'inter-assay_stdev ({metric})': {mean_inter_assay_stdev:.2f}")
    print(f"Mean of 'inter-assay_cv ({metric})': {mean_inter_assay_cv:.2f}")
    
    return filtered_df


## Usage
## filtered_df = summarise_imprecision(data_rdml_indiv_imprecision, metric='rdml_mean_ng/L', sample_id_substring = 'NFL', min_inter_assay_freq = 3)


def build_qc_df(df: pd.DataFrame, sample_id: list = None, stdev_range: int = 2) -> pd.DataFrame:
    """
    build_qc_df() → automatically constructs a QC dataframe from either
    analyse_imprecision() or analyze_sample_stats() output.
    It's intended for use with process_qc_data()

    For analyse_imprecision(): uses inter-assay_mean and inter-assay_stdev columns.
    For analyze_sample_stats(): uses mean and std columns.

    Args:
        df (pd.DataFrame): Input dataframe (output of analyse_imprecision() or analyze_sample_stats()).
        sample_id (list, optional): Subset of sample_ids to include. Default = all.
        stdev_range (int): Multiplier for the stdev range (e.g., 2 for mean ± 2*stdev).

    Returns:
        pd.DataFrame: QC dataframe with columns:
                      ['sample_id', 'mean', 'stdev', f'stdev_range={stdev_range}', 'fail raw_ng/L', 'fail mean_ng/L']
    """

    # Identify function source by column pattern
    imprecision_mean_cols = [c for c in df.columns if c.startswith('inter-assay_mean')]
    imprecision_stdev_cols = [c for c in df.columns if c.startswith('inter-assay_stdev')]

    if len(imprecision_mean_cols) > 1 or len(imprecision_stdev_cols) > 1:
        raise ValueError("Multiple inter-assay metrics detected. Re-run analyse_imprecision() with a single columns_to_analyse entry.")
    
    if len(imprecision_mean_cols) == 1 and len(imprecision_stdev_cols) == 1:
        mean_col = imprecision_mean_cols[0]
        stdev_col = imprecision_stdev_cols[0]
    elif {'mean', 'std'}.issubset(df.columns):
        mean_col, stdev_col = 'mean', 'std'
    else:
        raise ValueError("Input DataFrame does not match expected structure from analyse_imprecision() or analyze_sample_stats().")

    # Subset if sample_id list is provided
    if sample_id:
        df = df[df['sample_id'].isin(sample_id)]

    qc_records = []
    for _, row in df.iterrows():
        mean_val = float(row[mean_col])
        stdev_val = float(row[stdev_col])

        lower = int(round(mean_val - stdev_range * stdev_val))
        upper = int(round(mean_val + stdev_range * stdev_val))
        qc_records.append({
            'sample_id': row['sample_id'],
            'mean': mean_val,
            'stdev': stdev_val,
            f'stdev_range={stdev_range}': f'[{lower}, {upper}]',
            'fail raw_ng/L': None,
            'fail mean_ng/L': None
        })

    qc_df = pd.DataFrame(qc_records)
    return qc_df

# Example usage
# analyse_ss = analyze_sample_stats(data = mastertable, model_type = 'SLR; log10(x); exc_std0; ct; raw_ng/L', decimal_places = 'integer', format_timing= 'before')
# qc_df = build_qc_df(df, sample_id = ['NFL-QC-H'], stdev_range = 2)


def process_qc_data(qc_df, py_metatable):
    """
    Process QC data and identify failures based on standard deviation ranges.
    Uses mean values from qc_df to evaluate failures for both raw and mean ng/L values.
    
    Parameters:
    qc_df (pd.DataFrame): DataFrame containing QC information
    py_metatable (pd.DataFrame): DataFrame containing measurement data with dilution column
    
    Returns:
    pd.DataFrame: Updated QC DataFrame with fail values
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    qc_df = qc_df.copy()
    
    def parse_range(range_str):
        """Extract min and max values from range string '[min, max]'"""
        return [float(x) for x in range_str.strip('[]').split(',')]
    
    def calculate_deviations(value, mean, std):
        """Calculate number of standard deviations from mean"""
        return abs(value - mean) / std
    
    def check_failures(sample_data, mean_val, std_val, value_column):
        """Check for failures in a specific value column"""
        fails = []
        for pos_idx, pos_row in sample_data.iterrows():
            value = pos_row[value_column]
            position = pos_row['position']
            
            # Check if value is outside the acceptable range defined by mean ± 2*std
            if (value < (mean_val - 2*std_val)) or (value > (mean_val + 2*std_val)):
                dev_calc = calculate_deviations(value, mean_val, std_val)
                fail_str = f"[{position};{value:.0f};{dev_calc:.2f}]"
                fails.append(fail_str)
        return fails
    
    # Process each QC sample
    for idx, row in qc_df.iterrows():
        sample_id = row['sample_id']
        mean_val = row['mean']
        std_val = row['stdev']
        
        # Filter py_metatable for current sample_id and dilution
        sample_data = py_metatable[
            (py_metatable['sample_id'] == sample_id) & 
            (py_metatable['dilution'] == 10)
        ]
        
        # Check raw values
        raw_fails = check_failures(sample_data, mean_val, std_val, 
                                 'SLR; log10(x); exc_std0; ct; raw_ng/L')
        if raw_fails:
            qc_df.at[idx, 'fail raw_ng/L'] = '; '.join(raw_fails)
            
        # Check mean values
        mean_fails = check_failures(sample_data, mean_val, std_val,
                                  'SLR; log10(x); exc_std0; ct; mean_ng/L')
        if mean_fails:
            qc_df.at[idx, 'fail mean_ng/L'] = ', '.join(mean_fails)
    
    return qc_df


def calc_PL_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                    popt: np.ndarray, concentration: np.ndarray,
                    n_params: int = 4) -> Dict:
    """
    Calculate comprehensive fit metrics focusing on non-redundant measures
    with robust error handling. Includes R²-adjusted for interpretability
    alongside AICc for complexity adjustment.
    """
    n = len(y_true)
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Degrees of freedom and complexity measures
    dof = n - n_params
    complexity_ratio = n_params / n
    
    # Error-handled calculations
    try:
        syx = np.sqrt(ss_res / dof) if dof > 0 else np.inf
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # R²-adjusted with error handling
        r_squared_adj = 1 - ((1 - r_squared) * (n - 1) / (n - n_params - 1)) if n > n_params + 1 else -np.inf
        rmse = np.sqrt(np.mean(residuals**2))

        # Information criterion (only AICc as primary measure)
        if n > n_params + 1:
            aic = n * np.log(ss_res/n) + 2 * n_params
            aicc = aic + (2 * n_params * (n_params + 1)) / (n - n_params - 1)

        else:
            aicc = np.inf
            
        # Residual analysis
        residual_skewness = pearson3.fit(residuals)[2]
        residual_pattern = np.abs(np.corrcoef(concentration, residuals)[0,1])
        
        return {'r_squared': r_squared,
                'r_squared_adj': r_squared_adj,
                'rmse': rmse,
                'aicc': aicc,
                'residual_skewness': residual_skewness,
                'residual_pattern': residual_pattern,
                'dof': dof,
                'complexity_ratio': complexity_ratio,
                'syx': syx,
                'ss_res': ss_res,
                'ss_tot': ss_tot,
                'residuals': residuals}
    
    except Exception as e:
        print(f"Warning: Error in metric calculation - {str(e)}")

        return {'r_squared': 0,
                'r_squared_adj': -np.inf,
                'rmse': np.inf,
                'aicc': np.inf,
                'residual_skewness': np.nan,
                'residual_pattern': np.nan,
                'dof': dof,
                'complexity_ratio': complexity_ratio,
                'syx': np.inf,
                'ss_res': ss_res,
                'ss_tot': ss_tot,
                'residuals': np.zeros_like(y_true)}


def calc_PL_score(metrics: Dict, popt: np.ndarray, 
                            response_mean: float, conc_min: float, 
                            conc_max: float) -> float:
    """
    Calculate composite score using non-redundant metrics and appropriate penalties
    """

    # 1. Parameter reasonability (unchanged as it's independent of other metrics)
    param_scores = {
        'A_reasonable': 0.8 <= (popt[0] / response_mean) <= 1.2,
        'B_reasonable': -10 <= popt[1] <= 10,
        'C_reasonable': conc_min/2 <= popt[2] <= conc_max*2,
        'D_reasonable': 0.8 <= (popt[3] / response_mean) <= 1.2
    }
    param_reasonable = np.mean(list(param_scores.values()))
    
    # 2. Model complexity penalty (using AICc)
    # Normalize AICc to [0,1] range where 1 is better. Use soft maximum to avoid division by zero
    aicc_norm = 1 / (1 + np.exp((metrics['aicc'] - 100) / 50))
    
    # 3. Residual quality score (combining pattern and distribution)
    residual_quality = ((1 - metrics['residual_pattern']) *                 # Low correlation is better
                        (1 - min(abs(metrics['residual_skewness']), 1)))    # Low skewness is better
    
    # 4. Relative precision score using Sy.x
    # Normalize relative to response mean
    precision_score =   1 / (1 + (metrics['syx'] / response_mean))
    
    # 5. Complexity penalty based on degrees of freedom
    complexity_penalty = 1 / (1 + np.exp(5 * (metrics['complexity_ratio'] - 0.5)))
    
    # Combine scores with weights prioritizing:
    # - AICc (accounts for both fit quality and complexity)
    # - Residual quality (indicates model appropriateness)
    # - Parameter reasonability (domain-specific constraints)
    weights = { 'aicc': 0.35,                                               # Highest weight as it combines fit quality and complexity
                'residual_quality': 0.25,                                   # Important for detecting model inadequacy
                'param_reasonable': 0.20,                                   # Domain-specific constraints
                'precision': 0.10,                                          # Additional measure of fit quality
                'complexity': 0.10}                                         # Extra penalty for high parameter count
    
    final_score = (weights['aicc'] * aicc_norm +
                    weights['residual_quality'] * residual_quality +
                    weights['param_reasonable'] * param_reasonable +
                    weights['precision'] * precision_score +
                    weights['complexity'] * complexity_penalty)
    
    return final_score


def get_PL_metrics(best_method: str, transform_x: str, best_score: float,
                        param_names: list, popt: np.ndarray, 
                        response: np.ndarray, y_fit: np.ndarray, 
                        concentration: np.ndarray) -> pd.DataFrame:
    """
    Return a long-form pandas DataFrame containing all fit metrics.
    """
    metrics = calc_PL_metrics(response, y_fit, popt, concentration)
    
    metrics_data = pd.DataFrame({
        'Metric': [
            'Fitting Method',
            'Transform',
            'Composite Score',
            *param_names,
            'R²',
            'Adjusted R²',
            'RMSE',
            'AICc',
            'Standard Error (Sy.x)',
            'Degrees of Freedom',
            'Residual Pattern',
            'Residual Skewness',
            'Complexity Ratio',
            'Sum of Squares (Residual)',
            'Sum of Squares (Total)'
        ],
        'Value': [
            best_method,
            transform_x,
            best_score,
            *popt,
            metrics['r_squared'],
            metrics['r_squared_adj'],
            metrics['rmse'],
            metrics['aicc'],
            metrics['syx'],
            metrics['dof'],
            metrics['residual_pattern'],
            metrics['residual_skewness'],
            metrics['complexity_ratio'],
            metrics['ss_res'],
            metrics['ss_tot']
        ]})
    
    return metrics_data


# def calc_PL_conc(y: float = None, dilution: float = 1.0,
#                  A: float = None, B: float = None, C: float = None, 
#                  D: float = None, G: float = None,
#                  transform_x: str = 'linear(x)',
#                  model: str = '4PL') -> float:
#     """
#     Corrected concentration calculation function for qPCR data.
    
#     Parameters
#     ----------
#     y : float
#         Cq value
#     A, B, C, D : float
#         4PL parameters where:
#         - A is the upper asymptote (Cq at zero/low concentration)
#         - D is the lower asymptote (Cq at infinite/high concentration)
#         - C is the EC50 (concentration at midpoint)
#         - B is the slope parameter
#     """
#     if model == '4PL':
#         try:
#             if transform_x == 'linear(x)':
#                 # Rearranged 4PL formula solving for x (concentration)
#                 x = C * ((A - D) / (y - D) - 1) ** (1 / B)
                
#             elif transform_x == 'log10(x)':  
#                 # Rearranged 4PL formula solving for log10(x)
#                 log_x = C - (1/B) * np.log((D - y)/(y - A))
#                 x = 10**log_x
            
#             # Check if result is complex before rounding
#             if np.iscomplex(x):
#                 return np.nan

#             # Return dilution-adjusted concentration if calculation succeeded
#             return round(x * dilution, 4)
            
#         except (ValueError, ZeroDivisionError):
#             # Return nan if calculation failed (e.g. due to invalid math operations)
#             return np.nan
    
#     elif model == '5PL':
#         if G is None:
#             raise ValueError("G parameter required for 5PL model")
            
#         try:
#             if transform_x == 'linear(x)':
#                 x = C * ((((A - D) / (y - D)) ** (1/G) - 1) ** (1/B))
            
#             elif transform_x == 'log10(x)':
#                 # Manually derived from 5PL log10 fit, y= D + (A - D) / (1 + np.exp(B * (x - C)))**(G)
#                 # double check it
#                 log_x = (np.log(((((A-D)/(y-D))**(1/G))-1))/(B)) + C
#                 x = 10**log_x

#             # Check if result is complex before rounding
#             if np.iscomplex(x):
#                 return np.nan

#             return round(x * dilution, 4)
            
#         except (ValueError, ZeroDivisionError):
#             return np.nan

def calc_PL_conc(y: float = None, dilution: float = 1.0,
                 A: float = None, B: float = None, C: float = None, 
                 D: float = None, G: float = None,
                 transform_x: str = 'linear(x)',
                 model: str = '4PL') -> float:
    """
    Corrected concentration calculation function for qPCR data.
    
    Parameters
    ----------
    y : float
        Cq value
    A, B, C, D : float
        4PL parameters where:
        - A is the upper asymptote (Cq at zero/low concentration)
        - D is the lower asymptote (Cq at infinite/high concentration)
        - C is the EC50 (concentration at midpoint)
        - B is the slope parameter
    """
    try:
        if model == '4PL':
            if transform_x == 'linear(x)':
                # Rearranged 4PL formula solving for x (concentration)
                x = C * ((A - D) / (y - D) - 1) ** (1 / B)
                
            elif transform_x == 'log10(x)':  
                # Rearranged 4PL formula solving for log10(x)
                log_x = C - (1/B) * np.log((D - y)/(y - A))
                x = 10**log_x
            
            # Check if result is complex before rounding
            if np.iscomplex(x):
                return np.nan
            
            # Return dilution-adjusted concentration if calculation succeeded
            return round(x * dilution, 4)
    
        elif model == '5PL':
            if G is None:
                return np.nan
                
            if transform_x == 'linear(x)':
                x = C * ((((A - D) / (y - D)) ** (1/G) - 1) ** (1/B))
            
            elif transform_x == 'log10(x)':
                # Manually derived from 5PL log10 fit, y= D + (A - D) / (1 + np.exp(B * (x - C)))**(G)
                # double check it
                log_x = (np.log(((((A-D)/(y-D))**(1/G))-1))/(B)) + C
                x = 10**log_x

            # Check if result is complex before rounding
            if np.iscomplex(x):
                return np.nan
            
            return round(x * dilution, 4)
            
    except Exception:
        return np.nan


def robust_4pl_fit(concentration: np.ndarray, 
                   response: np.ndarray,
                   transform_x: Literal["linear(x)", "log10(x)"] = "log10(x)",
                   plot: bool = True) -> Tuple[np.ndarray, np.ndarray, float, dict, List[Dict]]:
    """
    Robust 4PL fitting function with data-based bounds strategy and comprehensive fit selection.
    """

    # Input validation for concentration transforms
    if transform_x == "log10(x)" and np.any(concentration > 20):
        raise ValueError("""Calculation Aborted: ProxiPal suspects that user has provided x input as linear concentration but has selected transform_x = log10(x). 
                        Evidence: At least one concentration value exceeds 20 and transform_x = log10(x)""")
    
    if transform_x == "linear(x)" and not np.any(concentration > 20):
        raise ValueError("""Calculation Aborted: ProxiPal suspects that user has provided x input as log concentration but has selected transform_x = linear(x). 
                        Evidence: No concentration value exceeds 20 and transform_x = linear(x)""")

    if transform_x == "log10(x)" and np.any(concentration == 0):
        raise ValueError("""Calculation Aborted: ProxiPal suspects user has provided a concentration value = 0 and selected transform_x = log10(x). 
                        Concentration = 0 values cannot be evaluated; curves requiring these fits must be either 
                        1. performed with linear concentration values or 
                        2. introduce user-substituted log10(near-zero) values instead of 0.""")

    def xlog10_4PL(x, A, B, C, D):
        """4PL logistic equation for log-transformed concentrations"""
        return A + (D - A) / (1 + np.exp(-B * (x - C)))
    
    def xlinear_4PL(x, A, B, C, D):
        """4PL function for non-transformed (linear) concentration data"""
        return ((A - D) / (1.0 + (x / C)**B)) + D
    
    # Select appropriate function
    fit_func = xlinear_4PL if transform_x == "linear(x)" else xlog10_4PL
    
    # Data characteristics
    response_range = np.ptp(response)
    response_min = np.min(response)
    response_max = np.max(response)
    response_mean = np.mean(response)
    conc_min = np.min(concentration)
    conc_max = np.max(concentration)
    conc_mid = np.median(concentration)
    
    # Initialize tracking
    best_fit = None
    best_score = -np.inf
    best_method = None
    best_params = None
    all_fits = []
    
    # Define fitting strategies (same as before)
    strategies = []
    if transform_x == "linear(x)":
        strategies.extend([
            {
                "name": "Data-based bounds",
                "p0": [response_max, -1.0, conc_mid, response_min],
                "bounds": ([response_min - response_range, -10.0, conc_min, response_min - response_range],
                          [response_max + response_range, 10.0, conc_max, response_max + response_range]),
                "method": "trf"
            },
            {
                "name": "Constrained EC50",
                "p0": [response_max, -1.0, conc_mid, response_min],
                "bounds": ([-np.inf, -np.inf, conc_min/2, -np.inf],
                          [np.inf, np.inf, conc_max*2, np.inf]),
                "method": "trf"
            },
            # Add LM method version
            {
                "name": "Unconstrained LM",
                "p0": [response_max, -1.0, conc_mid, response_min],
                "bounds": None,
                "method": "lm"
            }
        ])
    else:
        strategies.extend([
            {
                "name": "Log transform strategy",
                "p0": [response_min - 0.2*response_range, -1.0, conc_mid, response_max + 0.2*response_range],
                "bounds": ([-np.inf, -10.0, -np.inf, -np.inf],
                          [np.inf, -0.01, np.inf, np.inf]),
                "method": "trf"
            },
            {
                "name": "Log transform LM",
                "p0": [response_min - 0.2*response_range, -1.0, conc_mid, response_max + 0.2*response_range],
                "bounds": None,
                "method": "lm"
            }
        ])
    
    # Try each strategy
    multipliers = [0.1, 0.5, 1.0, 2.0, 10.0]
    # param_names = ['A (upper asymptote)', 'B (Hill slope)', 'C (EC50)', 'D (lower asymptote)'] if transform_x == "linear(x)" else \
    #              ['A (min asymptote)', 'B (steepness)', 'C (midpoint)', 'D (max asymptote)']
    param_names = ['A (upper asymptote)', 'B (slope)', 'C (EC50)', 'D (lower asymptote)']

    for strategy in strategies:
        for mult in multipliers:
            try:
                p0_scaled = [p * mult for p in strategy["p0"]]
                
                if strategy["bounds"] is None:
                    popt, pcov = curve_fit(fit_func, concentration, response,
                                         p0=p0_scaled, method=strategy["method"],
                                         maxfev=5000)
                else:
                    popt, pcov = curve_fit(fit_func, concentration, response,
                                         p0=p0_scaled, bounds=strategy["bounds"],
                                         method=strategy["method"], maxfev=5000)
                
                y_fit = fit_func(concentration, *popt)
                
                # Calculate comprehensive metrics
                metrics = calc_PL_metrics(response, y_fit, popt, concentration)
                composite_score = calc_PL_score(metrics, popt, response_mean, conc_min, conc_max)
                
                # Store all successful fits
                if not np.any(np.isnan(y_fit)) and not np.any(np.isinf(y_fit)):
                    fit_attempt = {
                        'strategy_name': strategy["name"],
                        'multiplier': mult,
                        'parameters': dict(zip(param_names, popt)),
                        'predictions': y_fit,
                        'metrics': metrics,
                        'composite_score': composite_score,
                        'method': strategy["method"],
                        'success': True
                    }
                    all_fits.append(fit_attempt)
                    
                    if composite_score > best_score and not np.isnan(composite_score):
                        best_fit = (popt, y_fit)
                        best_score = composite_score
                        best_method = strategy["method"]
                        best_params = p0_scaled
                    
            except Exception as e:
                all_fits.append({
                    'strategy_name': strategy["name"],
                    'multiplier': mult,
                    'error': str(e),
                    'success': False
                })
                continue
    
    if best_fit is None:
        raise RuntimeError("Could not find a suitable fit with any method")
    
    popt, y_fit = best_fit
    
    metrics_df = get_PL_metrics(  # Create metrics_df even when not plotting
        best_method=best_method,
        transform_x=transform_x,
        best_score=best_score,
        param_names=param_names,
        popt=popt,
        response=response,
        y_fit=y_fit,
        concentration=concentration)
    
    # Check R-squared value
    r_squared = metrics_df.loc[metrics_df['Metric'] == 'R²', 'Value'].iloc[0]
    if r_squared < 0.8:
        print(f"4PL; {transform_x}; The R² is unexpectedly low for this fit. ProxiPal recommends checking your x, y inputs and visual plot(s)")

    # Create 4PL equation string based on transform
    if transform_x == "log10(x)":
        equation = f"y = {popt[0]:.4f} + ({popt[3]:.4f} - {popt[0]:.4f}) / (1 + exp(-{popt[1]:.4f} * (x - {popt[2]:.4f})))"
    elif transform_x == "linear(x)":
        equation = f"y = (({popt[0]:.4f} - {popt[3]:.4f}) / (1 + (x/{popt[2]:.4f})^{popt[1]:.4f})) + {popt[3]:.4f}"


    # Add plotting functionality
    if plot:
        plt.figure(figsize=(10, 6))
        
        if transform_x == "linear(x)":
            # For linear transform, plot as is
            plt.scatter(concentration, response, color='blue', alpha=0.6, label='Data points')
            x_smooth = np.linspace(min(concentration), max(concentration), 1000)
            y_smooth = fit_func(x_smooth, *popt)
            plt.plot(x_smooth, y_smooth, 'r-', label='Fitted curve')
            plt.xscale('linear')
            plt.xlabel('Concentration')
            
        else:
            # For log10 transform, convert display to actual concentrations
            actual_concentrations = 10**concentration
            plt.scatter(actual_concentrations, response, color='blue', alpha=0.6, label='Data points')
            
            # Generate smooth curve
            x_smooth_log = np.linspace(-1.3, max(concentration) + 0.1, 1000)
            y_smooth = fit_func(x_smooth_log, *popt)
            
            # Convert to actual concentrations for display
            x_smooth_actual = 10**x_smooth_log
            plt.plot(x_smooth_actual, y_smooth, 'r-', label='Fitted curve')
            
            plt.xscale('log')
            plt.xlabel('Concentration')
            plt.xlim(10**-1.3, 10**(max(concentration) + 0.1))
        
        plt.ylabel('Response')
        plt.grid(True, which='both', linestyle='--', alpha=0.3)

        r_squared = metrics_df.loc[metrics_df['Metric'] == 'R²', 'Value'].iloc[0]
        equation_text = f'R² = {r_squared:.4f}\n{equation}'
        plt.text(0.5, 0.95, equation_text, transform=plt.gca().transAxes,
                horizontalalignment='center', verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Set x-axis to use decimal format
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.gca().xaxis.get_major_formatter().set_scientific(False)
        plt.gca().tick_params(axis='x', which='major')
        
        plt.tight_layout()
        plt.show()
       
    return metrics_df, equation

## USAGE
# Call function on log(x) data
# concentration_log = np.array([3.30103, 2.60206, 1.90309, 1.20412, 0.50515, -0.19382, -0.89279])
# response = np.array([14.5196, 17.2045, 19.6159, 21.98645, 24.24765, 26.72125, 28.6469])
# metrics_df, equation = robust_4pl_fit(concentration_log, response, transform_x="log10(x)", plot = True)


def robust_5pl_fit(concentration: np.ndarray, 
                   response: np.ndarray,
                   transform_x: Literal["linear(x)", "log10(x)"] = "log10(x)",
                   plot: bool = False) -> Tuple[pd.DataFrame, str]:
    """
    Robust 5PL fitting function with data-based bounds strategy and comprehensive fit selection.
    NOTE: Unlike previous versions, this function expects data to already be transformed if 
    transform_x="log10(x)". Transformation should be handled by the calling function.
    
    Args:
        concentration: Array of concentration values (should be log10-transformed if transform_x="log10(x)")
        response: Array of response values
        transform_x: Type of x-axis transformation ("linear(x)" or "log10(x)")
        plot: If True, creates a figure with the fit and residuals
        
    Returns:
        Tuple containing metrics DataFrame and equation string
    """

    # Input validation for concentration transforms
    if transform_x == "log10(x)" and np.any(concentration > 20):
        raise ValueError("""Calculation Aborted: ProxiPal suspects that user has provided x input as linear concentration but has selected transform_x = log10(x). 
                        Evidence: At least one concentration value exceeds 20 and transform_x = log10(x)""")
    
    if transform_x == "linear(x)" and not np.any(concentration > 20):
        raise ValueError("""Calculation Aborted: ProxiPal suspects that user has provided x input as log concentration but has selected transform_x = linear(x). 
                        Evidence: No concentration value exceeds 20 and transform_x = linear(x)""")

    if transform_x == "log10(x)" and np.any(concentration == 0):
        raise ValueError("""Calculation Aborted: ProxiPal suspects user has provided a concentration value = 0 and selected transform_x = log10(x). 
                        Concentration = 0 values cannot be evaluated; curves requiring these fits must be either 
                        1. performed with linear concentration values or 
                        2. introduce user-substituted log10(near-zero) values instead of 0.""")

    def xlog10_5PL(x, A, B, C, D, G):
        """5PL logistic equation for log-transformed concentrations"""
        return D + (A - D) / (1 + np.exp(B * (x - C)))**(G)
    
    def xlinear_5PL(x, A, B, C, D, G):
        """5PL function for non-transformed (linear) concentration data"""
        return D + (A - D) / (1 + (x/C)**B)**(G)

    # Select appropriate function based on transform_x
    fit_func = xlinear_5PL if transform_x == "linear(x)" else xlog10_5PL
    
    # Data characteristics (using input concentration directly)
    response_range = np.ptp(response)
    response_min = np.min(response)
    response_max = np.max(response)
    response_mean = np.mean(response)
    conc_min = np.min(concentration)
    conc_max = np.max(concentration)
    conc_mid = np.median(concentration)
    
    # Initialize tracking
    best_fit = None
    best_score = -np.inf
    best_method = None
    best_params = None
    all_fits = []
    
    # Define fitting strategies
    strategies = []
    if transform_x == "linear(x)":
        strategies.extend([
            {
                "name": "Data-based bounds",
                "p0": [response_max, 1.0, conc_mid, response_min, 1.0],
                "bounds": ([response_min - response_range, -10.0, conc_min, response_min - response_range, 0.1],
                          [response_max + response_range, 10.0, conc_max, response_max + response_range, 10.0]),
                "method": "trf"
            },
            {
                "name": "Unconstrained LM",
                "p0": [response_max, 1.0, conc_mid, response_min, 1.0],
                "bounds": None,
                "method": "lm"
            },
            # Add LM method version
            {
                "name": "Unconstrained LM",
                "p0": [response_max, -1.0, conc_mid, response_min],
                "bounds": None,
                "method": "lm"
            }
        ])
    else:
        strategies.extend([
            {
                "name": "Log transform strategy",
                "p0": [response_max, 2.0, conc_mid, response_min, 1.0],
                "bounds": ([response_min - response_range, 0.1, conc_min - 1, response_min - response_range, 0.1],
                          [response_max + response_range, 10.0, conc_max + 1, response_max + response_range, 10.0]),
                "method": "trf"
            },
            {
                "name": "Log transform LM",
                "p0": [response_max, 2.0, conc_mid, response_min, 1.0],
                "bounds": None,
                "method": "lm"
            }
        ])
    
    # Try each strategy with different multipliers
    multipliers = [0.1, 0.5, 1.0, 2.0, 10.0]
    param_names = ['A (upper asymptote)', 'B (slope)', 'C (EC50)', 
                  'D (lower asymptote)', 'G (asymmetry)']
    
    for strategy in strategies:
        for mult in multipliers:
            try:
                p0_scaled = [p * mult for p in strategy["p0"]]
                
                if strategy["bounds"] is None:
                    popt, pcov = curve_fit(fit_func, concentration, response,
                                         p0=p0_scaled, method=strategy["method"],
                                         maxfev=5000)
                else:
                    popt, pcov = curve_fit(fit_func, concentration, response,
                                         p0=p0_scaled, bounds=strategy["bounds"],
                                         method=strategy["method"], maxfev=5000)
                
                y_fit = fit_func(concentration, *popt)

                # Calculate comprehensive metrics
                metrics = calc_PL_metrics(response, y_fit, popt, concentration)
                composite_score = calc_PL_score(metrics, popt, response_mean, conc_min, conc_max)
                
                if not np.any(np.isnan(y_fit)) and not np.any(np.isinf(y_fit)):
                    fit_attempt = {
                        'strategy_name': strategy["name"],
                        'multiplier': mult,
                        'parameters': dict(zip(param_names, popt)),
                        'predictions': y_fit,
                        'metrics': metrics,
                        'composite_score': composite_score,
                        'method': strategy["method"],
                        'success': True
                    }
                    all_fits.append(fit_attempt)
                    
                    if composite_score > best_score and not np.isnan(composite_score):
                        best_fit = (popt, y_fit)
                        best_score = composite_score
                        best_method = strategy["method"]
                        best_params = p0_scaled
                        
            except Exception as e:
                all_fits.append({
                    'strategy_name': strategy["name"],
                    'multiplier': mult,
                    'error': str(e),
                    'success': False
                })
                continue
    
    if best_fit is None:
        raise RuntimeError("Could not find a suitable fit with any method")
    
    popt, y_fit = best_fit
    
    metrics_df = get_PL_metrics(  # Create metrics_df even when not plotting
        best_method=best_method,
        transform_x=transform_x,
        best_score=best_score,
        param_names=param_names,
        popt=popt,
        response=response,
        y_fit=y_fit,
        concentration=concentration)
    
    # Check R-squared value
    r_squared = metrics_df.loc[metrics_df['Metric'] == 'R²', 'Value'].iloc[0]
    if r_squared < 0.8:
        print(f"5PL; {transform_x}; The R² is unexpectedly low for this fit. ProxiPal recommends checking your x, y inputs and visual plot(s)")

    # Create 5PL equation string based on transform
    if transform_x == "log10(x)":
        equation = f"y = {popt[0]:.4f} + ({popt[3]:.4f} - {popt[0]:.4f}) / (1 + exp(-{popt[1]:.4f} * (x - {popt[2]:.4f})))^{popt[4]:.4f}"
    elif transform_x == "linear(x)":
        equation = f"y = {popt[3]:.4f} + ({popt[0]:.4f} - {popt[3]:.4f}) / (1 + (x/{popt[2]:.4f})^{popt[1]:.4f})^{popt[4]:.4f}"

    # Add plotting functionality
    if plot:
        plt.figure(figsize=(10, 6))
        
        if transform_x == "linear(x)":
            # For linear transform, plot as is
            plt.scatter(concentration, response, color='blue', alpha=0.6, label='Data points')
            x_smooth = np.linspace(min(concentration), max(concentration), 1000)
            y_smooth = fit_func(x_smooth, *popt)
            plt.plot(x_smooth, y_smooth, 'r-', label='Fitted curve')
            plt.xscale('linear')
            plt.xlabel('Concentration')
            
        else:
            # For log10 transform, convert display to actual concentrations
            actual_concentrations = 10**concentration
            plt.scatter(actual_concentrations, response, color='blue', alpha=0.6, label='Data points')
            
            # Generate smooth curve
            x_smooth_log = np.linspace(-1.3, max(concentration) + 0.1, 1000)
            y_smooth = fit_func(x_smooth_log, *popt)
            
            # Convert to actual concentrations for display
            x_smooth_actual = 10**x_smooth_log
            plt.plot(x_smooth_actual, y_smooth, 'r-', label='Fitted curve')
            
            plt.xscale('log')
            plt.xlabel('Concentration')
            plt.xlim(10**-1.3, 10**(max(concentration) + 0.1))
        
        plt.ylabel('Response')
        plt.grid(True, which='both', linestyle='--', alpha=0.3)

        r_squared = metrics_df.loc[metrics_df['Metric'] == 'R²', 'Value'].iloc[0]
        equation_text = f'R² = {r_squared:.4f}\n{equation}'
        plt.text(0.5, 0.95, equation_text, transform=plt.gca().transAxes,
                horizontalalignment='center', verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Set x-axis to use decimal format
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.gca().xaxis.get_major_formatter().set_scientific(False)
        plt.gca().tick_params(axis='x', which='major')
        
        plt.tight_layout()
        plt.show()

    return metrics_df, equation

## Example Usage: Test both transforms
# concentration_log = np.array([3.30103, 2.60206, 1.90309, 1.20412, 0.50515, -0.19382, -0.89279])
# response = np.array([14.5196, 17.2045, 19.6159, 21.98645, 24.24765, 26.72125, 28.6469])
# metrics_df, equation = robust_5pl_fit(concentration_log, response, transform_x="log10(x)", plot = True)


def calc_metatable_PL(py_metatable, threshold_type='rdml_Cq (mean eff)', 
                      transform_x='linear(x)', model = '4PL', std0_status='exc_std0', export=False):
    """
    Enhanced version of calc_metatable that uses 5PL fitting and stores all metrics.
    
    Parameters
    ----------
    py_metatable : pandas.DataFrame
        Input metatable containing the data
    threshold_type : str, optional
        Column name for threshold values. Default is 'rdml_Cq (mean eff)'
    transform_x : str, optional
        Type of x transformation. Either 'linear(x)' or 'log10(x)'. Default is 'linear(x)'
    std0_status : str, optional
        Either 'exc_std0' or 'inc_std0'. Default is 'exc_std0'.
        Note: For log10(x) transform, std0_status is always set to 'exc_std0'
    export : bool, optional
        Whether to export results to CSV. Default is False
        
    Returns
    -------
    pandas.DataFrame
        Updated metatable with complete 5PL regression metrics
    """

    # Validate model parameter
    valid_models = ['4PL', '5PL']
    if model not in valid_models:
        raise ValueError(f"Model must be one of {valid_models}, got {model}")

    # Force exc_std0 for log10(x) transform
    if transform_x == 'log10(x)' and std0_status == 'inc_std0':
        print("Attention user: Mismatched arg inputs. Where transform_x = 'log10(x)', std0_status must = exc_std0. This change has been forced.' ")
        std0_status = 'exc_std0'

    if 'sub_bg' in threshold_type:
        std0_status = 'exc_std0'

    # Avoid copy warning
    robust_metatable = py_metatable.copy()
    
    # Error strings created by excel should be removed
    robust_metatable = robust_metatable.replace('#DIV/0!', np.nan)
    robust_metatable = robust_metatable.replace('#N/A', np.nan)
    
    # Check for column py_known_conc. if not present, create it.
    robust_metatable = add_py_known_conc(robust_metatable.copy())
        
    # Assign column name for threshold mean   
    threshold_mean = threshold_type + '; mean'
    
    # Where "usr_ignore" = 1 the threshold_mean is not calculated
    robust_metatable.loc[robust_metatable["usr_ignore"] == 1, threshold_mean] = robust_metatable.loc[robust_metatable["usr_ignore"] == 1, threshold_type]
    
    # Filter the dataframe for "usr_ignore" != 1 and calculate means
    filtered_df = robust_metatable[robust_metatable["usr_ignore"] != 1].copy()
    filtered_df.loc[:, threshold_mean] = filtered_df.groupby("rep_id")[threshold_type].transform("mean")
        
    # Identify standards in use
    stds_used_list = filtered_df['usr_std'].unique().tolist()
    
    metrics_mapping = { 'Fitting Method': 'fitting_method',
                        'Transform': 'transform',
                        'Composite Score': 'composite_score',
                        'A (upper asymptote)': 'A',
                        'B (slope)': 'B',
                        'C (EC50)': 'C',
                        'D (lower asymptote)': 'D',
                        'R²': 'R-squared',
                        'Adjusted R²': 'R-squared-adj',
                        'RMSE': 'RMSE',
                        'AICc': 'AICc',
                        'Standard Error (Sy.x)': 'Standard Error of the Estimate',
                        'Degrees of Freedom': 'Degrees of Freedom',
                        'Residual Pattern': 'residual_pattern',
                        'Residual Skewness': 'residual_skewness',
                        'Complexity Ratio': 'complexity_ratio',
                        'Sum of Squares (Residual)': 'Sum of Squares, residual',
                        'Sum of Squares (Total)': 'Sum of Squares, total'}
        
    if model == '5PL':
        # Split the dictionary into two parts at the insertion point
        first_half = {k: metrics_mapping[k] for k in list(metrics_mapping.keys())[:7]}  # up to and including 'D'
        second_half = {k: metrics_mapping[k] for k in list(metrics_mapping.keys())[7:]}  # from 'R²' onwards
        metrics_mapping = first_half | {'G (asymmetry)': 'G'} | second_half
    
    # Process each standard
    for i in stds_used_list:
        std = 'std' + str(i)
        pattern = r"{}\[.*?\]_".format(std)
    
        # Create metatable subset
        std_df = filtered_df[filtered_df['sample_id'].str.contains(pattern, regex=True)]
        
        # Determine which columns to check for NA values
        required_cols = ['py_known_conc', threshold_mean]
        if transform_x == 'log10(x)' or std0_status == 'exc_std0':
            required_cols.append('py_known_conc_log10')
            
        # Drop NA values and calculate means for replicates
        valid_data = std_df.dropna(subset=required_cols)
        valid_data_means = valid_data.groupby('rep_id')[required_cols].mean()

        # Extract x and y values
        y = valid_data_means[threshold_mean].values
        x = valid_data_means['py_known_conc_log10'].values if transform_x == 'log10(x)' else valid_data_means['py_known_conc'].values

        try:
            if model == '4PL':                  # Use robust_4pl_fit for improved fitting
                metrics_df, equation = robust_4pl_fit(x, y, transform_x=transform_x, plot=False)

            elif model == '5PL':                  # Use robust_5pl_fit for improved fitting
                metrics_df, equation = robust_5pl_fit(x, y, transform_x=transform_x, plot=False)            
            
            # Store all metrics in metatable
            for idx, row in metrics_df.iterrows():
                metric = row['Metric']
                value = row['Value']
                
                if metric in metrics_mapping:
                    column_name = f"{model}; {transform_x}; {std0_status}; {threshold_type}; {metrics_mapping[metric]}"
                    robust_metatable.loc[robust_metatable['usr_std'] == i, column_name] = (
                        value if isinstance(value, str) else round(value, 4)
                    )
            
            # Store equation separately
            equation_col = f"{model}; {transform_x}; {std0_status}; {threshold_type}; equation"
            robust_metatable.loc[robust_metatable['usr_std'] == i, equation_col] = equation
                    
        except Exception as e:
            print(f"Fitting failed for standard {std}: {str(e)}")
            continue
    
    # Update metatable with calculations
    mask = robust_metatable["usr_ignore"] != 1

    # Calculate mean thresholds
    robust_metatable.loc[mask, threshold_mean] = robust_metatable.loc[mask].groupby("rep_id")[threshold_type].transform("mean")

    # Calculate concentrations and recoveries
    for type_suffix in ['raw', 'mean']:
        threshold_col = threshold_type if type_suffix == 'raw' else threshold_mean

        robust_metatable.loc[mask, f"{model}; {transform_x}; {std0_status}; {threshold_type}; {type_suffix}_ng/L"] = robust_metatable.loc[mask].apply(
            lambda row: calc_PL_conc(
                y=row[threshold_col], 
                dilution=row['dilution'],
                A=row[f"{model}; {transform_x}; {std0_status}; {threshold_type}; A"],
                B=row[f"{model}; {transform_x}; {std0_status}; {threshold_type}; B"],
                C=row[f"{model}; {transform_x}; {std0_status}; {threshold_type}; C"],
                D=row[f"{model}; {transform_x}; {std0_status}; {threshold_type}; D"],
                G=row[f"{model}; {transform_x}; {std0_status}; {threshold_type}; G"] if model == '5PL' else None,
                transform_x=transform_x, model=model,
            ), axis=1)
        
        # Calculate recoveries
        robust_metatable.loc[mask, f"{model}; {transform_x}; {std0_status}; {threshold_type}; {type_suffix}_recovery"] = robust_metatable.loc[mask].apply(
            lambda row: round(row[f"{model}; {transform_x}; {std0_status}; {threshold_type}; {type_suffix}_ng/L"] / row['py_known_conc'], 2) 
            if row['py_known_conc'] != 0 else np.nan, axis=1)
    
    # Export if requested
    if export:
        if len(py_metatable['filepath_txt'].unique().tolist()) == 1:
            proxipal_path = Path(py_metatable['filepath_txt'].unique().tolist()[0])
            exports_folder = proxipal_path.parent / 'exports'
            
            if not exports_folder.exists():
                print('No local /exports folder found. Creating one.')
                exports_folder.mkdir(parents=True)
                
            robust_metatable.to_csv(exports_folder / 'py_metatable.csv', index=False)
    
    return robust_metatable

## Example usage (same interface as before)
# SLR_4PL_metatable = calc_metatable_PL(py_metatable=SLR_metatable, model = '4PL', threshold_type='rdml_Cq (mean eff)', transform_x='log10(x)', std0_status='exc_std0', export=False)
# SLR_5PL_metatable = calc_metatable_PL(py_metatable=SLR_metatable, threshold_type='rdml_Cq (mean eff)', transform_x='linear(x)', model = '5PL', std0_status='exc_std0', export=False)


def extract_logistic_metrics(fitted_df, logistic_regression='4PL', threshold_type='rdml_Cq (mean eff)', 
                           transform_x='log10(x)', std0_status='exc_std0'):
    """
    Extract logistic regression metrics (4PL or 5PL) from a fitted dataframe into a simple DataFrame.
    
    Parameters
    ----------
    fitted_df : pandas.DataFrame
        DataFrame containing fit results
    logistic_regression : str, optional
        Type of logistic regression, either '4PL' or '5PL'. Default is '5PL'
    threshold_type : str, optional
        Column name for threshold values. Default is 'rdml_Cq (mean eff)'
    transform_x : str, optional
        Type of x transformation used. Default is 'log10(x)'
    std0_status : str, optional
        Either 'exc_std0' or 'inc_std0'. Default is 'exc_std0'.
        Note: For log10(x) transform, std0_status is always 'exc_std0'
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with metrics as rows and standards as columns
    """
    # Validate logistic regression type
    if logistic_regression not in ['4PL', '5PL']:
        raise ValueError("logistic_regression must be either '4PL' or '5PL'")
    
    # Force exc_std0 for log10(x) transform
    if transform_x == 'log10(x)':
        std0_status = 'exc_std0'
        
    # Get unique standards
    standards = fitted_df['usr_std'].dropna().unique()
    
    # Define metrics and their column names in fitted_df
    metrics_order = [
        'Fitting Method',
        'Transform',
        'std0_status',
        'Composite Score',
        'A (upper asymptote)',
        'B (slope)',
        'C (EC50)',
        'D (lower asymptote)',
        'G (asymmetry)',
        'R²',
        'Adjusted R²',
        'RMSE',
        'AICc',
        'Standard Error (Sy.x)',
        'Degrees of Freedom',
        'Residual Pattern',
        'Residual Skewness',
        'Complexity Ratio',
        'Sum of Squares (Residual)',
        'Sum of Squares (Total)'
    ]
    
    # Create column mapping
    column_mapping = {
        'Fitting Method': f"{logistic_regression}; {transform_x}; {std0_status}; {threshold_type}; fitting_method",
        'Transform': f"{logistic_regression}; {transform_x}; {std0_status}; {threshold_type}; transform",
        'std0_status': std0_status,
        'Composite Score': f"{logistic_regression}; {transform_x}; {std0_status}; {threshold_type}; composite_score",
        'A (upper asymptote)': f"{logistic_regression}; {transform_x}; {std0_status}; {threshold_type}; A",
        'B (slope)': f"{logistic_regression}; {transform_x}; {std0_status}; {threshold_type}; B",
        'C (EC50)': f"{logistic_regression}; {transform_x}; {std0_status}; {threshold_type}; C",
        'D (lower asymptote)': f"{logistic_regression}; {transform_x}; {std0_status}; {threshold_type}; D",
        'G (asymmetry)': f"{logistic_regression}; {transform_x}; {std0_status}; {threshold_type}; G",
        'R²': f"{logistic_regression}; {transform_x}; {std0_status}; {threshold_type}; R-squared",
        'Adjusted R²': f"{logistic_regression}; {transform_x}; {std0_status}; {threshold_type}; R-squared-adj",
        'RMSE': f"{logistic_regression}; {transform_x}; {std0_status}; {threshold_type}; RMSE",
        'AICc': f"{logistic_regression}; {transform_x}; {std0_status}; {threshold_type}; AICc",
        'Standard Error (Sy.x)': f"{logistic_regression}; {transform_x}; {std0_status}; {threshold_type}; Standard Error of the Estimate",
        'Degrees of Freedom': f"{logistic_regression}; {transform_x}; {std0_status}; {threshold_type}; Degrees of Freedom",
        'Residual Pattern': f"{logistic_regression}; {transform_x}; {std0_status}; {threshold_type}; residual_pattern",
        'Residual Skewness': f"{logistic_regression}; {transform_x}; {std0_status}; {threshold_type}; residual_skewness",
        'Complexity Ratio': f"{logistic_regression}; {transform_x}; {std0_status}; {threshold_type}; complexity_ratio",
        'Sum of Squares (Residual)': f"{logistic_regression}; {transform_x}; {std0_status}; {threshold_type}; Sum of Squares, residual",
        'Sum of Squares (Total)': f"{logistic_regression}; {transform_x}; {std0_status}; {threshold_type}; Sum of Squares, total"
    }
    
    # Initialize results dictionary for all standards
    results = []
    
    # Extract metrics for each metric
    for metric in metrics_order:
        row_values = {'Metric': metric}
        
        # Special handling for std0_status
        if metric == 'std0_status':
            # Add the same std0_status value for all standards
            for std in standards:
                std_name = f'std{std}'
                row_values[std_name] = std0_status
        else:
            col_name = column_mapping[metric]
            # Get value for each standard
            for std in standards:
                std_name = f'std{std}'
                std_row = fitted_df[fitted_df['usr_std'] == std].iloc[0]
                value = std_row[col_name] if col_name in std_row else None
                row_values[std_name] = value
            
        results.append(row_values)
    
    # Create DataFrame
    metrics_df = pd.DataFrame(results)
    
    return metrics_df

## Extract metrics for all standards
# metrics_df = extract_logistic_metrics(SLR_4PL_metatable, logistic_regression='4PL', threshold_type='rdml_Cq (mean eff)', transform_x='log10(x)', std0_status='exc_std0')


def plot_logistic_standards(metatable, logistic_regression='4PL', threshold_type='rdml_Cq (mean eff)', 
                 transform_x='log10(x)', std0_status='exc_std0', figsize=(10, 6), separate_plots=False):
    """
    [Previous docstring remains the same]
    """
    if logistic_regression not in ['4PL', '5PL']:
        raise ValueError("logistic_regression must be either '4PL' or '5PL'")

    standards = metatable[(metatable['usr_std'].notna()) & (metatable['usr_std'] > 0)]['usr_std'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(standards)))
    threshold_mean = f"{threshold_type}; mean"
    
    # Determine text positioning based on conditions
    is_n0 = '_N0' in threshold_type
    is_log10 = transform_x == 'log10(x)'
    
    # Set initial position and direction of movement
    if is_log10 and not is_n0:  # bottom left, stack up
        start_x, start_y = 0.02, 0.1
        y_direction = 1  # positive for stacking up
    elif not is_log10 and not is_n0:  # top right, stack down
        start_x, start_y = 0.6, 0.9
        y_direction = -1  # negative for stacking down
    elif is_log10 and is_n0:  # top left, stack down
        start_x, start_y = 0.02, 0.9
        y_direction = -1  # negative for stacking down
    else:  # linear(x) and N0 - bottom right, stack up
        start_x, start_y = 0.6, 0.1
        y_direction = 1  # positive for stacking up
    
    if separate_plots:
        plots_dict = {}
        for idx, std in enumerate(standards):
            fig, ax = plt.subplots(figsize=figsize)
            color = colors[idx]
            std_name = f'Standard {std}'
            
            plot_single_logistic_standard(ax, metatable, std, color, threshold_type, threshold_mean, 
                               transform_x, std0_status, std_name, logistic_regression, 
                               single_plot=True, text_pos=(start_x, start_y))
            
            format_pl_axes(ax, metatable, threshold_type, transform_x)
            plt.tight_layout()
            plots_dict[f'std{std}'] = {'fig': fig, 'ax': ax}
            
        return plots_dict
    
    else:
        fig, ax = plt.subplots(figsize=figsize)
        equations_text = []
        for idx, std in enumerate(standards):
            color = colors[idx]
            std_name = f'Standard {std}'
            eq_text = plot_single_logistic_standard(ax, metatable, std, color, threshold_type, 
                                         threshold_mean, transform_x, std0_status, std_name,
                                         logistic_regression)
            equations_text.append((eq_text, color))
        
        # Adjust vertical spacing based on regression type (5PL needs more space)
        spacing = 0.13 if logistic_regression == '5PL' else 0.11
        
        for idx, (text, color) in enumerate(equations_text):
            # Calculate position based on iteration
            y_pos = start_y + (spacing * idx * y_direction)
            ax.text(start_x, y_pos, text,
                   transform=ax.transAxes, fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, boxstyle='round,pad=0.5'))
        
        format_pl_axes(ax, metatable, threshold_type, transform_x)
        plt.tight_layout()
        
        return {'fig': fig, 'ax': ax}

def plot_single_logistic_standard(ax, metatable, std, color, threshold_type, threshold_mean, 
                       transform_x, std0_status, std_name, logistic_regression, 
                       single_plot=False, text_pos=None):
    """Helper function to plot a single standard curve"""
    mask = (metatable['usr_std'] == std) & (metatable['usr_ignore'] != 1)
    std_data = metatable[mask].copy()
    
    if len(std_data) == 0:
        print(f"No data found for standard {std}")
        return None
    
    grouped = std_data.groupby('rep_id').agg({
        'py_known_conc': 'mean',
        threshold_mean: 'mean' if threshold_mean in std_data.columns else threshold_type
    }).sort_values('py_known_conc')
    
    # Plot scatter points
    ax.scatter(grouped['py_known_conc'], 
              grouped[threshold_mean if threshold_mean in grouped.columns else threshold_type], 
              color=color, alpha=0.7, s=50)
    
    param_prefix = f"{logistic_regression}; {transform_x}; {std0_status}; {threshold_type}"
    A = std_data[f"{param_prefix}; A"].iloc[0]
    B = std_data[f"{param_prefix}; B"].iloc[0]
    C = std_data[f"{param_prefix}; C"].iloc[0]
    D = std_data[f"{param_prefix}; D"].iloc[0]
    R2 = std_data[f"{param_prefix}; R-squared"].iloc[0]
    G = std_data[f"{param_prefix}; G"].iloc[0] if logistic_regression == '5PL' else None
    
    x_min = max(0.005, grouped['py_known_conc'].min() * 0.1)
    x_max = grouped['py_known_conc'].max() * 2
    
    # Create smooth x values based on scale type
    if transform_x == 'log10(x)':
        x_smooth = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
    else:
        x_smooth = np.linspace(x_min, x_max, 1000)
    
    if logistic_regression == '5PL':
        if transform_x == 'log10(x)':
            y_smooth = D + (A - D) / (1 + np.exp(B * (np.log10(x_smooth) - C)))**(G)
        else:
            y_smooth = D + (A - D) / (1 + (x_smooth/C)**B)**(G)
    else:  # 4PL
        if transform_x == 'log10(x)':
            y_smooth = A + (D - A) / (1 + np.exp(-B * (np.log10(x_smooth) - C)))
        else:
            y_smooth = ((A - D) / (1.0 + (x_smooth / C)**B)) + D
    
    # Plot the curve using regular plot for linear x-axis
    if transform_x == 'log10(x)':
        ax.semilogx(x_smooth, y_smooth, '-', color=color)
    else:
        ax.plot(x_smooth, y_smooth, '-', color=color)
    
    # Generate equation text
    text = f"{std_name}:"
    if logistic_regression == '5PL':
        if transform_x == 'log10(x)':
            eq = f"y = {D:.2f} + ({A:.2f} - {D:.2f})/(1 + exp({B:.2f}(x - {C:.2f})))^{G:.2f}"
        else:
            eq = f"y = {D:.2f} + ({A:.2f} - {D:.2f})/(1 + (x/{C:.2f})^{B:.2f})^{G:.2f}"
    else:  # 4PL
        if transform_x == 'log10(x)':
            eq = f"y = {A:.2f} + ({D:.2f} - {A:.2f})/(1 + exp(-{B:.2f}(x - {C:.2f})))"
        else:
            eq = f"y = ({A:.2f} - {D:.2f})/(1 + (x/{C:.2f})^{B:.4f}) + {D:.2f}"
    
    text += f"\n{eq}\nR² = {R2:.4f}"
    
    if single_plot:
        x_pos, y_pos = text_pos if text_pos else (0.02, 0.1)
        ax.text(x_pos, y_pos, text,
               transform=ax.transAxes, fontsize=8,
               bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, boxstyle='round,pad=0.5'))
    
    return text

def format_pl_axes(ax, metatable, threshold_type, transform_x):
    """Helper function to format plot axes"""
    ax.set_xlabel('Concentration (ng/L)')
    ax.set_ylabel(threshold_type)
    ax.grid(True, which='both' if transform_x == 'log10(x)' else 'major', linestyle='--', alpha=0.3)
    
    ydata = metatable[metatable['usr_std'].notna()][threshold_type]
    ymin, ymax = ydata.min(), ydata.max()
    margin = (ymax - ymin) * 0.1
    ax.set_ylim(ymin - margin, ymax + margin)
    
    if transform_x == 'log10(x)':
        ax.set_xlim(left=0.005)
        formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, numticks=20))
        
        def format_func(x, p):
            if x < 1:
                return f"{x:.3f}"
            return f"{x:.0f}"
        
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_func))
    else:
        xdata = metatable[metatable['usr_std'].notna()]['py_known_conc']
        xmin, xmax = xdata.min(), xdata.max()
        margin = (xmax - xmin) * 0.1
        ax.set_xlim(xmin - margin, xmax + margin)
    
    ax.tick_params(axis='x', which='major', labelsize=8, rotation=45)
    
## USAGE For all standards on one plot
# result = plot_slr_standards(metatable=SLR_metatable, threshold_type='rdml_Cq (mean eff)', std0_status = 'exc_std0', figsize=(8, 6), separate_plots=False)


# Identifies the columns that contain ct, Cq, or N0 measurements
# Used when looping subtract_background() against all measurement columns
cycle_list = [
    'ct',
    'rdml_N0 (indiv eff - for debug use)',
    'rdml_Cq (indiv eff - for debug use)',
    'rdml_Cq with group threshold (indiv eff - for debug use)',
    'rdml_N0 (mean eff)',
    'rdml_Cq (mean eff)',
    'rdml_N0 (mean eff) - no plateau',
    'rdml_Cq (mean eff) - no plateau',
    'rdml_N0 (mean eff) - mean efficiency',
    'rdml_Cq (mean eff) - mean efficiency',
    'rdml_N0 (mean eff) - no plateau - mean efficiency',
    'rdml_Cq (mean eff) - no plateau - mean efficiency',
    'rdml_N0 (mean eff) - stat efficiency',
    'rdml_Cq (mean eff) - stat efficiency',
    'rdml_N0 (mean eff) - no plateau - stat efficiency',
    'rdml_Cq (mean eff) - no plateau - stat efficiency',
    'rdml_log2N0 (indiv eff - for debug use)',
    'rdml_log2N0 (mean eff)',
    'rdml_log2N0 (mean eff) - no plateau',
    'rdml_log2N0 (mean eff) - mean efficiency',
    'rdml_log2N0 (mean eff) - no plateau - mean efficiency',
    'rdml_log2N0 (mean eff) - stat efficiency',
    'rdml_log2N0 (mean eff) - no plateau - stat efficiency',
]


def subtract_background(df: pd.DataFrame, threshold_types: Union[str, List[str]]) -> pd.DataFrame:
    """
    Performs background subtraction on threshold values (N0, Cq, or Ct) grouped by usr_std.
    For N0: directly subtracts background
    For Cq/Ct: converts to 2^(-Ct) (invLog2), subtracts background, then converts back to cycles
    The function can only be applied to single experiment metatables and will check for this before processing.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing columns 'usr_std', threshold_types, 'py_known_conc', and 'filepath_csv'.
                         All rows must have the same filepath_csv value (len(df['filepath_csv'].unique()) == 1)
    threshold_types (str or List[str]): Column name(s) for the threshold values (e.g., 'rdml_N0', 'Cq', 'Ct').
                                      Can be a single string or a list of strings.
    
    Returns:
    pandas.DataFrame: Original DataFrame with new column(s) '{threshold_type} sub_bg' for each threshold type
    
    Raises:
    ValueError: If filepath_csv has multiple values or if filepath_csv column is missing
    """
    
    # Validate input
    if 'filepath_csv' not in df.columns:
        raise ValueError("DataFrame should be a metatable with a 'filepath_csv' column")
    
    if len(df['filepath_csv'].unique()) != 1:
        raise ValueError(f"All rows must have the same filepath_csv value. Found {len(df['filepath_csv'].unique())} different values")
    
    # Convert single threshold_type to list for consistent processing
    if isinstance(threshold_types, str):
        threshold_types = [threshold_types]
    
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Process each threshold type
    for threshold_type in threshold_types:
        # Create placeholder column
        df[threshold_type + ' - sub_bg'] = np.nan
        
        # Process each group
        for std_group in df['usr_std'].unique():
            # Get mask for current group
            group_mask = df['usr_std'] == std_group
            
            if '_N0' in threshold_type:
                # Calculate background (mean of blanks where py_known_conc = 0)
                background = df.loc[group_mask & (df['py_known_conc'] == 0), threshold_type].mean()
                
                # Subtract background from all values in group
                df.loc[group_mask, threshold_type + ' - sub_bg'] = df.loc[group_mask, threshold_type] - background
                
            else:
                # Get values for current group
                group_values = df.loc[group_mask, threshold_type]
                
                # Calculate invLog2 values as a Series
                invlog2_values = 2 ** (-group_values)
                
                # Calculate background from invLog2 values where py_known_conc = 0
                background = invlog2_values[df.loc[group_mask, 'py_known_conc'] == 0].mean()
                
                # Subtract background from invLog2 values
                invlog2_bg_subtracted = invlog2_values - background
                
                # Convert back to cycles where values are positive
                positive_mask = invlog2_bg_subtracted > 0
                df.loc[group_mask, threshold_type + ' - sub_bg'] = np.nan  # Initialize all to nan
                if positive_mask.any():
                    df.loc[group_mask & df.index.isin(positive_mask[positive_mask].index), 
                          threshold_type + ' - sub_bg'] = -np.log2(invlog2_bg_subtracted[positive_mask])
    
    return df

## USAGE
# py_metatable_subBG = subtract_background(df = py_metatable, threshold_types = cycle_list)


def calc_py_metatable_all_models(metatable: pd.DataFrame, rdml_check: bool = True, cycle_list = cycle_list, export: bool = True) -> pd.DataFrame:
    """
    Calculate calibration models for qPCR data using multiple regression approaches.
    
    This function processes qPCR data through various calibration models including Simple Linear
    Regression (SLR), 4-Parameter Logistic (4PL), and 5-Parameter Logistic (5PL) regression.
    It handles both background-subtracted and raw data, with different transformation options.
    
    Parameters
    ----------
    metatable : pd.DataFrame
        Input DataFrame containing qPCR metadata and results. Must include 'filepath_txt' column.
    rdml_check : bool, default=True
        Whether to validate against RDML file data.
    export : bool, default=True
        Whether to export the final results to a CSV file.
    cycle_list : list[str], optional
        List of cycle threshold types to process. If None, uses default thresholds.
        
    Returns
    -------
    pd.DataFrame
        Processed metatable containing all calculated models and their parameters.
    
    Notes
    -----
    The function applies the following modeling approaches:
    - Simple Linear Regression (SLR) with log10 and linear transformations
    - 4PL regression with various combinations of:
        - log10 vs linear x-transformation
        - including vs excluding standard 0
    - 5PL regression with the same combinations as 4PL
    """
    
    # Generate initial python metatable with basic calculations
    py_metatable = create_py_metatable(metatable=metatable, rdml_check=rdml_check, 
                                       threshold_type='ct', export=False)
    
    # Calculate and subtract background for all threshold types
    py_metatable = subtract_background(df=py_metatable.copy(), threshold_types=cycle_list)

    # Create expanded cycle list including background-subtracted versions
    cycle_list_bg = [x for item in cycle_list for x in (item, item + ' - sub_bg')]
    
    # Suppress warnings during model fitting
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # 1. Simple Linear Regression (SLR) Calculations
        # Process all threshold types with log10 transformation, excluding std0
        for item in cycle_list_bg:
            py_metatable = calc_metatable_SLR(
                py_metatable.copy(), 
                threshold_type=item, 
                std0_status='exc_std0',
                export=False
            )
        print('SLR; log10(x); exc_std0. Complete. Simple Linear Regression estimated for all threshold values of type ct, Cq and log2N0')
        
        # Process linear N0 thresholds including std0
        SLR_linear_inc_std0_list = [i for i in cycle_list_bg if '_N0' in i]
        for item in SLR_linear_inc_std0_list:
            py_metatable = calc_metatable_SLR(
                py_metatable.copy(), 
                threshold_type=item,
                std0_status='inc_std0',
                export=False
            )
        print('SLR; linear(x); inc_std0. Complete: Simple Linear Regression estimated for all threshold values of type _N0')

        # 2. Four Parameter Logistic (4PL) Calculations
        # Three combinations: 
        # a) log10(x) with exc_std0
        for item in cycle_list_bg:
            py_metatable = calc_metatable_PL(
                py_metatable=py_metatable,
                model='4PL',
                threshold_type=item, 
                transform_x='log10(x)',
                std0_status='exc_std0',
                export=False
            )
        print('4PL; log10(x); exc_std0. Complete: Four Parameter Logistic estimated for all threshold values')
        
        # b) linear(x) with exc_std0
        for item in cycle_list_bg:
            py_metatable = calc_metatable_PL(
                py_metatable=py_metatable,
                model='4PL',
                threshold_type=item, 
                transform_x='linear(x)',
                std0_status='exc_std0',
                export=False
            )
        print('4PL; linear(x); exc_std0. Complete: Four Parameter Logistic estimated for all threshold values')
        
        # c) linear(x) with inc_std0
        for item in cycle_list_bg:
            py_metatable = calc_metatable_PL(
                py_metatable=py_metatable,
                model='4PL',
                threshold_type=item, 
                transform_x='linear(x)',
                std0_status='inc_std0',
                export=False
            )
        print('4PL; linear(x); inc_std0. Complete: Four Parameter Logistic estimated for all threshold values')

        # 3. Five Parameter Logistic (5PL) Calculations
        # Same three combinations as 4PL
        for item in cycle_list_bg:
            py_metatable = calc_metatable_PL(
                py_metatable=py_metatable,
                model='5PL',
                threshold_type=item, 
                transform_x='log10(x)',
                std0_status='exc_std0',
                export=False
            )
        print('5PL; log10(x); exc_std0. Complete: Five Parameter Logistic estimated for all threshold values')
        
        for item in cycle_list_bg:
            py_metatable = calc_metatable_PL(
                py_metatable=py_metatable,
                model='5PL',
                threshold_type=item, 
                transform_x='linear(x)',
                std0_status='exc_std0',
                export=False
            )
        print('5PL; linear(x); exc_std0. Complete: Five Parameter Logistic estimated for all threshold values')
        
        for item in cycle_list_bg:
            py_metatable = calc_metatable_PL(
                py_metatable=py_metatable,
                model='5PL',
                threshold_type=item, 
                transform_x='linear(x)',
                std0_status='inc_std0',
                export=False
            )
        print('5PL; linear(x); inc_std0. Complete: Five Parameter Logistic estimated for all threshold values')
        
        # Export results if requested
        if export:
            if len(metatable['filepath_txt'].unique().tolist()) == 1:
                proxipal_path = Path(metatable['filepath_txt'].unique().tolist()[0])
                exports_folder = data_folder / proxipal_path.parent / 'exports'
                py_metatable.to_csv(exports_folder / 'py_metatable.csv')
        
        return py_metatable


# def create_plate_visualization(df, plate_format=96, palette=None, font_size=8, 
#                              value1=('rep_id', 'Rep: '), value2=('dilution', 'Dil: '),
#                              heatmap=False, heatmap_palette="vlag", heatmap_value=('ct', 'Cycle: '),
#                              cmap_exclusions=None):
#     """
#     Create a visualization of a microplate with sample information.
#     Colors cells based on rep_id for non-standard/QC wells or as a heatmap based on a specified value.
    
#     Args:
#         df: DataFrame with columns sample_id, position, rep_id, dilution
#         plate_format: Integer specifying total wells (default: 96)
#                      Supported formats: 6, 12, 24, 48, 96, 384, 1536
#         palette: List of hex color codes or None (default: uses Set2 palette)
#                 e.g., ['#1f77b4', '#ff7f0e', '#2ca02c', ...]
#         font_size: Base font size for text (default: 8)
#         value1: Tuple of (column_name, label) for first displayed value (default: ('rep_id', 'Rep: '))
#         value2: Tuple of (column_name, label) for second displayed value (default: ('dilution', 'Dil: '))
#         heatmap: Boolean to enable heatmap mode (default: False)
#         heatmap_palette: String specifying seaborn color palette for heatmap (default: "vlag")
#         heatmap_value: Tuple of (column_name, label) for heatmap value (default: ('ct', 'Cycle: '))
#         cmap_exclusions: List of well positions to exclude from color mapping and heatmap scaling (default: None)
#                         e.g., ['A1', 'H12']
#     """
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     import numpy as np
    
#     # Define plate dimensions for common formats
#     plate_dimensions = {
#         6: (2, 3),
#         12: (3, 4),
#         24: (4, 6),
#         48: (6, 8),
#         96: (8, 12),
#         384: (16, 24),
#         1536: (32, 48)
#     }
    
#     if plate_format not in plate_dimensions:
#         raise ValueError(f"Unsupported plate format: {plate_format}. Must be one of {list(plate_dimensions.keys())}")
    
#     # Validate that the requested columns exist in the dataframe
#     required_columns = ['sample_id', 'position', value1[0], value2[0]]
#     if heatmap:
#         required_columns.append(heatmap_value[0])
#     missing_columns = [col for col in required_columns if col not in df.columns]
#     if missing_columns:
#         raise ValueError(f"Missing required columns in dataframe: {missing_columns}")
    
#     # Initialize cmap_exclusions if None
#     if cmap_exclusions is None:
#         cmap_exclusions = []
    
#     num_rows, num_cols = plate_dimensions[plate_format]
    
#     # Create figure and axis
#     # Adjust figure size based on plate format and heatmap mode
#     fig_width = min(24, num_cols * 1.25)
#     fig_height = min(16, num_rows * 1)
#     if heatmap:
#         fig_height += 2  # Add extra height for colorbar
    
#     fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
#     # Hide axes
#     ax.set_xticks([])
#     ax.set_yticks([])
    
#     # Create grid for wells
#     rows = [chr(65 + i) for i in range(num_rows)]  # Generate row labels (A, B, C, ...)
#     cols = list(range(1, num_cols + 1))
    
#     # Set up color mapping
#     if heatmap:
#         # Create color normalization based on heatmap values, excluding specified wells
#         heatmap_values = df[~df['position'].isin(cmap_exclusions)][heatmap_value[0]]
#         vmin = heatmap_values.min()
#         vmax = heatmap_values.max()
#         norm = plt.Normalize(vmin=vmin, vmax=vmax)
#         cmap = sns.color_palette(heatmap_palette, as_cmap=True)
#     else:
#         # Use provided palette or default to Set2
#         if palette is None:
#             color_palette = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', 
#                             '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']
#         else:
#             color_palette = palette
        
#         # Create a dynamic color mapping as we encounter rep_ids
#         rep_colors = {}
#         next_color_idx = 0
    
#     # Calculate cell size - adjust based on plate format
#     cell_width = 1
#     cell_height = 1
    
#     def calculate_text_color(facecolor):
#         """Helper function to determine appropriate text color based on background color."""
#         if isinstance(facecolor, (tuple, list, np.ndarray)):
#             # For RGB tuples from heatmap
#             rgb = [int(x * 255) for x in facecolor[:3]]
#             brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
#             return 'black' if brightness > 128 else 'white'
        
#         if facecolor in ['white', 'black', 'grey', 'lightgrey', 'lightblue']:
#             return 'black' if facecolor in ['white', 'lightgrey', 'lightblue'] else 'white'
        
#         hex_color = facecolor.lstrip('#')
#         rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
#         brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
#         return 'black' if brightness > 128 else 'white'

#     # Use the input font size directly, only scaling for very large plate formats
#     fontsize = font_size
#     if plate_format >= 384:
#         fontsize *= 0.75

#     # Draw grid and add well information
#     for i, row in enumerate(rows):
#         for j, col in enumerate(cols):
#             position = f'{row}{col}'
#             well_data = df[df['position'] == position]
            
#             if not well_data.empty:
#                 sample_id = well_data['sample_id'].iloc[0]
#                 val1 = well_data[value1[0]].iloc[0]
#                 val2 = well_data[value2[0]].iloc[0]
                
#                 # Check if it's a standard well or QC well
#                 is_standard = 'std' in str(sample_id).lower() and '[' in str(sample_id)
#                 is_qch = 'QC-H' in str(sample_id) and '[' in str(sample_id)
#                 is_qcm = 'QC-M' in str(sample_id) and '[' in str(sample_id)
#                 is_qcl = 'QC-L' in str(sample_id) and '[' in str(sample_id)
                
#                 # Process the sample_id display consistently for all modes
#                 if is_standard or is_qch or is_qcm or is_qcl:
#                     displayed_sample_id = sample_id.split("_")[0]
#                 else:
#                     displayed_sample_id = sample_id
                
#                 if heatmap and position not in cmap_exclusions:
#                     heatmap_val = well_data[heatmap_value[0]].iloc[0]
#                     facecolor = cmap(norm(heatmap_val))
#                     text = f'{displayed_sample_id}\n{value1[1]}{val1}\n{value2[1]}{val2}\n{heatmap_value[1]}{heatmap_val:.2f}'
#                 else:
#                     if heatmap:
#                         heatmap_val = well_data[heatmap_value[0]].iloc[0]
#                         text = f'{displayed_sample_id}\n{value1[1]}{val1}\n{value2[1]}{val2}\n{heatmap_value[1]}{heatmap_val:.2f}'
#                     else:
#                         text = f'{displayed_sample_id}\n{value1[1]}{val1}\n{value2[1]}{val2}'
                    
#                     # Set cell color
#                     if position in cmap_exclusions:
#                         facecolor = 'white'
#                     elif is_qch:
#                         facecolor = 'black'
#                     elif is_qcm:
#                         facecolor = 'grey'
#                     elif is_qcl:
#                         facecolor = 'lightgrey'
#                     elif is_standard:
#                         facecolor = 'lightblue'
#                     else:
#                         if val1 not in rep_colors:
#                             rep_colors[val1] = color_palette[next_color_idx % len(color_palette)]
#                             next_color_idx += 1
#                         facecolor = rep_colors[val1]
                
#                 text_color = calculate_text_color(facecolor)
                
#             else:
#                 text = 'N/A'
#                 text_color = 'black'
#                 facecolor = 'white'
            
#             rect = plt.Rectangle((j * cell_width, (num_rows-1-i) * cell_height), 
#                                cell_width, cell_height, 
#                                fill=True,
#                                facecolor=facecolor,
#                                ec='black')
#             ax.add_patch(rect)
#             ax.text(j * cell_width + cell_width/2, 
#                    (num_rows-1-i) * cell_height + cell_height/2,
#                    text,
#                    ha='center',
#                    va='center',
#                    fontsize=fontsize,
#                    color=text_color)
    
#     # Add row labels
#     for i, row in enumerate(rows):
#         ax.text(-0.2, (num_rows-1-i) * cell_height + cell_height/2, 
#                 row, 
#                 ha='center', 
#                 va='center', 
#                 fontweight='bold',
#                 fontsize=fontsize)
    
#     # Add column labels
#     for j, col in enumerate(cols):
#         ax.text(j * cell_width + cell_width/2, num_rows + 0.2, 
#                 str(col), 
#                 ha='center', 
#                 va='center', 
#                 fontweight='bold',
#                 fontsize=fontsize)
    
#     # Set figure limits
#     ax.set_xlim(-0.5, num_cols)
#     ax.set_ylim(-0.5, num_rows + 0.5)

#     # Add colorbar if heatmap is enabled
#     if heatmap:
#         # Adjust plot position to maintain size
#         ax.set_position([0.1, 0.2, 0.8, 0.7])  # [left, bottom, width, height]
        
#         # Create a new axis for colorbar below the main plot
#         cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])  # [left, bottom, width, height]
#         sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#         cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal', label=heatmap_value[0])
#         cbar.ax.tick_labels = fontsize
#     else:
#         plt.tight_layout()
        
#     return fig

def create_plate_visualization(df, plate_format=96, palette=None, font_size=8, 
                             value1=('rep_id', 'Rep: '), value2=('dilution', 'Dil: '),
                             heatmap=False, heatmap_palette="vlag", heatmap_value=('ct', 'Cycle: '),
                             cmap_exclusions=None, values_decimal_places=2, heatmap_decimal_places=2):
    """
    Create a visualization of a microplate with sample information.
    Colors cells based on rep_id for non-standard/QC wells or as a heatmap based on a specified value.

    Args:
        df: DataFrame with columns sample_id, position, rep_id, dilution
        plate_format: Integer specifying total wells (default: 96)
        palette: List of hex color codes or None (default: uses Set2 palette)
        font_size: Base font size for text (default: 8)
        value1: Tuple of (column_name, label) for first displayed value
        value2: Tuple of (column_name, label) for second displayed value
        heatmap: Boolean to enable heatmap mode
        heatmap_palette: Seaborn palette for heatmap
        heatmap_value: Tuple of (column_name, label) for heatmap value
        cmap_exclusions: List of well positions to exclude from color mapping
        values_decimal_places: Decimal places for value1 and value2 (if numeric)
        heatmap_decimal_places: Decimal places for heatmap value
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    plate_dimensions = {
        6: (2, 3), 12: (3, 4), 24: (4, 6), 48: (6, 8),
        96: (8, 12), 384: (16, 24), 1536: (32, 48)
    }

    if plate_format not in plate_dimensions:
        raise ValueError(f"Unsupported plate format: {plate_format}. Must be one of {list(plate_dimensions.keys())}")

    required_columns = ['sample_id', 'position', value1[0], value2[0]]
    if heatmap:
        required_columns.append(heatmap_value[0])
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataframe: {missing_columns}")

    if cmap_exclusions is None:
        cmap_exclusions = []

    num_rows, num_cols = plate_dimensions[plate_format]
    fig_width = min(24, num_cols * 1.25)
    fig_height = min(16, num_rows * 1)
    if heatmap:
        fig_height += 2

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xticks([])
    ax.set_yticks([])

    rows = [chr(65 + i) for i in range(num_rows)]
    cols = list(range(1, num_cols + 1))

    if heatmap:
        heatmap_values = df[~df['position'].isin(cmap_exclusions)][heatmap_value[0]]
        vmin = heatmap_values.min()
        vmax = heatmap_values.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = sns.color_palette(heatmap_palette, as_cmap=True)
    else:
        if palette is None:
            color_palette = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3',
                             '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']
        else:
            color_palette = palette
        rep_colors = {}
        next_color_idx = 0

    cell_width = 1
    cell_height = 1

    def calculate_text_color(facecolor):
        if isinstance(facecolor, (tuple, list, np.ndarray)):
            rgb = [int(x * 255) for x in facecolor[:3]]
            brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
            return 'black' if brightness > 128 else 'white'
        if facecolor in ['white', 'black', 'grey', 'lightgrey', 'lightblue']:
            return 'black' if facecolor in ['white', 'lightgrey', 'lightblue'] else 'white'
        hex_color = facecolor.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
        return 'black' if brightness > 128 else 'white'

    fontsize = font_size
    if plate_format >= 384:
        fontsize *= 0.75

    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            position = f'{row}{col}'
            well_data = df[df['position'] == position]

            if not well_data.empty:
                sample_id = well_data['sample_id'].iloc[0]
                val1_raw = well_data[value1[0]].iloc[0]
                val2_raw = well_data[value2[0]].iloc[0]

                # Format val1 and val2 if numeric
                if isinstance(val1_raw, (int, float, np.integer, np.floating)):
                    val1 = f"{val1_raw:.{values_decimal_places}f}"
                else:
                    val1 = str(val1_raw)

                if isinstance(val2_raw, (int, float, np.integer, np.floating)):
                    val2 = f"{val2_raw:.{values_decimal_places}f}"
                else:
                    val2 = str(val2_raw)

                is_standard = 'std' in str(sample_id).lower() and '[' in str(sample_id)
                is_qch = 'QC-H' in str(sample_id) and '[' in str(sample_id)
                is_qcm = 'QC-M' in str(sample_id) and '[' in str(sample_id)
                is_qcl = 'QC-L' in str(sample_id) and '[' in str(sample_id)

                displayed_sample_id = sample_id.split("_")[0] if (is_standard or is_qch or is_qcm or is_qcl) else sample_id

                if heatmap:
                    heatmap_val = well_data[heatmap_value[0]].iloc[0]
                    formatted_val = f"{heatmap_val:.{heatmap_decimal_places}f}"
                    text = f'{displayed_sample_id}\n{value1[1]}{val1}\n{value2[1]}{val2}\n{heatmap_value[1]}{formatted_val}'
                    facecolor = cmap(norm(heatmap_val)) if position not in cmap_exclusions else 'white'
                else:
                    text = f'{displayed_sample_id}\n{value1[1]}{val1}\n{value2[1]}{val2}'
                    if position in cmap_exclusions:
                        facecolor = 'white'
                    elif is_qch:
                        facecolor = 'black'
                    elif is_qcm:
                        facecolor = 'grey'
                    elif is_qcl:
                        facecolor = 'lightgrey'
                    elif is_standard:
                        facecolor = 'lightblue'
                    else:
                        if val1 not in rep_colors:
                            rep_colors[val1] = color_palette[next_color_idx % len(color_palette)]
                            next_color_idx += 1
                        facecolor = rep_colors[val1]

                text_color = calculate_text_color(facecolor)
            else:
                text = 'N/A'
                text_color = 'black'
                facecolor = 'white'

            rect = plt.Rectangle((j * cell_width, (num_rows - 1 - i) * cell_height), 
                                 cell_width, cell_height, 
                                 fill=True, facecolor=facecolor, ec='black')
            ax.add_patch(rect)
            ax.text(j * cell_width + cell_width / 2, 
                    (num_rows - 1 - i) * cell_height + cell_height / 2,
                    text, ha='center', va='center',
                    fontsize=fontsize, color=text_color)

    for i, row in enumerate(rows):
        ax.text(-0.2, (num_rows - 1 - i) * cell_height + cell_height / 2, 
                row, ha='center', va='center', fontweight='bold', fontsize=fontsize)
    for j, col in enumerate(cols):
        ax.text(j * cell_width + cell_width / 2, num_rows + 0.2, 
                str(col), ha='center', va='center', fontweight='bold', fontsize=fontsize)

    ax.set_xlim(-0.5, num_cols)
    ax.set_ylim(-0.5, num_rows + 0.5)

    if heatmap:
        ax.set_position([0.1, 0.2, 0.8, 0.7])
        cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal', label=heatmap_value[0])
        cbar.ax.tick_params(labelsize=fontsize)
    else:
        plt.tight_layout()

    return fig



# # Example usage:
# fig1 = create_plate_visualization(py_metatable, plate_format = 96, palette = ['#FFFFFF'], font_size=12)
# fig2 = create_plate_visualization(py_metatable, plate_format = 96, font_size=12)

# # To show only fig1:
# plt.close('all')  # Close all existing figures
# plt.figure(fig2)
# plt.show()

def summarise_signal(dataframe: pd.DataFrame, signal_name: str, signal_col: str, cycles: tuple = (0,30)) -> pd.DataFrame:
    """
    Summarize a given signal from a dataframe by extracting the specified 'cycles' values, 
    computing their mean and standard deviation, and returning the dataframe with these added columns.
    
    Parameters:
    - dataframe (pd.DataFrame): The input dataframe.
    - signal_name (str): The name of the signal (used for creating new column names).
    - signal_col (str): The column in the dataframe that contains the signal data.
    - cycles (tuple): The range of signal values to consider for summarization. Default is (0, 30).
    
    Returns:
    - pd.DataFrame: The dataframe with added columns for the mean and standard deviation of the signal.
    """
    
    # Create a copy of the input dataframe to avoid in-place modifications
    df = dataframe.copy()

    # Convert string representation of list to actual list of floats
    def convert_to_float_list(x):
        if isinstance(x, str):
            x = x.replace("'", "").strip('[]').split(',')
            return [float(i) for i in x]
        return []

    df[signal_col] = df[signal_col].apply(convert_to_float_list)

    # Adjust start and end based on cycles and the available data
    start, end = cycles
    if start is None:
        start = 0
    if end is None:
        end = len(df[signal_col].iloc[0])

    # Extract the range specified in 'cycles' from the signal column
    df['cycle_range'] = df[signal_col].apply(lambda x: x[start:end])
    
    # Calculate the mean for these values
    df['mean_name'] = df['cycle_range'].apply(lambda x: sum(x)/len(x) if len(x) != 0 else np.nan)
    
    # Calculate the standard deviation for these values
    df['stdev_name'] = df['cycle_range'].apply(lambda x: (sum([(i - sum(x)/len(x))**2 for i in x])/len(x))**0.5 if len(x) != 0 else np.nan)

    # Calculate suffix based on the length of cycle_range
    suffix = str(len(df['cycle_range'].iloc[0]))

    # Rename columns based on the computed suffix
    df = df.rename(columns={
        'cycle_range': signal_name + '_' + suffix,
        'mean_name': signal_name + '_mean_' + suffix,
        'stdev_name': signal_name + '_stdev_' + suffix
    })

    return df

# # Usage
# # Calculate mean and standard deviations for cycling by reaction (by row)
# df = summarise_signal(data, 'rox', 'rox (eds; multicomponent data)', cycles=(0,30))
# df[['rox_30','rox_mean_30', 'rox_stdev_30']].head()

def analyze_sample_stats(data: pd.DataFrame, 
                        model_type: str = 'SLR; log10(x); exc_std0; ct; raw_ng/L', 
                        decimal_places: Union[int, str] = 0, 
                        format_timing: str = 'before') -> pd.DataFrame:
    """
    analyze_sample_stats() → performs the pooled CV calculation. It analyses single reactions from all measurements in all assays for each sample_id and provides a simple statistical 
    summary (mean, stdev, CV). This function differs from analyse_imprecision() in that no distinctions are made between intra or inter assay performance and outlier filters are not offered. 
    
    Parameters:
    -----------
    data: DataFrame with columns 'sample_id' and measurement values
    model_type: Name of the column containing measurement values
    decimal_places: Controls output format
        - int (0,1,2..): Number of decimal places to round to (returns float)
        - 'integer': Returns integer values
    format_timing: When to apply formatting
        - 'after': Format after calculations (more precise)
        - 'before': Format before calculations (matches displayed values)
    
    Returns:
    --------
    DataFrame: Statistics for each sample_id
    """

    if format_timing == 'before':
        print("""Attention User! Setting format_timing = before. Therefore Mean and Standard Deviation values will be rounded prior to calculating imprecision ranges. """
              """Check tables carefully for evidence of rounding errors. For example: """)
        print("Setting format_timing = before. Lower bound evaluated as round(mean) - round(2*standard deviation)")
        print("Setting format_timing = after. Lower bound evaluated as round(mean - (2*standard deviation))")
    elif format_timing == 'after':
        print("User format_timing = after. Imprecision ranges will first be calculated from native Mean and Standard Deviation values. Rounding will be applied to the ranges afterwards.")

    def format_number(value: float) -> Union[int, float]:
        """Format a number according to specified decimal places or as integer"""
        if pd.isna(value):
            return np.nan
        if decimal_places == 'integer':
            return int(round(value))
        else:
            return round(value, decimal_places)
    
    def calculate_stats(group: pd.Series) -> pd.Series:
        """Calculate statistical measures for a group of samples"""
        # Drop NaN and ensure numeric
        group = pd.to_numeric(group, errors='coerce').dropna()
        n: int = len(group)
        if n == 0:
            return pd.Series({
                'mean': np.nan, 'std': np.nan, 'cv_percent': np.nan,
                'ci_lower': np.nan, 'ci_upper': np.nan, 'n_samples': 0,
                'std1_range': [np.nan, np.nan],
                'std2_range': [np.nan, np.nan],
                'std3_range': [np.nan, np.nan]
            })

        mean: float = group.mean()
        std: float = group.std()
        cv: float = (std / abs(mean)) * 100 if mean != 0 else np.nan
        
        # Calculate 95% confidence interval
        if n > 1:
            ci: Tuple[float, float] = stats.t.interval(
                confidence=0.95,
                df=n-1,
                loc=mean,
                scale=stats.sem(group)
            )
        else:
            ci = (mean, mean)

        if format_timing == 'before':
            mean = format_number(mean)
            std1 = format_number(std)
            std2 = format_number(2*std)
            std3 = format_number(3*std)

            std1_lower = format_number(mean) - format_number(std1)
            std1_upper = format_number(mean) + format_number(std1)
            std2_lower = format_number(mean) - format_number(std2)
            std2_upper = format_number(mean) + format_number(std2)
            std3_lower = format_number(mean) - format_number(std3)
            std3_upper = format_number(mean) + format_number(std3)
        
        else:  # 'after'
            std1 = std
            std2 = 2 * std
            std3 = 3 * std
            
            std1_lower = format_number(mean - std1)
            std1_upper = format_number(mean + std1)
            std2_lower = format_number(mean - std2)
            std2_upper = format_number(mean + std2)
            std3_lower = format_number(mean - std3)
            std3_upper = format_number(mean + std3)
            
            mean = format_number(mean)
            std1 = format_number(std)
            
        return pd.Series({
            'mean': format_number(mean),
            'std': round(std, 2),
            'cv_percent': round(cv, 1) if not pd.isna(cv) else np.nan,
            'ci_lower': round(ci[0], 1) if not pd.isna(ci[0]) else np.nan,
            'ci_upper': round(ci[1], 1) if not pd.isna(ci[1]) else np.nan,
            'n_samples': n,
            'std1_range': [std1_lower, std1_upper],
            'std2_range': [std2_lower, std2_upper],
            'std3_range': [std3_lower, std3_upper]
        })
    
    def custom_sort_key(sample_id: str) -> Tuple[int, float, str]:
        has_std: bool = 'std' in sample_id.lower()
        match: Union[re.Match, None] = re.search(r'\[([\d.]+)\]', sample_id)
        if has_std:
            if match:
                return (1, float(match.group(1)), sample_id)
            return (1, float('inf'), sample_id)
        else:
            if match:
                return (0, float(match.group(1)), sample_id)
            return (0, float('inf'), sample_id)
    
    if format_timing not in ['before', 'after']:
        raise ValueError("format_timing must be either 'before' or 'after'")
    
    results: List[pd.Series] = []
    for sample_id, group in data.groupby('sample_id'):
        stats_series = calculate_stats(group[model_type])
        stats_series['sample_id'] = sample_id
        results.append(stats_series)
    
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[['sample_id', 'mean', 'std', 'cv_percent', 
                        'ci_lower', 'ci_upper', 'n_samples',
                        'std1_range', 'std2_range', 'std3_range']]
    
    stats_df = stats_df.sort_values('sample_id', key=lambda x: x.map(custom_sort_key)).reset_index(drop=True)
    
    return stats_df


# # Example Usage
# stats_results = analyze_sample_stats(data_f, model_type = 'usr_raw_ng/L', decimal_places = 'integer', format_timing = 'before')
# stats_results

def calculate_intraAssay_signal(mastertable, 
                              signals_dict: dict = {
                                  'rox': 'rox (eds; multicomponent data)',
                                  'sybr': 'sybr (eds; multicomponent data)'
                              }, 
                              reference_dye: str = 'rox',
                              cycles=(0,10)):
    """
    Calculate intra-assay signals for multiple signal measurements.
    
    Args:
        mastertable: Input data table
        signals_dict: Dictionary of {signal_name: signal_col} pairs. 
                     Defaults to ROX and SYBR signals.
        reference_dye: Signal to use as reference for delta calculations.
                      Set to None to skip delta calculations.
                      Defaults to 'rox'.
        cycles: Tuple of (start, end) cycle numbers. Defaults to (0,10).
    
    Returns:
        DataFrame with calculations for all signals
    """
    # Initialize empty DataFrame to store results
    final_ia_df = None
    
    print("\nProcessing signals and adding columns:")
    print("-" * 40)
    
    # Loop through each signal in the dictionary
    for signal_name, signal_col in signals_dict.items():
        print(f"\nProcessing signal: {signal_name} (column: {signal_col})")
        
        # Apply summarise_signal()     
        df = summarise_signal(mastertable, signal_name=signal_name, signal_col=signal_col, cycles=cycles)
        
        # Calculate cycle length
        len_cycle = cycles[1] - cycles[0]
        
        # Define column names for this signal
        mean_name = f"{signal_name}_mean_{len_cycle}"
        stdev_name = f"{signal_name}_stdev_{len_cycle}"
        
        # Transform and normalise
        ia_df = df.copy()
        
        # Create new column names
        new_columns = [
            f"{mean_name}_div1k",
            f"{stdev_name}_div1k",
            f"{mean_name}_norm"
        ]
        
        # Add calculations with the new column names
        ia_df[new_columns[0]] = (ia_df[mean_name]/1000).round(1)
        ia_df[new_columns[1]] = (ia_df[stdev_name]/1000).round(1)
        ia_df[new_columns[2]] = (ia_df[mean_name] / ia_df[mean_name].mean()).round(2)
        
        # Print the new columns being added
        print(f"Added columns for {signal_name}:")
        for col in new_columns:
            print(f"  → {col}")
        
        # For first iteration, initialize final_ia_df
        if final_ia_df is None:
            final_ia_df = ia_df
            print(f"Initial columns created: {len(ia_df.columns)}")
        else:
            # Merge new calculations with existing results
            # Using all columns from current ia_df except those that already exist
            new_cols = [col for col in ia_df.columns if col not in final_ia_df.columns]
            final_ia_df = final_ia_df.join(ia_df[new_cols])
            print(f"New columns added: {len(new_cols)}")
            print("Newly added columns:")
            for col in new_cols:
                print(f"  → {col}")
    
    # Calculate delta values if reference_dye is specified
    if reference_dye is not None:
        if reference_dye not in signals_dict:
            raise ValueError(f"Reference dye '{reference_dye}' not found in signals_dict")
            
        print(f"\nCalculating delta values using {reference_dye} as reference:")
        
        # Get all signals except reference_dye
        other_signals = [key for key in signals_dict.keys() if key != reference_dye]
        
        # Reference column name
        ref_col = f"{reference_dye}_mean_{len_cycle}_div1k"
        
        for signal in other_signals:
            # Create delta column names
            delta_col = f"delta_{reference_dye}_{signal}_{len_cycle}cycles"
            signal_col = f"{signal}_mean_{len_cycle}_div1k"
            
            # Calculate delta values
            final_ia_df[delta_col] = final_ia_df[ref_col] - final_ia_df[signal_col]
            final_ia_df[f"{delta_col}_norm"] = (final_ia_df[delta_col] / final_ia_df[delta_col].mean()).round(2)
            
            print(f"  → Added {delta_col}")
            print(f"  → Added {delta_col}_norm")
    
    print('NB: Normalised values are calculated on the historical mean across all wells NOT position-specific normalisation')
    print("\nFinal DataFrame Summary:")
    print(f"Total columns: {len(final_ia_df.columns)}")
    print("-" * 40)
    
    return final_ia_df

# # Example Usage:
# data_signal = calculate_intraAssay_signal(mastertable = mastertable,
#                                     signals_dict = {'rox': 'rox (eds; multicomponent data)',
#                                                     'sybr': 'sybr (eds; multicomponent data)'}, 
#                                     reference_dye = 'rox',
#                                     cycles=(0,10))
# threshold_type = 'rdml_log2N0 (mean eff) - no plateau - stat efficiency'
# raw_conc = 'SLR; log10(x); exc_std0; rdml_log2N0 (mean eff) - no plateau - stat efficiency; raw_ng/L'

# # Round as required for a pretty plot
# data_signal = data_signal.copy()
# data_signal[threshold_type] = round(data_signal[threshold_type], 1)
# data_signal[raw_conc]  = round(data_signal[raw_conc], 1)

# e91_path = '241128_e91_kruti_ClinValP1_QB/241128/241128_NfL-e91_kruti_ClinValP1.csv'
# e91_df = data_signal[data_signal['filepath_csv']==e91_path]

# # Example usage:
# e91_rox = create_plate_visualization(e91_df, plate_format = 96, palette = ['#FFFFFF'], font_size=8,
#                                     value1=(threshold_type, 'Cycle: '), value2=(raw_conc, 'ng/L: '),
#                                     heatmap = True, heatmap_palette="Spectral", heatmap_value=('rox_mean_10_norm', 'Rnorm: '), 
#                                     cmap_exclusions = ['A5','A6', 'B5', 'B6', 'C5', 'C6', 'D5', 'D6',
#                                                         'E5', 'E6', 'F5', 'F6', 'G5', 'G6', 'H5', 'H6'])

# # To show only fig1:
# plt.close('all')  # Close all existing figures
# plt.figure(e91_rox)
# plt.title('e91 ROX mean; Quant B; cycles 0-10', loc='center', y=27)  # y > 1.0 moves it up
# plt.show()
