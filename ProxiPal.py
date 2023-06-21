# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 14:02:26 2023

@author: smith.j
"""
# import ipywidgets as widgets
import os
from pathlib import Path
from typing import Union
# from typing import Dict
# from typing import List
import pandas as pd
import numpy as np
from datetime import datetime
import re
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import uuid
# import warnings
# import qgrid
import sys
import platform

now = datetime.now()
print("OS:\t\t\t", platform.platform())
print("Python version:\t\t", sys.version)

# Administrators should set their own password to higher level Jupyter notebook functions
proxipal_password = 'admin'

# Define paths
cwd  = Path(os.getcwd())
base_path = Path(os.path.join(*cwd.parts[:cwd.parts.index('python')]))
print('Date and Time:\t\t', now.strftime('%Y-%m-%d %H:%M:%S'))
print('Root folder:\t\t', base_path)
print('Python folder:\t\t', cwd)

# Define major directories
templates_folder = base_path / 'templates'
print('Templates folder:\t', templates_folder)
data_folder = base_path / 'data'
print('Data folder:\t\t', data_folder)
samples_folder = base_path / 'samples'
print('Samples folder:\t\t', samples_folder)
quality_folder = base_path / 'quality'
print('Quality folder:\t\t', quality_folder)
user_downloads = base_path / 'user_downloads'
print('User_downloads:\t\t', quality_folder)

# make the necessary folders, should they not exist already
folder_paths = [templates_folder, data_folder, samples_folder, quality_folder, user_downloads]
for folder_path in folder_paths:
    path = Path(folder_path)
    # Check if the folder exists
    if not path.exists():
        # Create the folder
        path.mkdir(parents=True, exist_ok=True)

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
        if export_format == '.txt':
            for file_path in export_file_list:
                if file_path.is_file():
                    with open(file_path, 'r') as file:
                        # Read the file line by line to avoid mixed encoding issues from QuantStudio export formats
                        file_contents = ''
                        for line in file:
                            file_contents += line
                        export_dict[get_file_and_parents(file_path)] = file_contents
                        # print(get_file_and_parents(file_path), 'success!')
                # else:
                    # print(get_file_and_parents(file_path), 'not found!')
                    
        elif export_format == '.csv':
            for file_path in export_file_list:
                if file_path.is_file():
                    export_dict[get_file_and_parents(file_path)] = pd.read_csv(file_path, header=None)
                    # print(get_file_and_parents(file_path), 'success!')
                # else:
                    # print(get_file_and_parents(file_path), 'not found!')
                
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
    Row values of this longform dataframe are given by experimental data
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
    with open(exports_folder / 'usr_vals.txt', 'w') as f:
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
    # Read in the samples table file as a DataFrame
    samples_table = pd.read_csv(path, header=None)
    
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


def calc_metatable_std_lin_reg(metatable: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Calculates the standard linear regression for the given metatable and exports the results as CSV files.
    The function processes the metatable, identifies the standards used, and calculates linear regression
    parameters (gradient, y-intercept) and PCR efficiency for each standard.

    Parameters
    ----------
    metatable : pd.DataFrame
        The input metatable containing data about samples, standards, and their Ct values.

    Returns
    -------
    std_calc_dict : Dict[str, pd.DataFrame]
        A dictionary containing the following DataFrames:
        - '<standard>_report_table': A report table for each standard containing sample_id, position, ct,
          py_mean_ct, py_known_conc, py_mean_ng/L, and py_recovery columns.
        - '<standard>_plot_table': A plot table for each standard containing the data used for plotting.
        - '<standard>_calc_table': A calculation table for each standard containing the calculated values.
        - 'py_metatable': The updated metatable with the calculated values.
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
    
    # All standards have the sample_id notation std#[concentration]_product. Data csv templates permit up to 4 standards
    # We identify all standards in use by regular expression, add a known concentration column to the metatable    
    conc_list = []
    pattern_std = r"std\d+\[(.*?)\]_"
    for s in metatable['sample_id'].tolist():
        match = re.search(pattern_std, s)
        if match:
            conc_list.append(float(match.group(1)))
        else:
            conc_list.append(np.nan)
            
    metatable['py_known_conc'] = conc_list
    
    # Prepare metatable for lin_reg calcs
    metatable = metatable.assign(py_linReg_grad = None, py_linReg_y_int = None, py_linReg_equation = None, 
                                  py_linReg_R2 = None, py_PCReff = None)
    
    # Set "py_mean_ct" to be equal to "ct" where "usr_ignore" is equal to 1
    metatable.loc[metatable["usr_ignore"] == 1, "py_mean_ct"] = metatable.loc[metatable["usr_ignore"] == 1, "ct"]
    
    # Select rows where "usr_ignore" is not equal to 1
    filtered_df = metatable[metatable["usr_ignore"] != 1]
    
    # Calculate the mean of the replicate groups
    filtered_df["py_mean_ct"] = filtered_df.groupby("rep_id")["ct"].transform("mean")
      
    # Identify how many standards are in use via the usr_std column
    stds_used_list = filtered_df['usr_std'].unique().tolist()
      
    # std_calc_dict will hold the dataframes used for reporting and linreg calculations
    std_calc_dict = {}
    
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
    
    
    # Begin iteration on stds present
    for i in stds_used_list:
        std = 'std' + str(i)
        pattern = r"{}\[(.*?)\]_".format(std)
        
        # create metatable subset on pattern
        std_df = filtered_df[filtered_df['sample_id'].str.contains(pattern)]
    
        # Limit std_df to columns of interest
        std_df = std_df.loc[:, ['position', 'eds_id', 'target', 'ct', 'tm', 'sample_id', 'tube_id', 
                                'rep_id', "dilution", "usr_std", "usr_raw_ng/L", "usr_mean_ng/L", 
                                "usr_recovery", 'usr_ignore', 'py_known_conc', 'py_mean_ct']]
               
        # Using column "py_known_conc" in std1_subdf dataframe calculate row-wise log10 and add the result under the column "py_known_conc_log10"
        log10_py_known_conc = np.log10(std_df['py_known_conc'])
        std_df['py_known_conc_log10'] = np.where(np.isinf(log10_py_known_conc), np.nan, log10_py_known_conc)
    
        # Collapse std_df into one row per duplicate trend_df; this processes entries as str so reassign types as int/float where possible.
        # This process turns every value in trend_df into a string
        trend_df = std_df.groupby('sample_id').agg(join_unique).sort_values('py_mean_ct', ascending=True).reset_index()
    
        # Following a string-based row collapse we'll have to return some trend_df values to float or integers
        trend_df["rep_id"] = trend_df["rep_id"].astype(int)
        trend_df["dilution"] = trend_df["dilution"].astype(float)
        trend_df["usr_std"] = trend_df["usr_std"].astype(float)
        trend_df["usr_mean_ng/L"] = trend_df["usr_mean_ng/L"].astype(float)
        trend_df["py_known_conc"] = trend_df["py_known_conc"].astype(float)
        trend_df["py_known_conc_log10"] = trend_df["py_known_conc_log10"].astype(float)
        trend_df["py_mean_ct"] = trend_df["py_mean_ct"].astype(float) #
    
        # drop na values from trend_df for regression
        valid_data = trend_df.dropna(subset=['py_known_conc_log10', 'py_mean_ct'])
        x = valid_data['py_known_conc_log10'].values.reshape(-1, 1)
        y = valid_data['py_mean_ct'].values.reshape(-1, 1)
            
        model = LinearRegression()
        model.fit(x, y)
        gradient = model.coef_[0][0]
        y_intercept = model.intercept_[0]
        line_eq = f"y = {gradient:.4f}x + {y_intercept:.4f}"
        
        y_pred = model.predict(x)
        r_square = r2_score(y, y_pred)
    
        # Add new columns to trend_df
        trend_df['py_linReg_equation'] = line_eq
        trend_df['py_linReg_R2'] = round(r_square, 4)
        trend_df['py_linReg_grad'] = round(gradient, 4)
        trend_df['py_linReg_y_int'] = round(y_intercept, 4)
        trend_df['py_PCReff'] = round((10**(-1/gradient) - 1) * 100, 2)
                
        # Row-wise calculate function mean calc_conc_from_linear_regression( )
        trend_df['py_mean_ng/L'] = trend_df.apply(lambda row: round(calc_conc_from_linear_regression(row['py_mean_ct'], row['dilution'], y_intercept, gradient), 4), axis=1)
    
        # Row-wise calculate recoveries
        trend_df['py_mean_recovery'] = trend_df.apply(lambda row: round(row['py_mean_ng/L'] / row['py_known_conc'], 2) if row['py_known_conc'] != 0 else np.nan, axis=1)
        
        # Define dataframe for report printing
        report_df = trend_df[['sample_id','position','ct', 'py_mean_ct', 'py_known_conc', 'py_mean_ng/L', 'py_mean_recovery']]
        
        # Prepare trend_df to merge back onto std_df. Ensure all values for tube_id and sample_id are strings
        std_df['tube_id'] = std_df['tube_id'].astype(str)
        std_df['sample_id'] = std_df['sample_id'].astype(str)
        
        trend_df['tube_id'] = trend_df['tube_id'].astype(str)
        trend_df['sample_id'] = trend_df['sample_id'].astype(str)
    
        # Define unique trend_df columns to add to std_df
        columns_to_keep = ['sample_id', 'tube_id', 'py_linReg_equation', 'py_linReg_R2', 'py_linReg_grad', 'py_linReg_y_int', 'py_PCReff', 'py_mean_ng/L', 'py_mean_recovery']
        trend_merge = trend_df[columns_to_keep]
    
        # Merge to std_df on tube_id or sample_id
        std_df = std_df.merge(trend_merge, on=['tube_id', 'sample_id'], how='outer')
    
        std_calc_dict[std + '_report_table'] = report_df
        std_calc_dict[std + '_plot_table'] = trend_df
        std_calc_dict[std + '_calc_table'] = std_df
    
        # Update the metatable for the relevant std #
        metatable.loc[metatable['usr_std'] == i, 'py_linReg_grad'] = round(gradient, 4)
        metatable.loc[metatable['usr_std'] == i, 'py_linReg_y_int'] = round(y_intercept, 4)
        metatable.loc[metatable['usr_std'] == i, 'py_linReg_equation'] = line_eq
        metatable.loc[metatable['usr_std'] == i, 'py_linReg_R2'] = round(r_square, 4)
    
        export_filenames = ['py_' + std + '_report_table.csv','py_' + std + '_plot_table.csv', 'py_' + std + '_calc_table.csv']
        
        report_df.to_csv(exports_folder / export_filenames[0], index=False)
        trend_df.to_csv(exports_folder / export_filenames[1], index=False)
        std_df.to_csv(exports_folder / export_filenames[2], index=False)
    
    # Update metatable with calculations
    mask = metatable["usr_ignore"] != 1
    metatable.loc[mask, "py_mean_ct"] = metatable.loc[mask].groupby("rep_id")["ct"].transform("mean")
    metatable.loc[mask, 'py_PCReff'] = metatable.loc[mask].apply(lambda row: round((10**(-1/row['py_linReg_grad']) - 1) * 100, 2), axis=1)
    metatable.loc[mask, 'py_raw_ng/L'] = metatable.loc[mask].apply(lambda row: round(calc_conc_from_linear_regression(row['ct'], row['dilution'], row['py_linReg_y_int'], row['py_linReg_grad']), 4), axis=1)
    metatable.loc[mask, 'py_mean_ng/L'] = metatable.loc[mask].apply(lambda row: round(calc_conc_from_linear_regression(row['py_mean_ct'], row['dilution'], row['py_linReg_y_int'], row['py_linReg_grad']), 4), axis=1)   
    metatable.loc[mask, 'py_raw_recovery'] = metatable.loc[mask].apply(lambda row: round(row['py_raw_ng/L'] / row['py_known_conc'], 2) if row['py_known_conc'] != 0 else np.nan, axis=1)
    metatable.loc[mask, 'py_mean_recovery'] = metatable.loc[mask].apply(lambda row: round(row['py_mean_ng/L'] / row['py_known_conc'], 2) if row['py_known_conc'] != 0 else np.nan, axis=1)
    
    # Write calculated metatable as py_metatable
    std_calc_dict['py_metatable'] = metatable
    metatable.to_csv(exports_folder / 'py_metatable.csv', index=False)
    
    # Create a collapsed sample table, akin to the standards table for reporting
    sample_table_brief = metatable[['sample_id', 'tube_id', 'position', 'target', 'usr_ignore', 'ct', 'usr_mean_ng/L', 'py_mean_ct', 'py_mean_ng/L', 'calibrator']]
    
    pattern = r"std\d+\[(.*?)\]_"
    
    # 1. Drop any row with the regular expression defined by 'pattern'
    sample_table_brief = sample_table_brief[~sample_table_brief['sample_id'].str.contains(pattern)]
    
    # 2. Collapse any row with the same 'sample_id' AND 'tube_id'. String conversion is required where np.nan is present.
    sample_table_brief['tube_id'] = sample_table_brief['tube_id'].astype(str)
    sample_table_brief['sample_id'] = sample_table_brief['sample_id'].astype(str)
    grouped = sample_table_brief.groupby(['sample_id', 'tube_id'])
    
    # 3. If column values are different for any collapsed rows, write them as a string "value 1 / value 2 / value 3 ....."
    collapsed_df = grouped.agg(join_unique).reset_index()
    collapsed_df.to_csv(exports_folder / 'sample_report_table.csv', index=False)
    std_calc_dict['sample_report_table'] = collapsed_df
    
    return std_calc_dict

# USAGE
# std_calc_dict = calc_metatable_std_lin_reg(pd.read_csv(r'C:\Users\smith.j\MEGA\MEGAsync\COLONIAL\ProxiPal\data\230112_e24_kruti\230410\exports\metatable.csv'))


def plot_LinReg(plot_table, w = 775, h = 420):
    """
    This function creates a scatter plot with a linear regression trendline using the given data.
    The data is expected to have columns 'py_known_conc_log10' and 'py_mean_ct', where
    'py_known_conc_log10' is the log10 of the known concentration and 'py_mean_ct' is the mean cycle threshold.
    
    Args:
        plot_table (pd.DataFrame): A DataFrame containing 'py_known_conc_log10' and 'py_mean_ct' columns.
        
    Returns:
        None
    """
    # Drop rows with missing (NaN) values
    plot_table = plot_table.dropna(subset=['py_known_conc_log10', 'py_mean_ct'])
    
    # # A4 paper dimensions in inches
    # A4_WIDTH = 8.27
    # A4_HEIGHT = 11.69
    
    # # Calculate the new width
    # new_width = 1.3 * A4_WIDTH
    
    # Create a scatter plot with py_known_conc_log10 on the x-axis and py_mean_ct on the y-axis
    fig = go.Figure()
    
    # Adjust marker size and style
    fig.add_trace(go.Scatter(x=plot_table['py_known_conc_log10'], y=plot_table['py_mean_ct'], mode='markers', marker=dict(size=10, color='black')))
    
    # Calculate linear trendline
    x = plot_table['py_known_conc_log10']
    y = plot_table['py_mean_ct']
    slope, intercept = np.polyfit(x, y, 1)
    x_trend = np.linspace(min(x), max(x), 100)
    y_trend = slope * x_trend + intercept
    
    # Add linear trendline
    fig.add_trace(go.Scatter(x=x_trend, y=y_trend, mode='lines', name='Linear Trendline', line=dict(color='red', width=1)))
    
    # Set the x-axis and y-axis titles
    # fig.update_layout(xaxis_title='py_known_conc_log10', yaxis_title='py_mean_ct', width=new_width*72, height=A4_HEIGHT*72/2,
    fig.update_layout(xaxis_title='py_known_conc_log10', yaxis_title='py_mean_ct', width=w, height=h,

                      plot_bgcolor='white',
                      xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey'),
                      yaxis=dict(tick0=0, dtick=2, showgrid=True, gridwidth=1, gridcolor='lightgrey'),
                      margin=dict(t=10, b=60, l=80, r=10),
                      showlegend=False)
    
    # Add the trendline equation to the plot
    trendline_equation = f'y = {slope:.4f}x + {intercept:.4f}'
    fig.add_annotation(text=trendline_equation, xref='x domain', yref='y domain', x=0.05, y=0.15, showarrow=False, font=dict(size=14))
    
    # Calculate R-squared value
    r_squared = r2_score(y, slope * x + intercept)
    
    # Add the R-squared value to the plot
    fig.add_annotation(text=f'R² = {r_squared:.4f}', xref='x domain', yref='y domain', x=0.05, y=0.06, showarrow=False, font=dict(size=14))
    
    # Add a border around the plot
    fig.update_layout(
        shapes=[
            dict(type='rect', xref='paper', yref='paper', x0=0, x1=1, y0=0, y1=1, fillcolor='rgba(0, 0, 0, 0)', line=dict(color='black', width=0.1))
        ]
    )
    
    # Show the plot
    fig.show()


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


def load_most_recent_mastertable(folder: Path):
    '''
    Load the most recent CSV file with a "mastertable.csv" pattern from the given folder.
    
    Parameters
    ----------
    folder : Path
        The folder containing the CSV files.
    
    Raises
    ------
    ValueError
        If no "mastertable.csv" files are found in the folder.
    
    Returns
    -------
    most_recent_df : DataFrame
        A pandas DataFrame containing the data from the most recent CSV file.
    '''      
    
    # List all files in the folder and filter only the files with the correct pattern
    files = [f for f in folder.iterdir() if "mastertable.csv" in f.name]
    
    if not files:
        raise ValueError("No mastertable.csv files found in " + str(folder))
    
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

    return most_recent_df, most_recent_file



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

def batch_py_metatables(eds2txt_match_dict: dict, eds2csv_match_dict: dict) -> None:
    """
    Given two dictionaries that match EDS files to txt and csv files respectively, this function
    processes each matched file set in a batch.

    :param eds2txt_match_dict: dictionary that maps EDS files to txt files.
    :param eds2csv_match_dict: dictionary that maps EDS files to csv files.

    The function iterates over each file set, checks if the txt and csv files exist for the path_key,
    and if they do, it tries to create a data metatable and calculate standard deviations using linear regression.
    If the files don't exist or if the processing fails due to FileNotFoundError, an appropriate message is printed.
    """
    
    # Generate pivot dataframe from the matched filenames
    df_pivot = review_matched_filenames(eds2txt_match_dict, eds2csv_match_dict)
    
    # Iterate over the 'path_key' list
    for path in df_pivot['path_key'].tolist():
        # Check if both 'txt' and 'csv' files exist for the given 'path_key'
        if (df_pivot.loc[df_pivot['path_key'] == path, ['txt', 'csv']].all(axis=1)).any():
            try:
                # If they exist, try to create a data metatable and calculate standard deviations
                create_data_metatable(eds2txt_match_dict, eds2csv_match_dict, path)
                path_csv = data_folder / (path + '.csv')
                path_metatable = path_csv.parent / 'exports/metatable.csv'
                calc_metatable_std_lin_reg(pd.read_csv(path_metatable))
                print(path, 'processed')
            except FileNotFoundError:
                # If processing fails, print an error message
                print(path, 'create_data_metatable() or calc_metatable_std_lin_reg() failed')
        else:
            # If 'txt' and 'csv' files do not exist, print an error message
            print(path, 'is missing, or mislabeled, an eds, txt, or csv file and cannot be processed')
            
    return None
