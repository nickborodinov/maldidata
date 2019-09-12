#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created JL2418 ML

@author: izl
"""
#%% Import necessary modules
import os
import matplotlib.pyplot as plt
import h5py
from MALDIdata import MALDIdata
import sys
# Helper Fuctions
def soft_append(container, addendum):
    '''
    Appends addemdum item to container only if addendum is not already member
    of container.
    
    Input:
    --------
    container : list
        container object to be appended
    addendum : any
        value to be soft-appended to container
    '''
    if addendum not in container:
        container.append(addendum)
        return
    return

def print_indexed_spectrum(x, y_data, y_index, xlabel='', ylabel='', title='', save=False):
    """
    Helper function for easily displaying indexed spectra.

    x: list or array of floats
        x axis ticks

    y_data: list or array of floats
        height of data points on y axis

    y_index: list or array of integers
        index of y_data values in x list

    xlabel: string
        x axis label in generated figure
        default = ''

    ylabel: string
        y axis label in generated figure
        default = ''

    title: string
        title of generated figure, also used as filename if saved

    save: bool
        saves figure to title.png if True
    """
    y = np.zeros_like(x)
    #Populate y from y_data, using y_index to insert values into y
    for i, j in enumerate(y_index):
        y[j] = y_data[i]
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #If save is True, saves figure. Constructs filename from provided tile
    if save:
        filename = title + '.png'
        plt.savefig(filename)
    #Else, displays the figure
    else:
        plt.show()
    return

def getIonImage(mz, MALDIdata, tol=0.1, savepath=''):
    """
    #!!! DEPRECATED
    Helper function for easily displaying ion images.

    mz: float
        m/z ratio of ion to be queried

    MALDIdata: MALDIdata object
        MALDIdata instance to query for mz intensity and distribution data

    tol: float
        ion image mass tolerance. Result mass image is sum of all intensities
        from mz-tol to mz+tol
        default = 0.1

    savepath: string
        path to save ion image.png, requires absolute path
        default='' (creates file in CWD)
    """
    #Identify mz window
    mz_center = mz
    mz_low = mz_center - tol
    mz_high = mz_center + tol
    #Extract bounds for indices of ions to be summed using getIonIndex()
    idx_bounds = (getIonIndex(mz_low, MALDIdata.mass_bar), getIonIndex(mz_high, MALDIdata.mass_bar))
    #Infer remaining indices from extrema
    idx = range(idx_bounds[0], idx_bounds[1])
    #Try to initialize sum map as map of first indexed ion
    try:
        sum_map = MALDIdata.data[:, :, idx[0]]
    except:
        #If no ion can be found, prints warning and returns prematurely
        print('No index found for m/z = %d'%(mz))
        return
    #If ion is found, add remaining intensity maps to sum map
    for i, j in enumerate(idx):
        if i == 0:
            #Skip first map, already used to initialize
            pass
        else:
            sum_map = sum_map + MALDIdata.data[:, :, j]
    #Construct figure
    plt.figure()
    plt.imshow(sum_map)
    plt.set_cmap('Greys')
    plt.colorbar()
    #Image filename inferred from input variables
    imgName = 'ionImage_%s_tol_%s.png'%(str(mz), str(tol))
    #If savepath is provided, rename image with savepath
    if savepath != '':
        imgName = os.path.join(savepath, imgName)
    #Save figure to PNG
    plt.savefig(imgName)
    #Close figure window
    plt.close()
    return

def getIonIndex(mz, array):
    """
    #!!! DEPRECATED
    Helper function for coverting m/z values to nearest indices for extraction.

    Returns index of first entry in array to be less than or equal to mz,
    assumes array is in increasing (or otherwise optimized) sort order.

    mz: float
        m/z ratio of ion to located in array
    """
    #Iterate through enumerator of array
    for i, j in enumerate(array):
        #If input value is greater than array value, return index of array value
        if j >= mz:
            return i
    #If no appropriate value is found, print alert and return None
    print('No index found for ion %i.'%(mz))
    return None

def find_target_ions(target_list, h5_filename):
    hf = h5py.File(h5_filename, 'r')
    average_spectrum = np.array(hf['Squared_data']['ROI_01']['Average spectrum'])
    hf.close()
    hits = {}
    for species, ion_list in target_list.items():
        intensity_list = []
        for ion in ion_list:
            mindex = find_nearest_member(mz_bar, ion)
            intensity = experimental_average[mindex]
            intensity_list.append(intensity)
        hits[species] = intensity_list
    return hits

#%%
# Paths and file names
'''
Accepts 2 different modes of file declaration:

    Default behavior is to declare files based on manual input into the except
    statement below. Default behavior is achieved by running the script without
    command line or terminal arguments.

    Alternatively can be executed from a command line or terminal, with arg
    imzml_absolute_fname
'''
try:
    #If executed from command line
    imzml_fname_w_path = sys.argv[1]
    dataDir, imzml_fname = os.path.split(imzml_fname_w_path)
    print('\nUsing manually entered paths and filenames.')
except IndexError:
    #Else
    imzml_fname = r"F:\saggital 7\mouse brain saggital 7.imzML"
    dataDir = r'F:\saggital 7'
    imzml_fname_w_path = os.path.join(dataDir, imzml_fname)
    print('Using source-coded paths and filenames.')
#Infer H5 filename from imzML filename
h5f_fname = imzml_fname[:-5] + 'h5'
#Assign absolute filenames
h5f_fname_w_path = os.path.join(dataDir, h5f_fname)
#Set True to record log of script execution, set False to only execute
log = True
# Begins logging if true
if log:
    log_warnings = []
    #Import time module for timekeeping
    import time
    #Record script startup time
    start_time = time.time()
    print('Logging enabled.')
#%% Initialize MALDIdata object
maldi = MALDIdata()
maldi.log = log
print('Converting %s to HDF5'%(imzml_fname))
#%% Import
# Read in from imzML/ibd, normalize, and write to shiny new h5 file
maldi.write_msi_to_hdf(h5f_fname_w_path, imzml_fname_w_path, norm='FAST')
#%% Square
# Crop imported, normalized spectra which do not fit into rectangular window
maldi.square_imzml(h5f_fname_w_path, ROI = 'auto')
#%% Filter
# Apply filters to squared data
maldi.filter_imzml(h5f_fname_w_path, mode = '')
#%% Calculate PCA
#Build ROI list
#Execute PCA on filtered data
maldi.calc_pca(h5f_fname_w_path)
#%%
pca_data = maldi.load_pca_from_hdf(h5f_fname_w_path)
