# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 10:23:10 2018

@author: s4k
"""

#%%
from MALDIdata import MALDIdata
import matplotlib.pyplot as plt
import os.path
import numpy as np
import h5py
#%%
h5f_path=r'E:\TylerData\MouseBrains\Slide 1'
h5f_fname=r'FiduciaryTest_Slide1_GlueBackground_1A.h5'
h5f_fname_w_path=os.path.join(h5f_path,h5f_fname)

maldi = MALDIdata()
maldi.log = True
#%%
#data = maldi.load_msi_from_hdf(h5f_fname_w_path)
#%%
pca_data = maldi.load_pca_from_hdf(h5f_fname_w_path)

print('Loaded %s'%(h5f_fname))
#%%
def _dump_pca_spectra(pca_data, save=False, rotation=0):
    maps = pca_data['ROI_01']['Maps']
    eigenvectors = pca_data['ROI_01']['Endmembers']
    eigenvalues = pca_data['ROI_01']['Eigenvalues']
    try:
        mz_bar = np.array(data['ROI_01']['mz_bar'])
    except:
        mz_bar = np.linspace(0,len(eigenvectors[0]),len(eigenvectors[0]))
    y, x, z = maps.shape
    if y > x:
        maps = np.rot90(maps)
    for i in np.flip(range(z),0):
        fig, ax = plt.subplots(2,1)
        image = maps[:,:,i]
        rotation_steps = rotation // 90
        for s in range(rotation_steps):
            image = np.rot90(image)
        ax[0].imshow(image, cmap='gray')
        ax[1].plot(mz_bar, eigenvectors[i])
        fig.suptitle('PCA Component %i'%(i+1))
        if save:
            plt.savefig('PCA Component %i'%(i+1))
            plt.close()
    scree = eigenvalues / sum(eigenvalues)
    running_scree = [sum(scree[:i]) for i in range(20)]
    plt.figure()
    plt.plot(range(20), scree)
    plt.plot(range(20), running_scree)
    plt.title('Scree plot')
    if save:
        plt.savefig('Scree.png')
        plt.close()
    

def _scatter_pca_spectra(pca_data, save=False):
    eigenvectors = pca_data['ROI_01']['Endmembers']
    n_comps = len(eigenvectors)
    for i in range(n_comps-1):
        plt.figure()
        plt.scatter(eigenvectors[i], eigenvectors[i+1])
        plt.xlabel('Principal Component %i'%(i+1))
        plt.ylabel('Principal Component %i'%(i+2))
        if save:
            plt.savefig('PCA %i vs %i'%((i+1),(i+2)))
            plt.close()

def _show_average_spectrum(h5f_fname_w_path, norm=None, save=False):
    hf = h5py.File(h5f_fname_w_path, 'r')
    average_spectrum = np.array(hf['Squared_data']['ROI_01']['Average spectrum'])
    mz_bar = np.array(hf['MALDI_001']['m_over_z'])
    plt.figure()
    plt.plot(mz_bar, average_spectrum)
    plt.title('Average Spectrum')
    plt.ylabel('Intensity')
    plt.xlabel('m/z')
    if save:
        plt.savefig('Average spectrum.png')
        plt.close()
        
def _find_nearest_member_(container, query):
    '''
    Finds the member of a container whose value is nearest to query. Returns
    index of nearest value within container. Intended to be used when 
    list.index(query) is, for whatever reason, not a viable option for locating
    the desired value within the container.
    
    Input:
    --------
    container : container variable (eg list, tuple, set, Numpy array)
        The container to be searched by the function
    query : number (eg int or float)
        Value to be searched for within container
    '''
    try:
        diffs = abs(container - query)
    except:
        diffs = []
        for entry in container:
            difference = entry - query
            diffs.append(abs(difference))
    minimum = min(diffs)
    try:
        mindex = list(diffs).index(minimum)
    except ValueError:
        #If no matching value can be found, return None object
        return None
    return mindex
        
def get_spectrum(h5f_fname_w_path, index, norm=''):
    hf = h5py.File(h5f_fname_w_path, 'r')
    raw_dataset = hf['MALDI_001']['Intensities']
    raw_spectrum = np.array(raw_dataset[index])
    if norm.upper() == '':
        pass
    else:
        pass
    hf.close()
    return raw_spectrum

def get_ion_map(h5f_fname_w_path, mz, tol=0.25, norm=None, tol_mode='rel'):
    hf = h5py.File(h5f_fname_w_path, 'r')
    raw_grp = hf['MALDI_001']
    square_grp = hf['Squared_data']['ROI_01']
    squared_indices = np.array(square_grp['Indices'])
    mz_bar = np.array(raw_grp['m_over_z'])
    mz_bounds = [mz-mz*tol/100, mz+mz*tol/100]
    index_bounds = [_find_nearest_member_(mz_bar,ion) for ion in mz_bounds]
    for index in index_bounds:
        if index == None:
            return None
    #Format index window to replicate np.where() functionality
    index_window = (np.array(range(index_bounds[0], index_bounds[1])),)
    window_length = len(index_window[0])
    raw_dset = raw_grp['Intensities']
    height, length = square_grp.attrs.get('Image Dimensions')
    ion_map = np.empty((height, length))
    if norm != None:
        norm = norm.capitalize()
    for i, row in enumerate(squared_indices):
        row_spectra = np.array([raw_dset[index] for index in row])
        row_intensities = row_spectra[:,index_bounds[0]:index_bounds[-1]+1]
        if norm != None:
            try:
                row_n_factors = np.array([raw_grp['Normalization Factors'][norm][index] for index in row])
                row_len = len(row_n_factors)
                row_n_intensities = np.array([row_intensities[i]/row_n_factors[i] for i in range(row_len)])
            except KeyError:
                print('Selected normalization mode has not been calculated.')
                return
        else:
            row_n_intensities = row_intensities
        ion_map[i] = np.sum(row_n_intensities, 1)
        print(i)
    return ion_map

def get_sum_map(h5f_fname_w_path, norm=None, save=False):
    hf = h5py.File(h5f_fname_w_path, 'r')
    raw_grp = hf['MALDI_001']
    raw_dset = raw_grp['Intensities']
    squared_indices = np.array(hf['Squared_data']['ROI_01']['Indices'])
    if norm != None:
        norm = norm.upper()
        n_factor_group = raw_grp['Normalization Factors']
        try:
            n_factors = np.array(n_factor_group[norm])
        except KeyError:
            print('Selected normalization mode has not been calculated.')
            return None
    summed_map = np.zeros_like(squared_indices, dtype='float32')
    for i, row in enumerate(squared_indices):
        row_raw_spectra = np.array([raw_dset[index] for index in row])
        if norm != None:
            row_n_factors = [n_factors[index] for index in row]
            row_len = len(row_n_factors)
            row_normalized_spectra = np.empty_like(row_raw_spectra)
            for j in range(row_len):
                row_normalized_spectra[j] = row_raw_spectra[j] / row_n_factors[j]
        else:
            row_normalized_spectra = row_raw_spectra
        row_pixels = np.sum(row_normalized_spectra,1)
        summed_map[i] = row_pixels
    hf.close()
    return summed_map

def list_norms(h5f_fullname):
    hf = h5py.File(h5f_fullname, 'r')
    n_modes = [None]
    try:
        n_factor_grp = hf['MALDI_001']['Normalization Factors']
    except:
        return None
    n_modes = [key for key in n_factor_grp.keys()]
    hf.close()
    return n_modes

def new_normalization(h5f_fname_w_path, norm=''):
    hf = h5py.File(h5f_fname_w_path, 'r+')
    n_factors = maldi._calculate_new_normalization_(norm,hf)
    hf.close()
    return n_factors

def calc_fast_norms(h5f_fname_w_path):
    hf = h5py.File(h5f_fname_w_path, 'r+')
    norm_modes = [
            'base peak',
            'median',
            'mean',
            ]
    

#%%
new_normalization(h5f_fname_w_path, 'fast')