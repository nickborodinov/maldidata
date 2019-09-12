#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created JL2418 ML

@author: izl
"""

#%% Import modules
from pyimzml.ImzMLParser import ImzMLParser
import numpy as np
import h5py
from scipy import signal

def find_nearest_member(container, query):
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
                        
                    Output:
                    --------
                    mindex : int
                        Index of item in container whose value most nearly matches query
                    '''
                    try:
                        diffs = abs(container - query)
                    except:
                        diffs = []
                        for entry in container:
                            difference = entry - query
                            diffs.append(abs(difference))
                    minimum = min(diffs)
                    mindex = list(diffs).index(minimum)
                    return mindex
                
#%%
class MALDIdata(object):
    '''
    Class for handling imzML/ibd-exported MALDI data. Requires imaging mass 
    spectrometry data formatted as .imzML/.ibd file pairs.
    
    Contains methods for conversion of imzML/ibd files to HDF5 datafiles. 
    Contains additional methods for subsequent processing and manipulation of
    the resultant HDF5 datasets.
    '''
    def __init__(self):
        '''
        Initialize MALDIData instance.
        '''
        #Verbose mode is OFF by default
        self.log = False
        return
    
    def write_msi_to_hdf(self, h5f_fname_w_path, imzml_fname_w_path, norm=''):
        """
        Converts imzML data to HDF5 format. Iterates through imzML and, one 
        spectrum at a time, reads and writes spectral data to h5 as new raw
        dataset. Spectra are stored to HDF5 dataset in the order in which
        they are read from imzML file.
        """
        
        imzml = ImzMLParser(imzml_fname_w_path)
        #Infer dataset dimensions from first spectrum in imzML
        samp_mz, samp_int = imzml.getspectrum(0)
        length = len(samp_mz)
        self.mzsaxis = np.asarray(samp_mz)
        height = len(imzml.mzOffsets)
        del samp_int, samp_mz
        
        #Create and open hdf5 file
        hf = h5py.File(h5f_fname_w_path, 'w')
        print('\n%i spectra x %i mz data points'%(height, length))
        print()
        print('Importing...')
        print('')
        grp = hf.create_group('MALDI_001')

        dset1 = grp.create_dataset('Intensities', shape=(height, length),
                                   chunks=(1, 1024), dtype='float64')
        grp.create_dataset('m_over_z', data=self.mzsaxis)
        dset3 = grp.create_dataset('Coordinates', shape=(height,2))
        grp.create_group('Normalization Factors')
        
        # Iterates through spectra contained in .ibd binary using getspectrum()
        # For each spectrum, writes mz axis and intensity list as line in
        # appropriate datasets in group "MALDI_001"
        summed_ints = np.zeros(length)
        max_int = 0
        for i in range(height):
            mz, intensity = imzml.getspectrum(i)
            try:
                coordinates = imzml.get_physical_coordinates(i)
            except KeyError:
                imzml.imzmldict['pixel size y'] = imzml.imzmldict['pixel size x']
                coordinates = imzml.get_physical_coordinates(i)
            point = np.asarray([coordinates[0], coordinates[1]])
            dset1[i] = intensity
            dset3[i] = point
            if max(intensity) > max_int:
                max_int = max(intensity)
            summed_ints = summed_ints + intensity
            if self.log and i > 0:
                if i % 1000 == 0:
                    print('%i / %i'%((i, height)))
        average_spectrum = summed_ints / height
        grp.create_dataset('Average spectrum', data=(average_spectrum), dtype='float64')
        del summed_ints, mz, intensity, coordinates, point, max_int
        
        #Revert Numpy error handling to default setting (print)
        np.seterr(all = 'print')
        # Clean up, flush buffer, close file
        print('Finished importing!')
        if norm=='' or norm.upper()=='NONE':
            pass
        else:
            self._calculate_new_normalization_(norm, hf)
        hf.flush()
        hf.close()
        return

    def square_imzml(self, h5f_fname_w_path, ROI='auto'):
        """
        Crops spectra which do not fall inside rectangular window. Reads in all
        coordinates from plaintext spotlist, trims away border spectra so that
        remaining data forms rectangular profile for image generation.
        
        Input:
        --------
        h5f_fname_w_path: str
            Absolute filename of h5 file, with path
        ROI : str
            Keyword for selection of ROI handling mode (default ROI='auto')
        """
        
        hf = h5py.File(h5f_fname_w_path)
        raw = hf['MALDI_001']
        mz_bar = np.asarray(raw['m_over_z'])
        coordinates = np.asarray(hf['MALDI_001']['Coordinates'])
        try:
            squared = hf.create_group('Squared_data')
        except:
            del hf['Squared_data']
            squared = hf.create_group('Squared_data')
            
        # Crop dataset to rectangular area (ie. remove non-square edges)
        #Read in all (x,y) coordinates into separate lists
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        unique_xs = list(set(x))
        unique_ys = list(set(y))
        unique_xs.sort()
        unique_ys.sort()
        
        #Determine number of spots sampled along both x and y axes
        Dx = len(unique_xs)    #Integer number of spots sampled along x axis
        Dy = len(unique_ys)    #Integer number of spots sampled along y axis
        '''
        Spotlist coordinates are expressed as the distance of the sampled
        point, in micrometers, from the center of the target plate adapter.
        '''
        
        #Find extrema of coordinates in microns
        minx = min(unique_xs)   #Extreme left coordinate, in nanometers
        maxx = max(unique_xs)   #Extreme right coordinate, in nanometers
        miny = min(unique_ys)   #Extreme lower coordinate, in nanometers
        maxy = max(unique_ys)   #Extreme upper coordinate, in nanometers
        bxb = np.zeros((Dx, Dy), bool)
        bxi = np.full_like(bxb, -1, int)
        for idx, coordinate in enumerate(coordinates):
            (x, y) = coordinate
            x_index = unique_xs.index(x)
            y_index = unique_ys.index(y)
            bxb[x_index, y_index] = True
            bxi[x_index, y_index] = idx
            
        #Crop columns (crop along y axis)
        continue_x = True
        continue_y = True
        i=0
        j=0
        while continue_x==True or continue_y==True:
            if i==0 and j==0:
                test_array = bxb
                test_array_t = test_array.transpose()
            else:
                test_array = bxb[i:-1-i+1, i:-1-j+1]
                test_array_t = test_array.transpose()
            x_test = [all(column) for column in test_array_t]
            y_test = [all(row) for row in test_array]
            if all(x_test): continue_x = False
            if all(y_test): continue_y = False
            if continue_x: i+=1
            if continue_y: j+=1
        if i == 0 and j == 0:
            index_window = bxi
        elif i == 0 and j != 0:
            index_window = bxi[:, j:-j-1]
        elif i != 0 and j == 0:
            index_window = bxi[i:-i-1, :]
        else:
            index_window = bxi[i:-i-1, j:-j-1]
        ptsn = index_window.shape
        squared_spectra = index_window.flatten()
        (ptsx, ptsy) = ptsn
        self.indxs = squared_spectra
        sl_idx = 0
        
        #Attempt to create new datasets, groups, and associated attributes
        try:
            roi_grp = squared.create_group('ROI_%02i'%(sl_idx+1))
            mzs = roi_grp.create_dataset('m_over_z', data = raw['m_over_z'])
            ind = roi_grp.create_dataset('Indices', data=squared_spectra.reshape(ptsn))
            roi_grp.create_dataset('Coordinates', data=coordinates[squared_spectra])
            roi_grp.attrs.create('Image Dimensions', ptsn)
            #Fails if any dataset, group, or attribute exists, assumes process has
            #already been completed and reads in data from h5
        except:
            roi_grp = hf['Squared_data']
            mzs = roi_grp['m_over_z']
            ints = roi_grp['Intensities']
            return
            #Recalculate height to reflect cropped spectra
            
        print('Calculating average spectrum...')
        summed_ints = np.zeros_like(mz_bar)
        total_sum_map = np.zeros(shape=ptsn)
        #Iterate through passed indices enumerator
        indices = ind[:]
        for i, row in enumerate(indices):
            for j, value in enumerate(row):
                spectrum = raw['Intensities'][value][:]
                value = spectrum[0]
                if np.isnan(value): continue
                if value == -np.pi: continue
                summed_ints += spectrum
                total_sum_map[i,j] = np.sum(spectrum)
                
        raw.create_dataset('Sum map', data=total_sum_map)
        print()
        print('Squared %i spectra.'%(len(squared_spectra)))
        print('Squared shape: %i x %i'%(ptsn[0], ptsn[1]))
        
        # Calculate and write averaged spectrum to hdf5 file
        average_spectrum = summed_ints/(ptsn[0] * ptsn[1])
        roi_grp.create_dataset('Average spectrum', data=average_spectrum,
                               dtype='float32')
        hf.close()
        return

    def filter_imzml(self, h5f_fname_w_path, mode=''):
        """
        DEPRECATED
        Applies filter to input spectra, dumping m/z values which do not pass
        thresholding algorithm.

        Writes resultant filtered dataset as HDF5 dataset + metadata.
        """
        # Open hdf5 file for read/write
        roi_idx = 0
        hf = h5py.File(h5f_fname_w_path)
        squared = hf['Squared_data']
        raw_grp = hf['MALDI_001']
        mz_bar = np.asarray(raw_grp['m_over_z'])
        try:
            filtered = hf.create_group('Filtered_data')
        except:
            del hf['Filtered_data']
            filtered = hf.create_group('Filtered_data')
        print()
        print('Filtering...')
        print()
        roi_grp = squared
        if mode.upper() == 'THRESHOLD':
            cutoff = 0.001
            print('Filter mode = Threshold (%3d%% of base peak)'%(cutoff*100))
            square_indices = np.asarray(roi_grp['Indices']).flatten()
            height = len(square_indices)
            average_spectrum = np.asarray(roi_grp['Average spectrum'])
            cutoff_intensity = cutoff * np.max(average_spectrum)
            passed_indices = np.where(average_spectrum > cutoff_intensity)
            print('%i peaks passed threshold'%(len(passed_indices[0])))
            length = len(passed_indices[0])
            filtered_data = np.empty(shape = (height, length))
            print('Writing filtered spectra.')
            for idx, index in enumerate(square_indices):
                spectrum = raw_grp['Intensities'][index]
                if not np.isnan(spectrum[0]):
                    filtered_data[idx] = spectrum[passed_indices]
                else: pass
            groupName = 'ROI_%02i'%(roi_idx+1)
            try:
                roi_filtered = filtered.create_group(groupName)
                ints = roi_filtered.create_dataset('Intensities', data=filtered_data,
                                               chunks=(1, length), dtype='float64')
                indices = roi_filtered.create_dataset('Indices', data=passed_indices[0])
                roi_filtered.create_dataset('Coordinates', data=np.asarray(squared[groupName]['Coordinates']))
                roi_filtered.create_dataset('mz_bar', data=mz_bar)
            except:
                roi_filtered = filtered['ROI_%02i'%(roi_idx+1)]
                ints = roi_filtered['Intensities']
                print('Group "Filtered_data" already exists as', filtered, '!')
                print()
        elif mode.upper() == 'LOD':
            print('Filter mode = MDL-Average')
            average_spectrum = roi_grp['Average spectrum']
            savgol_window_width = round((15/115096)*len(average_spectrum))
            if savgol_window_width <= 4:
                savgol_window_width = 5
            if savgol_window_width%2 == 0:
                savgol_window_width += 1
            filtered_average = signal.savgol_filter(average_spectrum, savgol_window_width, 4)
            noise_waveform = average_spectrum - filtered_average
            method_detection_limit = np.mean(average_spectrum) + 3 * np.std(noise_waveform)
            passed_peak_indices = np.where(average_spectrum > method_detection_limit)[0]
            square_indices = np.asarray(roi_grp['Indices']).flatten()
            height = len(square_indices)
            length = len(passed_peak_indices)
            filtered_data = np.empty(shape=(height, length))
            for idx, index in enumerate(square_indices):
                spectrum = raw_grp['Intensities'][index]
                if not np.isnan(spectrum[0]):
                    filtered_data[idx] = spectrum[passed_peak_indices]
                else: pass
            groupName = 'ROI_%02i'%(roi_idx+1)
            try:
                roi_filtered = filtered.create_group(groupName)
                ints = roi_filtered.create_dataset('Intensities', data=filtered_data,
                                               chunks=(1, length), dtype='float64')
                indices = roi_filtered.create_dataset('Indices', data=passed_peak_indices)
                roi_filtered.create_dataset('Coordinates', data=np.asarray(squared[groupName]['Coordinates']))
                roi_filtered.create_dataset('mz_bar', data=mz_bar)
                roi_filtered.attrs.create('Mode', [ord(c) for c in 'MDL-Average'])
                roi_filtered.attrs.create('MDL', method_detection_limit)
            except:
                roi_filtered = filtered['ROI_%02i'%(roi_idx+1)]
                ints = roi_filtered['Intensities']
                print('Group "Filtered_data" already exists as', filtered, '!')
                print()
        elif mode.upper() == 'NONE' or mode.upper() == '':
            print('No filter selected, filter not applied!')
            groupName = 'ROI_%02i'%(roi_idx+1)
            try:
                roi_filtered = filtered.create_group(groupName)
                roi_filtered.attrs.create('Filter Mode', [ord(c) for c in mode])
            except:
                roi_filtered = hf['ROI_%02i'%(roi_idx+1)]
                print('Group "Filtered_data" already exists as', filtered, '!')
                print()
        image_dims = squared[groupName].attrs.get('Image Dimensions')
        roi_filtered.attrs.create('Image Dimensions', image_dims)
        hf.close()
        return

    

    def calc_pca(self, h5_fname, n_comps=20, batch_size=1000, norm=''):
        """
        Perform Incremental Principle Component Analysis on squared data
        stored in the provided h5 datafile. Breaks dataset into sets of 1000
        spectra, which are fed to the IPCA model one at a time for training.
        Spectra are then similarly transformed according to results of the PCA
        in sets of 1000 at a time, then aggregated and stored to a new group
        in the same h5 datafile as the source dataset.
        """
        from sklearn.decomposition import PCA, IncrementalPCA
        hf = h5py.File(h5_fname, 'r+')
        raw_data = hf['MALDI_001']['Intensities']
        filtered = hf['Filtered_data']
        height, length = raw_data.shape
        
        #Apply relevant normalization
        norm = norm.capitalize()
        if norm == '':
            norm_factors = np.ones(height)
            norm = None
        else:
            try:
                norm_factors = np.array(hf['MALDI_001']['Normalization Factors'][norm])
            except KeyError:
                print('Desired normalization factors not yet calculated.')
                print('Calculate normalization factors, then try again.')
                return
        try:
            pca = hf.create_group('PCA')
        except:
            pca = hf['PCA']
            try:
                norm_group = pca.create_group(norm)
            except KeyError:
                print('Selected PCA has already been calculated, aborting.')
                return
        roi_list = list(filtered.keys())
        
        print('Calculating PCA')
        for roi in roi_list:
            roi_grp = pca.create_group(roi)
            if norm == None:
                norm_grp = roi_grp.create_group('Unnormalized')
            else:
                try:
                    norm_group = roi_grp.create_group(norm)
                except KeyError:
                    print('Selected PCA has already been calculated, aborting.')
                    return
            (ptsx, ptsy) = filtered[roi].attrs.get('Image Dimensions')
            
            # Perform PCA, store results as MALDIdata attributes
            model = IncrementalPCA(n_components=n_comps, batch_size=batch_size)
            steps = int(height/batch_size)
            print('Training PCA....')
            for i in range(steps):
                low_bound = i*1000
                high_bound = (i+1)*1000
                subset = raw_data[low_bound:high_bound]
                norm_factor_subset = norm_factors[low_bound:high_bound]
                normalized_subset = np.empty_like(subset)
                for j, spectrum in enumerate(subset):
                    n_factor = norm_factor_subset[j]
                    spectrum = subset[j]
                    norm_spectrum = spectrum / n_factor
                    normalized_subset[j] = norm_spectrum
                model.partial_fit(normalized_subset)
                print(high_bound)
            if steps == 0:
                high_bound = 0
            subset = raw_data[high_bound:]
            model.partial_fit(subset)
            fitted = np.empty((height, n_comps))
            
            print('Mapping spectra to PCA...')
            for i in range(steps):
                low_bound = i*1000
                high_bound = (i+1)*1000
                subset = raw_data[low_bound:high_bound]
                norm_factor_subset = norm_factors[low_bound:high_bound]
                normalized_subset = np.empty_like(subset)
                for j, spectrum in enumerate(subset):
                    n_factor = norm_factor_subset[j]
                    spectrum = subset[j]
                    norm_spectrum = spectrum / n_factor
                    normalized_subset[j] = norm_spectrum
                transformation = model.transform(normalized_subset)
                fitted[low_bound:high_bound] = transformation
                print(high_bound)
            fitted[high_bound:] = model.transform(raw_data[high_bound:])
            
            squared_indices = np.array(hf['Squared_data']['ROI_01']['Indices'])
            keep_indices = squared_indices.flatten()
            
            pca_maps = np.empty((ptsx, ptsy, n_comps))
            for index in keep_indices:
                spectrum = fitted[index]
                coordinates = np.where(squared_indices == index)
                row, column = (coordinates[0][0], coordinates[1][0])
                pca_maps[row, column, :] = spectrum

            roi_grp.create_dataset('Maps', data=pca_maps, dtype='float64')
            roi_grp.create_dataset('Endmembers', data=model.components_, dtype='float64')
            roi_grp.create_dataset('Eigenvalues', data=model.explained_variance_, dtype='float64')
            roi_grp.attrs['Number_of_components'] = n_comps
        hf.close()
        return
    
    def write_pca_to_hdf(self, h5f_fname_w_path):
        """
        #!!! DEPRECATED, FUNCTIONALITY INTEGRATED INTO calc_pca() METHOD
        
        Writes results of most recent call of calc_pca to new subgroup
        in raw data group of HDF5 file.
        """
        # Open HDF file
        print('Writing PCA to h5')
        self.h5f = h5py.File(h5f_fname_w_path)
        # Checks for preexisting PCA data
        conv_name = 'MALDI_001'
        m = 1
        for name in self.h5f[conv_name].keys():
            parts = name.split('_')
            if parts[0] == 'PCA' and int(parts[1]) >= m:
                m = int(parts[1])+1
        # Create new group "PCA_n" as subgroup of raw data, write metadata
        grp_name = 'PCA_' +str(m).zfill(3)
        grp = self.h5f[conv_name].create_group(grp_name)
        n_comps = self.pca_spectra.shape[0]
        grp.attrs['Number_of_components'] = n_comps
        # Write PCA data as new datasets in PCA group
        grp.create_dataset('Maps', data=self.pca_maps)
        grp.create_dataset('Endmembers', data=self.pca_spectra)
        grp.create_dataset('Eigenvalues', data=self.eigenvalues)
        self.analysis_name = grp_name
        self.h5f.flush()
        self.h5f.close()
    def load_msi_from_hdf(self, h5f_path, ROI=0):
        """
        Loads previously processed MSI data from HDF5 file into MALDIdata object.
        """
        try:
            hf = h5py.File(h5f_path, 'r')
        except:
            print(f'{h5f_path} could not be located.')
            return None
        filtered_grp = hf['Filtered_data']
        data_bin = {}
        if ROI > 0:
            roi_group_name = 'ROI_%02d'%(ROI)
            roi = filtered_grp[roi_group_name]
            roi_list = [roi]
        elif ROI == 0:
            roi_list = []
            roi_group_name_list = list(filtered_grp.keys())
            for group_name in roi_group_name_list:
                roi_group = filtered_grp[group_name]
                roi_list.append(roi_group)
        for roi in roi_list:
            (ptsx, ptsy) = roi.attrs.get('Image Dimensions')
            try:
                data = np.asarray(roi['Intensities']).reshape(ptsx, ptsy, -1)
                indices = np.asarray(roi['Indices'])
                mz_bar = np.asarray(roi['mz_bar'])
            except:
                data = None
                indices = None
                mz_bar = np.array(hf['MALDI_001']['m_over_z'])
            entry_data = {'intensity'   : data,
                          'indices'     : indices,
                          'mz_bar'      : mz_bar}
            try:
                n_factors = {}
                n_factor_group = hf['MALDI_001']['Normalization Factors']
                for key in n_factor_group.keys():
                    n_factors[key] = np.array(n_factor_group[key])
                entry_data['n_factors'] = n_factors
            except:
                pass
            data_bin[roi.name[-6:]] = entry_data
        hf.close()
        return data_bin

    def load_pca_from_hdf(self, h5_path, ROI=0):
        """
        Loads results of previously performed PCA from HDF5 file into MALDIdata object.
        """
        h5f = h5py.File(h5_path)
        pca_group = h5f['PCA']
        if ROI > 0:
            #Build 1-member ROI list
            roi_group_name_list = ['ROI_%02i'%(ROI)]
        elif ROI == 0:
            #DEFAULT BEHAVIOR
            #Build ROI list from all available ROI groups
            roi_group_name_list = list(pca_group.keys())
        else:
            print('No PCA data could be located!')
            return None
        data_bin = {}
        for roi_group_name in roi_group_name_list:
            roi_group = pca_group[roi_group_name]
            roi_data = {
                    'Maps'          :np.asarray(roi_group['Maps']),
                    'Endmembers'    :np.asarray(roi_group['Endmembers']),
                    'Eigenvalues'   :np.asarray(roi_group['Eigenvalues'])
                    }
            data_bin[roi_group_name] = roi_data
        return data_bin

    def copy_hdf_data(self, source, destination, recursive=False):
        """
        Helper function for copying hdf groups.

        Expects open hdf files or groups as destination and source.
        Copies groups and datasets in source to destination.
        Optional recursive functionality.
        """
        for item in source.items():
            if isinstance(item[1], h5py.Dataset):
                destination.create_dataset(item[0], data=source[item[0]])
            elif isinstance(item[1], h5py.Group):
                grp = destination.create_group(item[0])
                group_name = grp.name[1:]
                if recursive:
                    self.copy_hdf_data(source[group_name], destination[group_name])
            else:
                print('Could not copy'+str(item[0]))
        return
    
    @staticmethod
    def _compress_dataset_(dataset, parent_group, compression_level=4):
        '''
        Method creates a compressed version of the core raw dataset which is
        much faster to navigate for ion image generation. Stores compressed
        dataset to user-assigned group.
        
        Inputs:
        --------
            dataset: h5py Dataset object
                dataset to be compressed
            parent_group: h5py Group object
                group into which compressed dataset will be placed
        '''
        dataset_name = dataset.name.split('/')[-1]
        dataset_name = dataset_name+'_compressed'
        parent_group.create_dataset(dataset_name, shape=dataset.shape,
                                    dtype=dataset.dtype, chunks=True,
                                    compression='gzip',
                                    compresison_opts=compression_level)
        return
    
    def _calculate_new_normalization_(self, normalize_mode, hf=None):
        '''
        Calculate normalization factors for a normalization mode which has
        not been previously applied to dataset. Normalization factors are stored
        in subgroup alongside raw spectra, then applied in situ as needed. This
        allows for rapid reapplication of previously calculated normalization
        modes, and dynamic calculation of new modes long after initial dataset
        conversion.
        
        Input:
        --------
        normalize_mode : str
            Normalization mode to be calculated. See docs for keyword choices.
        hf : h5py.File object
            HDF5 datafile object containing dataset to be normalized (default hf=None)
        '''
        normalize_mode = normalize_mode.upper()
        if hf==None:
            print('No HDF dataset specified, ABORTING NORMALIZATION!')
            return
        elif normalize_mode == '':
            print('No normalization mode specified, ABORTING NORMALIZATION!')
            return
        #Check for preexisting normalization calculation results
        try:
            raw_grp = hf['MALDI_001']
            raw_dset = raw_grp['Intensities']
            n_factor_grp = raw_grp['Normalization Factors']
        except:
            raw_grp = hf['MALDI_001']
            raw_dset = raw_grp['Intensities']
            n_factor_grp = raw_grp.create_group('Normalization Factors')
        height, length = raw_dset.shape
        #Check whether desired normalization mode has already been applied to dataset
        if normalize_mode != 'FAST':
            try:
                normal_dset = n_factor_grp.create_dataset(normalize_mode.capitalize(), shape=(height,))
            except:
                print('Selected normalization mode has already been calculated for')
                print('the specified dataset.')
                while True:
                    recalculate_input = input('Recalculate normalization factors? [N/y]')
                    recalculate_input = recalculate_input.upper()
                    if recalculate_input == 'N' or recalculate_input == '':
                        print('ABORTING NORMALIZATION!')
                        return
                    elif recalculate_input == 'Y':
                        normal_dset = n_factor_grp[normalize_mode.capitalize()]
                        break
                    else:
                        print('Invalid input, try again.')
            #Set Numpy to raise an exception for any encountered runtime warnings
            np.seterr(all = 'raise')
            #Read and set various initialization parameters
            normalization_factors = np.ones(height)
            #Store normalization mode to h5 file as attribute
                #normalize_mode kwd is string, but h5 attributes cannot be string
                #First convert to ASCII values, store as list of ASCII values for reconstruction on read
            normal_dset.attrs.create('Mode', [ord(c) for c in normalize_mode])
            if normalize_mode == '' or normalize_mode.upper()=='NONE':
                print('No normalization mode selected, ABORTING NORMALIZATION!')
                return
            elif normalize_mode == 'BASE PEAK':
                #Internally normalize each spectrum by base peak intensity
                #Sets base peak intensity to 1.0, and calculates other peaks
                    #intensities as ratio of peak_intensity:base_peak_intensity
                print('Normalizing data...')
                print('Normalization mode = Base Peak')
                for i, spectrum in enumerate(raw_dset):
                    normalization_factors[i] = max(spectrum)
                    self._progress_counter_(i,1000,height)
            elif normalize_mode == 'SNR':
                #Applies a signal-to-noise ratio normalization.
                #Calculates standard deviation of spectrum intensities, sets as normalization factor
                print('Normalizing data...')
                print('Normalization mode = SNR')
                for idx, spectrum in enumerate(raw_dset):
                    normalization_factors[idx] = np.std(spectrum)
                    self._progress_counter_(idx,1000,height)
            elif normalize_mode == 'MEAN':
                #Normalize according to mean spectrum intensity
                #Attempt to replicate Bruker "TIC" normalization method
                print('Normalizing data...')
                print('Normalization mode = Mean')
                for idx, spectrum in enumerate(raw_dset):
                    normalization_factors[idx] = np.mean(spectrum)
                    self._progress_counter_(idx,1000,height)
    #        elif normalize_mode == 'TIC':
    #            #Scale peak intensities as a multiple of total ion count
    #            #Identical to mean signal intensity normalization, except for a
    #                #uniform scaling factor (in this case, 1.0)
    #            print('Normalizing data...')
    #            print('Normalization mode = TIC; full dataset')
    #            (height, length) = dset1.shape
    #            blocks = height//1000
    #            remainder = height%1000
    #            for i in range(blocks):
    #                start = i*1000
    #                end = ((i+1)*1000)
    #                spectrum_block = dset1[start:end]
    #                normalized_block = np.empty_like(spectrum_block)
    #                for idx, spectrum in enumerate(spectrum_block):
    #                    TIC = np.sum(spectrum)
    #                    try:
    #                        normalized_block[idx] = spectrum / TIC
    #                    except FloatingPointError:
    #                        normalized_block[idx] = np.nan
    #                dset1[start:end] = normalized_block
    #                if self.log:
    #                    completion = int((i*1000)/height * 100)
    #                    print(f'{completion}% complete')
    #            remainder_block = dset1[blocks*1000:]
    #            normalized_block = np.empty_like(remainder_block)
    #            for idx, spectrum in enumerate(remainder_block):
    #                TIC = np.sum(spectrum)
    #                normalized_block[idx] = spectrum / TIC
    #            dset1[blocks*1000:] = normalized_block
            elif normalize_mode == 'RMS':
                #Noramlize accoring to spectrum root mean squared (RMS) intensity
                # Calculates RMS intensity of each spectrum. RMS is set to 1.0,
                    #other peaks reported as ratio of orginal_intensity:RMS
                print('Normalizing data...')
                print('Normalization mode = RMS; full dataset')
                scale_factor = 1.0
                for idx, spectrum in enumerate(raw_dset):
                    S = np.sum(spectrum**2)
                    normalization_factors[idx] = np.sqrt(S/len(spectrum))
                    self._progress_counter_(idx,1000,height)
            elif normalize_mode == 'MEDIAN':
                #Force all pixels to have same median TIC; peak intensities are
                    #ratio of intensity:median_intensity
                print('Normalizing data...')
                print('Normalization mode = Median; full dataset')
                for idx, spectrum in enumerate(raw_dset):
                    #Replace any zeroes with 1e-16
                    #Prevents divide-by-zero error in normalization application
                    nonzero_indices = np.where(spectrum!=0)
                    nonzero_intensities = spectrum[nonzero_indices]
                    try:
                        normalization_factors[idx] = np.median(nonzero_intensities)
                    except:
                        print('Error processing spectrum %i'%(idx))
                        normalization_factors[idx] = -np.pi
                    self._progress_counter_(idx,1000,height)
        elif normalize_mode == 'FAST':
            '''
            Fast normalization mode performs the following normalization calculations
            simultaneously, rather than as separated operations:
                Base peak
                SNR
                Mean
                RMS
                Median
            It is called "fast" normalization because it performs all of these
            calculations in roughly the same amount of time that any one mode
            would require on its own by reducing the number of disc I/O operations
            required. As each spectrum is read-in from the file sequentially,
            normalization factors for each of the fast modes are calcualted for that
            spectrum and stored to memory. All normalization factors are then written
            to disk at the end of the function.
            '''
            normalization_factors = {
                    'BASE PEAK':np.ones(height),
                    'SNR':np.ones(height),
                    'MEAN':np.ones(height),
                    'RMS':np.ones(height),
                    'MEDIAN':np.ones(height),
                    }
            block_size = 1000
            block_count = height // block_size
            if height < block_size:
                upper_bound = 0
            else:
                for j in range(block_count):
                    lower_bound = j * block_size
                    upper_bound = j * block_size + block_size
                    block = np.array(raw_dset[lower_bound:upper_bound])
                    for i, spectrum in enumerate(block):
                        index = lower_bound + i
                        normalization_factors['BASE PEAK'][index] = np.max(spectrum)
                        normalization_factors['SNR'][index] = np.std(spectrum)
                        normalization_factors['MEAN'][index] = np.mean(spectrum)
                        S = np.sum(spectrum**2)
                        normalization_factors['RMS'][index] = np.sqrt(S/len(spectrum))
                        nonzero_indices = np.where(spectrum!=0)
                        nonzero_intensities = spectrum[nonzero_indices]
                        try:
                            normalization_factors['MEDIAN'][index] = np.median(nonzero_intensities)
                        except:
                            print('Error processing spectrum %i'%(idx))
                            print('Filling spectrum with -pi')
                            normalization_factors['RMS'][index] = -np.pi
                    self._progress_counter_((j+1)*1000,block_size,height)
            block = raw_dset[upper_bound:]
            for i, spectrum in enumerate(block):
                index = upper_bound + i
                normalization_factors['BASE PEAK'][index] = np.max(spectrum)
                normalization_factors['SNR'][index] = np.std(spectrum)
                normalization_factors['MEAN'][index] = np.mean(spectrum)
                S = np.sum(spectrum**2)
                normalization_factors['RMS'][index] = np.sqrt(S/len(spectrum))
                nonzero_indices = np.where(spectrum!=0)
                nonzero_intensities = spectrum[nonzero_indices]
                try:
                    normalization_factors['MEDIAN'][index] = np.median(nonzero_intensities)
                except:
                    print('Error processing spectrum %i'%(idx))
                    print('Filling spectrum with -pi')
                    normalization_factors['RMS'][index] = -np.pi
            for key, value in normalization_factors.items():
                try:
                    n_factor_grp.create_dataset(key, data=value)
                except RuntimeError:
                    print('Overwrite saved data for mode: %s [Y/n]?'%(key))
                    decision = input('')
                    decision = decision.upper()
                    if decision == 'Y' or decision == '':
                        n_factor_grp[key][:] = value
                    else:
                        pass
            return
            
        else:
            print('Unrecognized normalization mode!')
            print('Data not normalized!')
        
        return normalization_factors
    
    def _progress_counter_(self, i, interval, end):
        if self.log:
            if abs(i) > 0 and i % interval == 0:
                print('%i / %i'%(i,end))
        return