#MALDIdata

MALDIdata is a conversion codec for translation of imzML imaging mass spectrometry data files into HDF5 format.

## Installation

### Requirements
* Python 3.6
* numpy
* pyimzml
* h5py
* scipy

## Usage

To run in interactive shell:

    from MALDIdata import MALDIdata
    maldi = MALDIdata()
    maldi.write_msi_to_hdf(h5f_fname_w_path, 
    imzml_fname_w_path, normalize_mode)
    maldi.square_imzml(h5f_fname_w_path, 
    ROI_handling_mode)
    maldi.filter_imzml(h5f_fname_w_path, filter_mode)
    maldi.calc_pca(h5f_fname_w_path)
