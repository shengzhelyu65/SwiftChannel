import os

from Models.Estimator.estimator import ls_estimator
from Dataloader.dataloader import load_data_hdf5, get_training_samples_h_2_hS_symbol
from config_file.config import Config, FilesConfig
from Dataloader.dataloader import compress_signal
import h5py
import numpy as np
from tqdm import tqdm

def nr_ofdm_demodulate(rx_waveform_demo, n_subcarriers, n_symbols=1, length_cp=72, length_fft=1024):
    num_samples = rx_waveform_demo.shape[0]

    assert num_samples % (length_fft + length_cp) == 0, 'Invalid number of samples'
    rx_grid = np.zeros((n_subcarriers, n_symbols, rx_waveform_demo.shape[1]), dtype=complex)
    num_remaining_samples = length_fft - n_subcarriers
    
    for i in range(rx_waveform_demo.shape[1]):
        rx_waveform_demo_ant = rx_waveform_demo[length_cp:, i]
        fft_res = np.fft.fft(rx_waveform_demo_ant, length_fft)
        fft_res = np.fft.fftshift(fft_res)
        rx_grid[:, :, i] = fft_res[num_remaining_samples // 2:num_remaining_samples // 2 + n_subcarriers].reshape(n_subcarriers, n_symbols)
        
    if n_symbols == 1:
        return rx_grid[:, 0, :]

    return rx_grid

def ofdm_demodulate(rx_waveform, n_subcarriers=624, n_symbols=1, length_cp=72, length_fft=1024):
    (n_samples, length_fft_cp, _) = rx_waveform.shape
    if length_fft_cp == 1096:
        length_fft = 1024
        length_cp = 72
    elif length_fft_cp == 548:
        length_fft = 512
        length_cp = 36
    
    rx_grid_all = np.zeros((n_subcarriers, n_samples, rx_waveform.shape[2]), dtype=np.complex128)
    
    for i in range(n_samples):
        rx_waveform_demo = rx_waveform[i]
        rx_grid = nr_ofdm_demodulate(rx_waveform_demo, n_subcarriers, n_symbols, length_cp, length_fft)
        assert rx_grid.shape == (n_subcarriers, rx_waveform_demo.shape[1]), 'Invalid shape of rx_grid'
        rx_grid_all[:, i, :] = rx_grid
        
    return rx_grid_all

def main():
    config = Config()
    file_config = FilesConfig()
    
    ## Load data
    print("----- Loading data... -----")
    
    training_folder_path = file_config.train_folder_path
    
    config.frequency_compresion_ratio = 2
    config.spatial_compresion_ratio = 1
    
    ### Load the raw data from the training folder and apply the LS estimator to generate training samples
    hPerfect_key = file_config.hPerfect_key
    txGrid_key = file_config.txGrid_key
    rxWaveform_key = file_config.rxWaveform_key

    files = os.listdir(training_folder_path)

    hPerfect_files = [file for file in files if 'hestPerfect' in file]
    txGrid_files = [file for file in files if 'txGrid' in file]
    rxWaveform_files = [file for file in files if 'rxWaveform' in file]
    
    hPerfect_files = sorted(hPerfect_files)
    rxWaveform_files = sorted(rxWaveform_files)
    
    hPerfect_paths = [training_folder_path + file for file in hPerfect_files]
    txGrid_paths = [training_folder_path + file for file in txGrid_files]
    rxWaveform_paths = [training_folder_path + file for file in rxWaveform_files]
    
    txGrid = load_data_hdf5(txGrid_paths[0], txGrid_key)
    print("Shape of txGrid: ", txGrid.shape)
    
    for i in range(len(hPerfect_paths)):
        hPerfect = load_data_hdf5(hPerfect_paths[i], hPerfect_key)
        rxWaveform = load_data_hdf5(rxWaveform_paths[i], rxWaveform_key)
        rxGridDemo = ofdm_demodulate(rxWaveform, n_subcarriers=config.dim_frequency)
        
        print("Loaded data from: ", hPerfect_paths[i])
        print("Shape of hPerfect: ", hPerfect.shape)
        print("Shape of rxWaveform: ", rxWaveform.shape)
        print("Shape of rxGridDemo: ", rxGridDemo.shape)
        del rxWaveform
        
        # Estimation
        h_est_ls, h_perfect_srs, _, _, _ = ls_estimator(txGrid, rxGridDemo, hPerfect, config.spatial_compresion_ratio, config.frequency_compresion_ratio)    
        del hPerfect, rxGridDemo
        # Extract special slots
        h_estimated_train, h_perfect_srs_train, _ = get_training_samples_h_2_hS_symbol(h_est_ls, h_perfect_srs, None)
        del h_est_ls, h_perfect_srs
        
        # Save the np data
        sample_index = hPerfect_paths[i].split('/')[-1].split('_')[1]
        SNR_level = hPerfect_paths[i].split('/')[-1].split('_')[4]
        UEV = hPerfect_paths[i].split('/')[-1].split('_')[6]
        DEL = hPerfect_paths[i].split('/')[-1].split('_')[10].replace('.mat', '')
        FRE_COM = config.frequency_compresion_ratio
        SPA_COM = config.spatial_compresion_ratio
        file_name = f"train_{sample_index}_SNR_{SNR_level}_UEV_{UEV}_DEL_{DEL}_FRE_{FRE_COM}_SPA_{SPA_COM}"
        with h5py.File((training_folder_path + 'h_estimated_' + file_name + '.h5'), 'w') as hf:
            hf.create_dataset('h_estimated_trains', data=h_estimated_train, compression='gzip')
            
        with h5py.File((training_folder_path + 'h_perfect_srs_' + file_name + '.h5'), 'w') as hf:
            hf.create_dataset('h_perfect_srs_trains', data=h_perfect_srs_train, compression='gzip')
            
    ### Load the pre-processed samples and generate symbol-level training samples
    # List all HDF5 files in the folder
    files = os.listdir(training_folder_path)
    h_estimated_files = [file for file in files if 'h_estimated' in file and 'h5' in file and 'symbol_level' not in file and 'slot_level' not in file and 'FRE_2_SPA_1' in file]
    h_perfect_srs_files = [file for file in files if 'h_perfect_srs' in file and 'h5' in file and 'symbol_level' not in file and 'slot_level' not in file and 'FRE_2_SPA_1' in file]
    
    with h5py.File(os.path.join(training_folder_path, h_estimated_files[0]), 'r') as hf:
        h_estimated_train_tmp = hf['h_estimated_trains'][:]
        num_samples_per_file = h_estimated_train_tmp.shape[0]
        dim_frequency_compresed = h_estimated_train_tmp.shape[2]
        dim_antennas_compresed = h_estimated_train_tmp.shape[3]
        
    with h5py.File(os.path.join(training_folder_path, h_perfect_srs_files[0]), 'r') as hf:
        h_perfect_srs_train_tmp = hf['h_perfect_srs_trains'][:]
        dim_frequency = h_perfect_srs_train_tmp.shape[2]
        dim_antennas = h_perfect_srs_train_tmp.shape[3]
        
    del h_estimated_train_tmp, h_perfect_srs_train_tmp

    # Initialize a list to store the loaded datasets
    print("Number of samples per file: ", num_samples_per_file)
    num_samples = len(h_estimated_files) * num_samples_per_file
    print("Number of samples of all files: ", num_samples)
    h_estimated_trains = np.empty((num_samples, 2, dim_frequency_compresed, dim_antennas_compresed), dtype=np.float32)
    h_perfect_srs_trains = np.empty((num_samples, 2, dim_frequency, dim_antennas), dtype=np.float32)

    # Load each HDF5 file and store the dataset in the list
    for i in tqdm(range(len(h_estimated_files)), desc="Loading files"):
        with h5py.File(os.path.join(training_folder_path, h_estimated_files[i]), 'r') as hf:
            h_estimated_train = hf['h_estimated_trains'][:]
            if (i == 0):
                print("Shape of h_estimated_train: ", h_estimated_train.shape)
            h_estimated_trains[i*num_samples_per_file:(i+1)*num_samples_per_file] = np.array(h_estimated_train)
            
        with h5py.File(os.path.join(training_folder_path, h_perfect_srs_files[i]), 'r') as hf:
            h_perfect_srs_train = hf['h_perfect_srs_trains'][:]
            if (i == 0):
                print("Shape of h_perfect_srs_train: ", h_perfect_srs_train.shape)
            h_perfect_srs_trains[i*num_samples_per_file:(i+1)*num_samples_per_file] = np.array(h_perfect_srs_train)

    # Write the concatenated data into a new HDF5 file with compression
    with h5py.File(training_folder_path + 'h_estimated_trains_symbol_level.h5', 'w') as hf:
        hf.create_dataset('h_estimated_trains', data=h_estimated_trains, compression='gzip')
    print("Done saving h_estimated_trains_symbol_level.h5")
        
    with h5py.File(training_folder_path + 'h_perfect_srs_trains_symbol_level.h5', 'w') as hf:
        hf.create_dataset('h_perfect_srs_trains', data=h_perfect_srs_trains, compression='gzip')
    print("Done saving h_perfect_srs_trains_symbol_level.h5")
        
    ### Save into numpy file
    with h5py.File(training_folder_path + 'h_estimated_trains_symbol_level.h5', 'r') as hf:
        h_estimated_train = hf['h_estimated_trains'][:]
        
    h_estimated_train = np.array(h_estimated_train).astype(np.float32)
    config.frequency_compresion_ratio = 4
    config.spatial_compresion_ratio = 4
    h_estimated_trains = compress_signal(config, h_estimated_train)
    np.save(training_folder_path + f"h_estimated_trains_symbol_level_FRE_{config.frequency_compresion_ratio}_SPA_{config.spatial_compresion_ratio}.npy", h_estimated_trains)
        
    with h5py.File(training_folder_path + 'h_perfect_srs_trains_symbol_level.h5', 'r') as hf:
        h_perfect_srs_train = hf['h_perfect_srs_trains'][:]
        
    h_perfect_srs_train = np.array(h_perfect_srs_train).astype(np.float32)
    np.save(training_folder_path + 'h_perfect_srs_trains_symbol_level.npy', h_perfect_srs_train)

if __name__ == '__main__':
    main()