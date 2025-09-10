from config_file.config import Config, FilesConfig
from Models.Estimator.estimator import ls_estimator
from Dataloader.dataloader import load_data_hdf5, get_testing_samples_h_2_hS_symbol
from Dataloader.dataloader import compress_signal
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

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

def plot_grid(grid1, grid2):
    grid1_0 = np.abs(grid1[:,0,:])
    grid2_0 = np.abs(grid2[:,0,:])
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(grid1_0), aspect='auto', cmap='hot')
    plt.colorbar()
    plt.title('Grid 1')
    
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(grid2_0), aspect='auto', cmap='hot')
    plt.colorbar()
    plt.title('Grid 2')
    plt.show()

if __name__ == '__main__':
    config = Config()
    file_config = FilesConfig()
    
    ## Load data
    print("----- Loading data... -----")
    
    testing_folder_path = file_config.test_folder_path
    
    SNR_levels = [6, 10, 14, 18, 22, 26, 30]
    UEVs = [5, 20, 40, 60, 80, 100, 120]
    
    ### Convert to h5py files
    config.frequency_compresion_ratio = 2
    config.spatial_compresion_ratio = 1
    
    hPerfect_key = file_config.hPerfect_key
    txGrid_key = file_config.txGrid_key
    rxWaveform_key = file_config.rxWaveform_key

    files = os.listdir(testing_folder_path)

    txGrid_files = [file for file in files if 'txGrid' in file]
    txGrid_paths = [testing_folder_path + file for file in txGrid_files]
    txGrid = load_data_hdf5(txGrid_paths[0], txGrid_key)
    print("Shape of txGrid: ", txGrid.shape)
    
    for SNR_level in SNR_levels:
        for UEV in UEVs:
            hPerfect_files_snr_uev = [file for file in files if 'hestPerfect' in file and f'SNR_{SNR_level}' in file and f'UEV_{UEV}' in file]
            rxWaveform_files_snr_uev = [file for file in files if 'rxWaveform' in file and f'SNR_{SNR_level}' in file and f'UEV_{UEV}' in file]
            
            hPerfect_files_snr_uev = sorted(hPerfect_files_snr_uev)
            rxWaveform_files_snr_uev = sorted(rxWaveform_files_snr_uev)
            
            hPerfect_paths = [testing_folder_path + file for file in hPerfect_files_snr_uev]
            rxWaveform_paths = [testing_folder_path + file for file in rxWaveform_files_snr_uev]
            
            print("SNR level: ", SNR_level)
            print("UEV: ", UEV)
            
            h_estimated_tests_all = []
            h_perfect_srs_tests_all = []
            
            for i in range(len(hPerfect_paths)):
                hPerfect = load_data_hdf5(hPerfect_paths[i], hPerfect_key)
                rxWaveform = load_data_hdf5(rxWaveform_paths[i], rxWaveform_key)
                rxGridDemo = ofdm_demodulate(rxWaveform, n_subcarriers=config.dim_frequency)
                
                print("Loaded data from: ", hPerfect_paths[i])
                print("Shape of hPerfect: ", hPerfect.shape)
                print("Shape of rxWaveform: ", rxWaveform.shape)
                print("Shape of rxGridDemo: ", rxGridDemo.shape)
                
                # Estimation
                h_est_ls, h_perfect_srs, h_perfect_full, txGrid_non_zero, rxGrid_non_zero = ls_estimator(txGrid, rxGridDemo, hPerfect, config.spatial_compresion_ratio, config.frequency_compresion_ratio)
                print("Shape of txGrid_non_zero: ", txGrid_non_zero.shape)
                print("Shape of rxGrid_non_zero: ", rxGrid_non_zero.shape)
                del hPerfect, rxGridDemo
                
                # Extract special slots
                h_estimated_test, h_perfect_srs_test, _, _, _ = get_testing_samples_h_2_hS_symbol(h_est_ls, h_perfect_srs, h_perfect_full, txGrid_non_zero, rxGrid_non_zero)
                del h_est_ls, h_perfect_srs, h_perfect_full, txGrid_non_zero, rxGrid_non_zero
                
                h_estimated_tests = np.array(h_estimated_test).astype(np.float32)
                h_perfect_srs_tests = np.array(h_perfect_srs_test).astype(np.float32)
                del h_estimated_test, h_perfect_srs_test
                
                h_estimated_tests_all.append(h_estimated_tests)
                h_perfect_srs_tests_all.append(h_perfect_srs_tests)
                
            # Save the data
            h_estimated_tests_all = np.array(h_estimated_tests_all).reshape(-1, h_estimated_tests.shape[1], h_estimated_tests.shape[2], h_estimated_tests.shape[3])
            h_perfect_srs_tests_all = np.array(h_perfect_srs_tests_all).reshape(-1, h_perfect_srs_tests.shape[1], h_perfect_srs_tests.shape[2], h_perfect_srs_tests.shape[3])
            
            with h5py.File((testing_folder_path + 'h_estimated_test_SNR_' + str(SNR_level) + '_UEV_' + str(UEV) + '.h5'), 'w') as hf:
                hf.create_dataset('h_estimated_tests', data=h_estimated_tests_all, compression='gzip')
                
            with h5py.File((testing_folder_path + 'h_perfect_srs_test_SNR_' + str(SNR_level) + '_UEV_' + str(UEV) + '.h5'), 'w') as hf:
                hf.create_dataset('h_perfect_srs_tests', data=h_perfect_srs_tests_all, compression='gzip')
                    
    print("----- Converted to h5py -----")
    print("All SNR levels and UEVs Done")