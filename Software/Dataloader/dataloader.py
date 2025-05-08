import h5py
import numpy as np

def load_data_hdf5(file_path, key):
    with h5py.File(file_path, 'r') as f:
        data = f[key][:]
        
    data = data.view(np.complex128)
    data = transpose_data(data, data.ndim)
    return data

def transpose_data(data, num_dims):
    if num_dims == 3:
        return np.transpose(data, (2, 1, 0))
    elif num_dims == 4:
        return np.transpose(data, (3, 2, 1, 0))
    elif num_dims == 5:
        return np.transpose(data, (4, 3, 2, 1, 0))

def get_training_samples_h_2_hS_symbol(h_estimated, h_perfect_srs, h_perfect_full, output_full=False):
    print("------ Generate samples ------")
    
    num_samples = h_estimated.shape[1]
    
    h_estimated_train = np.zeros((h_estimated.shape[0], num_samples, h_estimated.shape[2], h_estimated.shape[3], 2), dtype=np.float32)
    h_perfect_srs_train = np.zeros((h_perfect_srs.shape[0], num_samples, h_perfect_srs.shape[2], h_perfect_srs.shape[3], 2), dtype=np.float32)
    if output_full:
        h_perfect_full_train = np.zeros((h_perfect_full.shape[0], num_samples, h_perfect_full.shape[2], h_perfect_full.shape[3], 2), dtype=np.float32)
    else:
        h_perfect_full_train = None

    h_estimated_train[:, :, :, :, 0] = np.real(h_estimated)
    h_estimated_train[:, :, :, :, 1] = np.imag(h_estimated)
    
    h_perfect_srs_train[:, :, :, :, 0] = np.real(h_perfect_srs)
    h_perfect_srs_train[:, :, :, :, 1] = np.imag(h_perfect_srs)
    
    if output_full:
        h_perfect_full_train[:, :, :, :, 0] = np.real(h_perfect_full)
        h_perfect_full_train[:, :, :, :, 1] = np.imag(h_perfect_full)
    
    h_estimated_train = np.transpose(h_estimated_train, (1, 4, 0, 2, 3)).astype(np.float32)
    h_perfect_srs_train = np.transpose(h_perfect_srs_train, (1, 4, 0, 2, 3)).astype(np.float32)
    
    h_estimated_train = np.reshape(h_estimated_train, (h_estimated_train.shape[0], h_estimated_train.shape[1], h_estimated_train.shape[2], h_estimated_train.shape[3]*h_estimated_train.shape[4])).astype(np.float32)
    h_perfect_srs_train = np.reshape(h_perfect_srs_train, (h_perfect_srs_train.shape[0], h_perfect_srs_train.shape[1], h_perfect_srs_train.shape[2], h_perfect_srs_train.shape[3]*h_perfect_srs_train.shape[4])).astype(np.float32)
    
    if output_full:
        h_perfect_full_train = np.transpose(h_perfect_full_train, (1, 4, 0, 2, 3)).astype(np.float32)
        h_perfect_full_train = np.reshape(h_perfect_full_train, (h_perfect_full_train.shape[0], h_perfect_full_train.shape[1], h_perfect_full_train.shape[2], h_perfect_full_train.shape[3]*h_perfect_full_train.shape[4])).astype(np.float32)
    
    print("Shape of h_estimated_train: ", h_estimated_train.shape)
    print("Shape of h_perfect_srs_train: ", h_perfect_srs_train.shape)
    if output_full:
        print("Shape of h_perfect_full_train: ", h_perfect_full_train.shape)

    return h_estimated_train, h_perfect_srs_train, h_perfect_full_train

def get_testing_samples_h_2_hS_symbol(h_estimated, h_perfect_srs, h_perfect_full, tx_grid, rx_grid, output_full=False):
    print("------ Generate samples ------")
    
    num_samples = h_estimated.shape[1]
    
    h_estimated_test = np.zeros((h_estimated.shape[0], num_samples, h_estimated.shape[2], h_estimated.shape[3], 2), dtype=np.float32)
    h_perfect_srs_test = np.zeros((h_perfect_srs.shape[0], num_samples, h_perfect_srs.shape[2], h_perfect_srs.shape[3], 2), dtype=np.float32)
    tx_grid_test = np.zeros((tx_grid.shape[0], num_samples, tx_grid.shape[2], 2), dtype=np.float32)
    rx_grid_test = np.zeros((rx_grid.shape[0], num_samples, rx_grid.shape[2], 2), dtype=np.float32)
    if output_full:
        h_perfect_full_test = np.zeros((h_perfect_full.shape[0], num_samples, h_perfect_full.shape[2], h_perfect_full.shape[3], 2), dtype=np.float32)
    else:
        h_perfect_full_test = None

    h_estimated_test[:, :, :, :, 0] = np.real(h_estimated)
    h_estimated_test[:, :, :, :, 1] = np.imag(h_estimated)
    
    h_perfect_srs_test[:, :, :, :, 0] = np.real(h_perfect_srs)
    h_perfect_srs_test[:, :, :, :, 1] = np.imag(h_perfect_srs)
    
    tx_grid_test[:, :, :, 0] = np.real(tx_grid)
    tx_grid_test[:, :, :, 1] = np.imag(tx_grid)
    
    rx_grid_test[:, :, :, 0] = np.real(rx_grid)
    rx_grid_test[:, :, :, 1] = np.imag(rx_grid)
    
    if output_full:
        h_perfect_full_test[:, :, :, :, 0] = np.real(h_perfect_full)
        h_perfect_full_test[:, :, :, :, 1] = np.imag(h_perfect_full)
    
    h_estimated_test = np.transpose(h_estimated_test, (1, 4, 0, 2, 3)).astype(np.float32)
    h_perfect_srs_test = np.transpose(h_perfect_srs_test, (1, 4, 0, 2, 3)).astype(np.float32)
    
    h_estimated_test = np.reshape(h_estimated_test, (h_estimated_test.shape[0], h_estimated_test.shape[1], h_estimated_test.shape[2], h_estimated_test.shape[3]*h_estimated_test.shape[4])).astype(np.float32)
    h_perfect_srs_test = np.reshape(h_perfect_srs_test, (h_perfect_srs_test.shape[0], h_perfect_srs_test.shape[1], h_perfect_srs_test.shape[2], h_perfect_srs_test.shape[3]*h_perfect_srs_test.shape[4])).astype(np.float32)
    tx_grid_test = np.transpose(tx_grid_test, (1, 3, 0, 2)).astype(np.float32)
    rx_grid_test = np.transpose(rx_grid_test, (1, 3, 0, 2)).astype(np.float32)
    
    if output_full:
        h_perfect_full_test = np.transpose(h_perfect_full_test, (1, 4, 0, 2, 3)).astype(np.float32)
        h_perfect_full_test = np.reshape(h_perfect_full_test, (h_perfect_full_test.shape[0], h_perfect_full_test.shape[1], h_perfect_full_test.shape[2], h_perfect_full_test.shape[3]*h_perfect_full_test.shape[4])).astype(np.float32)
    
    print("Shape of tx_grid_test: ", tx_grid_test.shape)
    print("Shape of rx_grid_test: ", rx_grid_test.shape)
    print("Shape of h_estimated_test: ", h_estimated_test.shape)
    print("Shape of h_perfect_srs_test: ", h_perfect_srs_test.shape)
    if output_full:
        print("Shape of h_perfect_full_test: ", h_perfect_full_test.shape)

    return h_estimated_test, h_perfect_srs_test, tx_grid_test, rx_grid_test, h_perfect_full_test

def compress_signal(config, h_estimated_trains):
    if config.frequency_compresion_ratio != 2:
        real_frequecy_compresion_ratio = config.frequency_compresion_ratio // 2
        h_estimated_trains = h_estimated_trains[:, :, ::real_frequecy_compresion_ratio, :]
    if config.spatial_compresion_ratio != 1:
        h_estimated_trains_tmp = np.zeros((h_estimated_trains.shape[0], h_estimated_trains.shape[1], h_estimated_trains.shape[2], h_estimated_trains.shape[3]//config.spatial_compresion_ratio), dtype=np.float32)
        for ant in range(h_estimated_trains.shape[3]//config.spatial_compresion_ratio//config.num_UE_antennas):
            h_estimated_trains_tmp[:, :, :, ant*config.num_UE_antennas:(ant+1)*config.num_UE_antennas] = \
                    h_estimated_trains[:, :, :, ant*config.spatial_compresion_ratio*config.num_UE_antennas:ant*config.spatial_compresion_ratio*config.num_UE_antennas + config.num_UE_antennas]
    else:
        h_estimated_trains_tmp = h_estimated_trains
    return h_estimated_trains_tmp