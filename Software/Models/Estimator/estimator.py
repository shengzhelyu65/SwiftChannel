import numpy as np

def ls_estimator(txGrid, rxGrid, h_perfect, spatial_compresion_ratio, frequency_compresion_ratio, output_full=False):
    print("------ LS Estimator ------")
    
    num_SRS_symbols = txGrid.shape[1]
    
    txGrid_non_zero = np.zeros((txGrid.shape[0]//frequency_compresion_ratio, num_SRS_symbols, txGrid.shape[2]), dtype=np.complex128)
    rxGrid_non_zero = np.zeros((rxGrid.shape[0]//frequency_compresion_ratio, num_SRS_symbols, rxGrid.shape[2]//spatial_compresion_ratio), dtype=np.complex128)
    H_est = np.zeros((txGrid.shape[0]//frequency_compresion_ratio, num_SRS_symbols, rxGrid.shape[2]//spatial_compresion_ratio, txGrid.shape[2]), dtype=np.complex128)
    H_perfect_srs = np.zeros((txGrid.shape[0], num_SRS_symbols, rxGrid.shape[2], txGrid.shape[2]), dtype=np.complex128)
    if output_full:
        H_perfect_full = np.zeros((txGrid.shape[0], num_SRS_symbols, rxGrid.shape[2], txGrid.shape[2]), dtype=np.complex128)
    else:
        H_perfect_full = None
          
    # Compress the txGrid, rxGrid for 3D together
    txGrid_non_zero = txGrid[::frequency_compresion_ratio, :, :]
    rxGrid_non_zero = rxGrid[::frequency_compresion_ratio, :, ::spatial_compresion_ratio]
    
    H_perfect_srs = h_perfect[:, :, :, :]
    if output_full:
        H_perfect_full = h_perfect[:, :, :, :]
    
    num_subcarriers_compressed, num_symbols_non_zero, num_tx = txGrid_non_zero.shape
    _, _, num_rx_compressed = rxGrid_non_zero.shape
    
    for subcarrier in range(num_subcarriers_compressed):
        for symbol in range(num_symbols_non_zero):
            for rx_ant in range(num_rx_compressed):
                for tx_ant in range(num_tx):
                    Y = rxGrid_non_zero[subcarrier, symbol, rx_ant]
                    X = txGrid_non_zero[subcarrier, symbol, tx_ant]

                    # LS estimation: H_est = Y / X for each transmit antenna
                    H_est[subcarrier, symbol, rx_ant, tx_ant] = Y / X
                
    print("Shape of estimated H by LS estimator: ", H_est.shape)
    print("Shape of perfect H with SRS patterns: ", H_perfect_srs.shape)
    if output_full:
        print("Shape of perfect H full matrix: ", H_perfect_full.shape)
    
    return H_est, H_perfect_srs, H_perfect_full, txGrid_non_zero, rxGrid_non_zero