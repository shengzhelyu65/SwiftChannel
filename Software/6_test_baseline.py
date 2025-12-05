import csv
import os
import shutil
import sys
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use which GPU to test
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # ignore the information messages

from Baselines.model_zoo import return_model
from Metrics.metrics import NMSELoss
from config_file.config import FilesConfig, BaselineConfig, Config
from Dataloader.dataloader import compress_signal

import torch
from torch.utils.data import DataLoader, TensorDataset

import h5py
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def prepare_data(h_estimated_test, h_perfect_test, batch_size, num_workers):
    """
    Prepare the testing and validation data by randomly splitting the testing data.
    """

    test_dataset = TensorDataset(torch.from_numpy(h_estimated_test), torch.from_numpy(h_perfect_test))
    print("Shape of test_dataset: ", h_estimated_test.shape, h_perfect_test.shape)
    
    g = torch.Generator()
    g.manual_seed(0)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)

    return test_loader

def save_test_fig(h_est, h_perfect, output, index, save_filename):
    """
    h_est: Estimated channel matrix (batch_size, 2, 78, 8)
    h_perfect: Perfect channel matrix (batch_size, 2, 624, 32)
    output: Output of the model (batch_size, 2, 624, 32)
    """
    
    # Plot the estimated and perfect channel matrix
    h_est_sample = h_est[0].cpu().detach().numpy()
    h_perfect_sample = h_perfect[0].cpu().detach().numpy()
    output_sample = output[0].cpu().detach().numpy()
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(np.abs(h_est_sample[index]), cmap='jet', aspect='auto')
    plt.colorbar()
    plt.title('Input Estimated Channel Matrix (Real)')
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(output_sample[index]), cmap='jet', aspect='auto')
    plt.colorbar()
    plt.title('Output of the Model (Real)')
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(h_perfect_sample[index]), cmap='jet', aspect='auto')
    plt.colorbar()
    plt.title('Perfect Channel Matrix (Real)')
    
    # plt.show()
    plt.savefig(save_filename)
    plt.close()
    
def test_model(model, model_config, device, random_seed, test_loader, exp_name, save_img=False):
    test_loss_fn = NMSELoss()
    
    # Set random seed
    set_seed(random_seed)
    
    # Testing
    model.eval().to(device)
    running_loss = 0.0
        
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Testing', unit='batch') as pbar:
            for batch_idx, (h_est, h_perfect) in enumerate(test_loader):
                h_est = h_est.to(device)
                h_perfect = h_perfect.to(device)
                
                output = model(h_est)
                
                loss = test_loss_fn(output, h_perfect)
                running_loss += loss.item()
                
                # Update the progress bar
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
                
        if save_img:
            index = 0
            save_test_fig(h_est, h_perfect, output, index, save_filename=exp_name + f"/test_{index}.png")
            
    test_loss = running_loss / len(test_loader)
    test_loss_db = 10 * np.log10(test_loss)
    print(f"Test Loss: {test_loss:.6f} Test Loss (dB): {test_loss_db:.6f}")
    
    save_txt = exp_name + "/test_loss.txt"
    with open(save_txt, 'w') as f:
        f.write(f"Test Loss: {test_loss}\n")
        f.write(f"Test Loss (dB): {test_loss_db}")
        
    return test_loss, test_loss_db

def load_checkpoint(model, optimizer, scheduler, from_checkpoint):
    checkpoint = torch.load(from_checkpoint, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
    return_val = [model, epoch]
    if optimizer is not None:
        return_val.append(optimizer)
    if scheduler is not None:
        return_val.append(scheduler)
    return return_val

def main():
    config = Config()
    file_config = FilesConfig()
    
    ## Load the model
    ex_date = "0000"
    ex_time = "0000"
    ex_model = "Channelformer"
    ex_loss = "NMSE"
    fre_com = 4
    spa_com = 4
    
    # Load baseline-specific configuration
    model_config = BaselineConfig(ex_model)
    
    folder_path = file_config.test_folder_path

    for path in os.listdir("Experiments/"):
        if ex_date in path and ex_time in path and ex_model in path and ex_loss in path and f"FRE_{fre_com}" in path and f"SPA_{spa_com}" in path:
            ex_folder = "Experiments/" + path
            model_path = "Experiments/" + path + '/model_checkpoint_best.pth'
            break
        
    print("----- Model -----")
    print(f"Loading model: {ex_model}")
    model = return_model(ex_model)
    model.to(model_config.device)
        
    model, epoch = load_checkpoint(model, None, None, model_path)
    print(f"Loaded model from epoch {epoch}")
    
    files = os.listdir(folder_path)
    
    current_time = time.strftime("%m%d_%H%M")
    total_loss_txt = ex_folder + "/test_result/total_test_loss_tiny_" + current_time + ".csv"
    if not os.path.exists(ex_folder + "/test_result"):
        os.makedirs(ex_folder + "/test_result")
    with open(total_loss_txt, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["SNR level", "UEV", "Test Loss", "Test Loss (dB)"])
        
    shutil.copyfile('6_test_baseline.py', ex_folder + '/test_result/test_baseline_' + current_time + '.py')
    
    ## Testing
    SNR_levels = [6, 10, 14, 18, 22, 26, 30]
    UEVs = [5, 20, 40, 60, 80, 100, 120]
    
    nmse_sum = 0
    for SNR_level in SNR_levels:
        for UEV in UEVs:
            ex_folder_test = ex_folder + "/Test_SNR_" + str(SNR_level) + "_UEV_" + str(UEV)
            if not os.path.exists(ex_folder_test):
                os.makedirs(ex_folder_test)
        
            hEst_files = [file for file in files if 'h_estimated_test' in file and f'SNR_{SNR_level}' in file and f'UEV_{UEV}' in file]
            hPerfect_files = [file for file in files if 'h_perfect_srs_test' in file and f'SNR_{SNR_level}' in file and f'UEV_{UEV}' in file]
            with h5py.File(folder_path + hEst_files[0], 'r') as hf:
                h_estimated_tests = hf['h_estimated_tests'][:]
            with h5py.File(folder_path + hPerfect_files[0], 'r') as hf:
                h_perfect_srs_tests = hf['h_perfect_srs_tests'][:]
    
            # Frequency and spatial compresion
            h_estimated_tests = compress_signal(config, h_estimated_tests)
            print("Shape of h_estimated_tests: ", h_estimated_tests.shape)
            print("Shape of h_perfect_srs_tests: ", h_perfect_srs_tests.shape)

            ## Testing
            test_loader = prepare_data(h_estimated_tests, h_perfect_srs_tests, model_config.batch_size, model_config.num_workers)
            test_loss, test_loss_db = test_model(model, model_config, model_config.device, model_config.random_seed, test_loader, ex_folder_test, save_img=True)
            
            with open(total_loss_txt, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([SNR_level, UEV, test_loss, test_loss_db])
                
            nmse_sum += test_loss
            
    nmse_sum_avg = nmse_sum / 20
    print("Average NMSE: ", nmse_sum_avg)
    
    print("Testing completed.")
        
if __name__ == '__main__':
    main()