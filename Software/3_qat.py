import os

import torch.ao.quantization

from Models.swiftchannel_student_rp import SwiftChannelRP
from Models.swiftchannel_student_quan import SwiftChannelQuan

from Metrics.metrics import NMSELoss
from config_file.config import FilesConfig, ModelConfig

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import time
import numpy as np
import random
import warnings
from tqdm import tqdm

warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='ignore',
    category=FutureWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensures deterministic algorithms are used
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_data(h_estimated_train, h_perfect_train, batch_size, num_workers):
    """
    Prepare the training and validation data by randomly splitting the training data.
    """
    
    train_valid_percentage = (0.8, 0.2)
    
    # Split the data into training and validation
    train_size = int(train_valid_percentage[0] * len(h_estimated_train))
    valid_size = len(h_estimated_train) - train_size

    train_dataset, valid_dataset = torch.utils.data.random_split(TensorDataset(torch.tensor(h_estimated_train), torch.tensor(h_perfect_train)), [train_size, valid_size])
    print("Train dataset size: ", len(train_dataset))
    print("Validation dataset size: ", len(valid_dataset))
    
    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    
    return train_loader, valid_loader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def setup_optimizer(model, model_config, learning_rate):
    if model_config.optimizer == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    # Add more optimizers if needed
    raise ValueError(f"Unsupported optimizer: {model_config.optimizer}")

def lr_lambda(epoch):
    decay_start_from = 20

    if epoch < decay_start_from:
        return 1.0
    else:
        return 0.7 ** ((epoch - decay_start_from) // 10)
    
def setup_scheduler(optimizer, model_config):
    if model_config.scheduler == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=model_config.lr_step_size, gamma=model_config.lr_gamma)
    elif model_config.scheduler == 'LambdaLR':
        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    raise ValueError(f"Unsupported scheduler: {model_config.scheduler}")
  
def setup_loss_fn(model_config):
    if model_config.train_loss == 'MSE':
        return nn.MSELoss(), NMSELoss()
    elif model_config.train_loss == 'NMSE':
        return NMSELoss(), NMSELoss()
    elif model_config.train_loss == 'L1':
        return nn.L1Loss(), NMSELoss()
    raise ValueError(f"Unsupported loss function: {model_config.train_loss}")

def save_txt(loss_record_train, loss_record_valid, learning_rate_record, exp_name):
    # save loss and learning rate to txt file
    with open(f"{exp_name}/loss_lr_record.txt", 'w') as f:
        epoch = 0
        
        for train_loss, valid_loss, lr in zip(loss_record_train, loss_record_valid, learning_rate_record):
            train_loss_dB = 10 * np.log10(train_loss)
            valid_loss_dB = 10 * np.log10(valid_loss)
            f.write(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Train Loss(dB): {train_loss_dB:.4f}, Valid Loss(dB): {valid_loss_dB:.4f}, Learning Rate: {lr:.6f}\n")
            epoch += 1
            
    # save into npy file
    total_loss = np.stack((loss_record_train, loss_record_valid, learning_rate_record), axis=1)
    np.save(f"{exp_name}/loss_lr_record.npy", total_loss)
    
def train_quan_model(model, model_config, num_epochs, random_seed, train_loader, valid_loader, exp_folder, learning_rate):
    epoch = 0
    best_valid_loss = float('inf')
    no_improve_epoch = 0
    loss_record_train, loss_record_valid, learning_rate_record = [], [], []

    # Setup optimizer, loss function, and scheduler
    optimizer = setup_optimizer(model, model_config, learning_rate=learning_rate)
    loss_fn, nmse_loss_fn = setup_loss_fn(model_config)
    scheduler = setup_scheduler(optimizer, model_config)
    
    # Set random seed
    set_seed(random_seed)

    # Training loop
    while epoch < num_epochs:
        model.train().to('cpu')
        running_loss = 0.0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit='batch') as pbar:
            for batch_idx, (h_est, h_perfect) in enumerate(train_loader):
                h_est = h_est.to('cpu')
                h_perfect = h_perfect.to('cpu')
                
                optimizer.zero_grad()
                
                output = model(h_est)
                
                loss = nmse_loss_fn(output, h_perfect)
                loss.backward()
                running_loss += loss.item()
                
                optimizer.step()
                
                # Update the progress bar
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
                
        epoch_loss = running_loss / len(train_loader)
        loss_record_train.append(epoch_loss)
                
        valid_loss = validate_model(model, 'cpu', valid_loader, nmse_loss_fn)
        loss_record_valid.append(valid_loss)
        
        learning_rate_record.append(optimizer.param_groups[0]['lr'])
        
        epoch_loss_dB = 10 * np.log10(epoch_loss)
        valid_loss_dB = 10 * np.log10(valid_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Valid Loss: {valid_loss:.4f}, Train Loss(dB): {epoch_loss_dB:.4f}, Valid Loss(dB): {valid_loss_dB:.4f}')
        
        scheduler.step()
        
        save_txt(loss_record_train, loss_record_valid, learning_rate_record, exp_folder)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            no_improve_epoch = 0
            save_checkpoint(model, optimizer, scheduler, epoch, f"{exp_folder}/model_quantized.pth")
        else:
            no_improve_epoch += 1
            
        # Early stopping
        if no_improve_epoch >= 30:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        epoch += 1
                
    print('Finished Training')
    return model

def validate_model(model, device, valid_loader, loss_fn):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for h_est, h_perfect in valid_loader:
            h_est = h_est.to(device)
            h_perfect = h_perfect.to(device)
            
            output = model(h_est)
                
            total_loss += loss_fn(output, h_perfect).item()
            
    valid_loss = total_loss / len(valid_loader)
    valid_loss_db = 10 * np.log10(valid_loss)
    return valid_loss
 
def save_checkpoint(model, optimizer=None, scheduler=None, epoch=0, filename=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint, filename)
    
def load_checkpoint(model, from_checkpoint=None):
    checkpoint = torch.load(from_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
    
def main():
    file_config = FilesConfig()
    model_config = ModelConfig()

    ## Load data
    print("----- Loading data... -----")
    
    folder_path = file_config.processed_folder_path
    
    h_estimated_trains = np.load(folder_path + 'h_estimated_trains_symbol_level_FRE_4_SPA_4.npy').astype(np.float32)
    h_perfect_srs_trains = np.load(folder_path + 'h_perfect_srs_trains_symbol_level.npy').astype(np.float32)
    
    print("Shape of h_estimated_trains: ", h_estimated_trains.shape)
    print("Shape of h_perfect_srs_trains: ", h_perfect_srs_trains.shape)

    ## Model
    model_fp32 = SwiftChannelRP(2, 2, upscale=4, middle_channels=8, feature_channels=12)
    model_quant = SwiftChannelQuan(2, 2, upscale=4, middle_channels=8, feature_channels=12)
    
    exp_date = '0508'
    exp_time = '1423'
    model_used = 'Distillation'
    loss_used = model_config.train_loss
    fre = 'FRE_4'
    spa = 'SPA_4'
    file_save_name = f"Experiments/Experiment_{exp_date}_{exp_time}_{model_used}_{loss_used}_{fre}_{spa}"
    
    model_fp32 = load_checkpoint(model_fp32, f"{file_save_name}/model_checkpoint_best_eval.pth")
    model_quant = load_checkpoint(model_quant, f"{file_save_name}/model_checkpoint_best_eval.pth")
    model_fp32.eval().to('cpu')
    model_quant.eval().to('cpu')
    
    print("Size of baseline model")
    print_size_of_model(model_fp32)
    
    # Per-channel quantization
    model_quant.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
    torch.backends.quantized.engine = 'x86'
    torch.ao.quantization.prepare_qat(model_quant.train(), inplace=True)
    
    # Quantization aware training
    train_loader, valid_loader = prepare_data(h_estimated_trains, h_perfect_srs_trains, model_config.batch_size, model_config.num_workers)
    _, nmse_loss_fn = setup_loss_fn(model_config)
    
    # save the quantized model
    current_date = time.strftime("%m%d")
    current_time = time.strftime("%H%M")
    model_used = 'QAT'
    loss_used = 'NMSE'
    file_save_name = f"Experiments/Experiment_{current_date}_{current_time}_{model_used}_{loss_used}"  
    if not os.path.exists(file_save_name):
        os.makedirs(file_save_name)
    
    num_epochs = 150
    learning_rate = 0.0005
    model_quant = train_quan_model(model_quant, model_config, num_epochs, model_config.random_seed, valid_loader, valid_loader, exp_folder=file_save_name, learning_rate=learning_rate)
    
    torch.ao.quantization.convert(model_quant, inplace=True)
    
    # Validate the quantized model
    valid_loss = validate_model(model_fp32, 'cpu', train_loader, nmse_loss_fn)
    print(f'Floating Point Model: Validation Loss: {valid_loss}')
    
    quan_loss = validate_model(model_quant, 'cpu', train_loader, nmse_loss_fn)
    print(f'Quantization Aware Training: Validation Loss: {quan_loss}')
    
    print("Size of model after quantization")
    print_size_of_model(model_quant)
    
if __name__ == '__main__':
    main()