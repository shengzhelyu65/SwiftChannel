import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # use which GPU to train
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # ignore the information messages

import sys
import shutil
from Baselines.model_zoo import return_model
from Metrics.metrics import NMSELoss
from config_file.config import Config, FilesConfig, BaselineConfig

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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
    
def log_training_gradient(model, writer, epoch, dataloader, i):
    total_norm = 0
    param_count = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    total_norm = total_norm ** (1. / 2)
    writer.add_scalar('Train/Gradient_b', total_norm, epoch * len(dataloader) + i)
    
def lr_lambda(epoch):
    decay_start_from = 0

    if epoch < decay_start_from:
        return 1.0
    else:
        return 0.9 ** ((epoch - decay_start_from) // 40)

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

def setup_optimizer(model, model_config):
    if model_config.optimizer == 'Adam':
        return optim.Adam(model.parameters(), lr=model_config.learning_rate)
    # Add more optimizers if needed
    raise ValueError(f"Unsupported optimizer: {model_config.optimizer}")

def setup_loss_fn(model_config):
    if model_config.train_loss == 'MSE':
        return nn.MSELoss(), NMSELoss()
    elif model_config.train_loss == 'NMSE':
        return NMSELoss(), NMSELoss()
    elif model_config.train_loss == 'L1':
        return nn.L1Loss(), NMSELoss()
    raise ValueError(f"Unsupported loss function: {model_config.train_loss}")

def setup_scheduler(optimizer, model_config):
    if model_config.scheduler == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=model_config.lr_step_size, gamma=model_config.lr_gamma)
    elif model_config.scheduler == 'LambdaLR':
        return LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"Unsupported scheduler: {model_config.scheduler}")

def save_plots(loss_record_train, loss_record_valid, learning_rate_record, exp_name):
    # Plot and save loss
    plt.figure()
    plt.plot(loss_record_train[2:], label='Training Loss')
    plt.plot(loss_record_valid[2:], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{exp_name}/loss.png")
    plt.close()

    # Plot and save learning rate
    plt.figure()
    plt.plot(learning_rate_record)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.savefig(f"{exp_name}/lr.png")
    plt.close()
    
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
    
def train_model(model, model_config, device, random_seed, train_loader, valid_loader, exp_name, save_model=False, save_img=False, from_checkpoint=None):
    epoch = 0
    no_improve_epoch = 0
    best_valid_loss = float('inf')
    loss_record_train, loss_record_valid, learning_rate_record = [], [], []

    # Setup tensorboard writer
    writer = SummaryWriter(f"{exp_name}/tensorboard/")

    # Setup optimizer, loss function, and scheduler
    model.train().to(device)
    optimizer = setup_optimizer(model, model_config)
    loss_fn, nmse_loss_fn = setup_loss_fn(model_config)
    scheduler = setup_scheduler(optimizer, model_config)
    
    # Set random seed
    set_seed(random_seed)
    
    # Resume training from checkpoint
    if from_checkpoint is not None:
        model, optimizer, scheduler, epoch = load_checkpoint(model, optimizer, scheduler, from_checkpoint)
        print(f"Resumed training from checkpoint at epoch {epoch}")

    # Training loop
    while epoch < model_config.num_epochs:
        model.train().to(device)
        running_loss = 0.0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{model_config.num_epochs}", unit='batch') as pbar:
            for batch_idx, (h_est, h_perfect) in enumerate(train_loader):
                h_est = h_est.to(device)
                h_perfect = h_perfect.to(device)
                
                optimizer.zero_grad()
                output = model(h_est)
                
                if model_config.train_loss == 'NMSE':
                    if epoch < 2:
                        loss = nn.MSELoss()(output, h_perfect)
                    else:
                        loss = loss_fn(output, h_perfect)
                else:
                    loss = loss_fn(output, h_perfect)
                
                loss.backward()
                
                if model_config.train_loss != 'NMSE':
                    nmse_loss = nmse_loss_fn(output, h_perfect)
                    running_loss += nmse_loss.item()
                    writer.add_scalar('Train/Loss_b', nmse_loss.item(), epoch * len(train_loader) + batch_idx)
                elif model_config.train_loss == 'NMSE':
                    running_loss += loss.item()
                    writer.add_scalar('Train/Loss_b', loss.item(), epoch * len(train_loader) + batch_idx)
                    
                log_training_gradient(model, writer, epoch, train_loader, batch_idx)
                
                optimizer.step()
                
                # Update the progress bar
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
                
        epoch_loss = running_loss / len(train_loader)
        loss_record_train.append(epoch_loss)
                
        valid_loss = validate_model(model, device, valid_loader, nmse_loss_fn, epoch, save_img, exp_name, model_config.batch_size)
        loss_record_valid.append(valid_loss)
        
        learning_rate_record.append(optimizer.param_groups[0]['lr'])
        
        epoch_loss_dB = 10 * np.log10(epoch_loss)
        valid_loss_dB = 10 * np.log10(valid_loss)
        print(f'Epoch {epoch+1}/{model_config.num_epochs}, Train Loss: {epoch_loss:.4f}, Valid Loss: {valid_loss:.4f}, Train Loss(dB): {epoch_loss_dB:.4f}, Valid Loss(dB): {valid_loss_dB:.4f}')
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            no_improve_epoch = 0
            if save_model:
                save_checkpoint(model, optimizer, scheduler, epoch, f"{exp_name}/model_checkpoint_best.pth")
        else:
            no_improve_epoch += 1
            
        if no_improve_epoch >= model_config.early_stopping_patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
        
        scheduler.step()
        
        save_txt(loss_record_train, loss_record_valid, learning_rate_record, exp_name)
        if save_img and epoch > 2:
            save_plots(loss_record_train, loss_record_valid, learning_rate_record, exp_name)
        
        epoch += 1
                
    print('Finished Training')
    
def validate_model(model, device, valid_loader, loss_fn, epoch, save_img, exp_name, batch_size):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        
        for h_est, h_perfect in valid_loader:
            h_est = h_est.to(device)
            h_perfect = h_perfect.to(device)
            
            output = model(h_est)
                
            total_loss += loss_fn(output, h_perfect).item()
            
        if save_img:
            fig_filename = exp_name + f"/epoch_{epoch}_valid.png"
            infer_during_training(h_est, h_perfect, output, fig_filename, index=epoch%batch_size)
            
    valid_loss = total_loss / len(valid_loader)
    return valid_loss

def infer_during_training(h_est, h_perfect, output, save_filename, index=0):
    """
    h_est: Estimated channel matrix (batch_size, 2, 78, 8)
    h_perfect: Perfect channel matrix (batch_size, 2, 624, 32)
    output: Output of the model (batch_size, 2, 624, 32)
    """
    
    # Plot the estimated and perfect channel matrix
    h_est_sample = h_est[index].cpu().detach().numpy()
    h_perfect_sample = h_perfect[index].cpu().detach().numpy()
    output_sample = output[index].cpu().detach().numpy()
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(np.abs(h_est_sample[0]), cmap='jet', aspect='auto')
    plt.colorbar()
    plt.title('Input Estimated Channel Matrix (Real)')
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(output_sample[0]), cmap='jet', aspect='auto')
    plt.colorbar()
    plt.title('Output of the Model (Real)')
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(h_perfect_sample[0]), cmap='jet', aspect='auto')
    plt.colorbar()
    plt.title('Perfect Channel Matrix (Real)')
    
    # plt.show()
    plt.savefig(save_filename)
    plt.close()
 
def save_checkpoint(model, optimizer, scheduler, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, filename)
    
def load_checkpoint(model, optimizer, scheduler, from_checkpoint):
    checkpoint = torch.load(from_checkpoint, weights_only=True, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, scheduler, epoch

def main():
    config = Config()
    file_config = FilesConfig()
    
    # Get model name from command line or use default
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        # Default model if not specified
        model_name = 'I_ResNet'
        print(f"No model specified, using default: {model_name}")
        print("Usage: python 6_train_baseline.py <model_name>")
        print("Available models: Channelformer, ChannelNet, FSRCNN, I_ResNet, LSiDNN, ReEsNet")
    
    # Load baseline-specific configuration
    model_config = BaselineConfig(model_name)
    
    folder_path = file_config.processed_folder_path
    
    ## Load processed data
    h_estimated_trains = np.load(folder_path + 'h_estimated_trains_symbol_level_FRE_4_SPA_4.npy').astype(np.float32)
    h_perfect_srs_trains = np.load(folder_path + 'h_perfect_srs_trains_symbol_level.npy').astype(np.float32)
      
    print("Shape of h_estimated_trains: ", h_estimated_trains.shape)
    print("Shape of h_perfect_srs_trains: ", h_perfect_srs_trains.shape)

    ## Model
    print("----- Model -----")
    print(f"Loading model: {model_name}")
    model = return_model(model_config.model)
    # print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    print(f"Training configuration:")
    print(f"  Batch size: {model_config.batch_size}")
    print(f"  Learning rate: {model_config.learning_rate}")
    print(f"  Optimizer: {model_config.optimizer}")
    print(f"  Scheduler: {model_config.scheduler}")
    print(f"  Train loss: {model_config.train_loss}")

    print("----- Training -----")
    current_date = time.strftime("%m%d")
    current_time = time.strftime("%H%M")
    model_used = model_config.model
    loss_used = model_config.train_loss
    fre_com = config.frequency_compresion_ratio
    spa_com = config.spatial_compresion_ratio
    file_save_name = f"Experiments/Experiment_{current_date}_{current_time}_{model_used}_{loss_used}_FRE_{fre_com}_SPA_{spa_com}"
    if not os.path.exists(file_save_name):
        os.makedirs(file_save_name)
        
    # Copy relevant config files
    shutil.copy('Baselines/baseline_config.yaml', file_save_name + '/baseline_config.yaml')
    shutil.copy('config_file/config.yaml', file_save_name + '/config.yaml')
    shutil.copy('config_file/files_config.yaml', file_save_name + '/files_config.yaml')
    shutil.copy('6_train_baseline.py', file_save_name + '/train_baseline.py')
    
    # Copy baseline model files
    baseline_model_file = f'Baselines/{model_name}.py'
    if os.path.exists(baseline_model_file):
        shutil.copy(baseline_model_file, file_save_name + f'/{model_name}.py')
    shutil.copy('Baselines/model_zoo.py', file_save_name + '/model_zoo.py')
    
    checkpoint_path = None
    train_loader, valid_loader = prepare_data(h_estimated_trains, h_perfect_srs_trains, model_config.batch_size, model_config.num_workers)
    train_model(model, model_config, model_config.device, model_config.random_seed, train_loader, valid_loader, file_save_name, save_model=True, save_img=True, from_checkpoint=checkpoint_path)
        
if __name__ == '__main__':
    main()