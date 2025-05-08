import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # use which GPU to train
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # ignore the information messages

import shutil

from Models.swiftchannel_teacher import SwiftChannelTeacher
from Models.swiftchannel_student import SwiftChannelStudent

from Metrics.metrics import NMSELoss
from config_file.config import FilesConfig, ModelConfig

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
    # Ensures deterministic algorithms are used
    # torch.use_deterministic_algorithms(True)
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
    decay_start_from = 10

    if epoch < decay_start_from:
        return 1.0
    else:
        return 0.9 ** ((epoch - decay_start_from) // 30)

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

def setup_optimizer(model, model_config, learning_rate):
    if model_config.optimizer == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
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
        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    raise ValueError(f"Unsupported scheduler: {model_config.scheduler}")

def save_plots(loss_record_train, loss_record_valid, learning_rate_record, exp_name):
    # Plot and save loss
    plt.figure()
    # plt.plot(loss_record_train, label='Training Loss')
    plt.plot(loss_record_valid[3:], label='Validation Loss')
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

def distillation_loss(student_output, teacher_output, true_output, student_middle, teacher_middle, student_features, teacher_features, loss_fn, epoch, alpha=0.5, beta=0.5, gamma=0.5, return_loss=False):  
    if epoch < 3:
        alpha = 1
        beta = 0
        gamma = 0
        loss_fn = nn.MSELoss()
    
    # Hard loss: Student vs. true output (ground truth)
    hard_loss = loss_fn(student_output, true_output)
    
    # Soft loss: Student vs. teacher output
    soft_loss = loss_fn(student_output, teacher_output)
    
    # Feature loss: Student vs. teacher feature maps
    feature_loss = 0.0
    for sm, tm in zip(student_middle, teacher_middle):
        feature_loss += nn.MSELoss()(sm, tm)
    feature_loss /= len(student_middle)

    # Combine hard and soft losses
    if not return_loss:
        return alpha * hard_loss + beta * soft_loss + gamma * feature_loss
    else:
        return alpha * hard_loss + beta * soft_loss + gamma * feature_loss, hard_loss, soft_loss, feature_loss

def train_student(student_model, teacher_model, model_config, device, random_seed, train_loader, valid_loader, \
    exp_name, save_model=False, save_img=False, from_checkpoint=None, alpha=0.5, beta=0.5, gamma=0.5, \
        learning_rate=1e-4, num_epochs=200, early_stopping_patience=50):
    epoch = 0
    no_improve_epoch = 0
    best_valid_loss = float('inf')
    loss_record_train, loss_record_valid, learning_rate_record = [], [], []
    loss_teacher = 0

    # Setup tensorboard writer
    writer = SummaryWriter(f"{exp_name}/tensorboard/")

    # Setup optimizer, loss function, and scheduler
    optimizer = setup_optimizer(student_model, model_config, learning_rate=learning_rate)
    loss_fn, nmse_loss_fn = setup_loss_fn(model_config)
    scheduler = setup_scheduler(optimizer, model_config)

    # Set random seed
    set_seed(random_seed)
    
    # Resume training from checkpoint
    if from_checkpoint is not None:
        student_model, optimizer, scheduler, epoch = load_checkpoint(student_model, optimizer, scheduler, from_checkpoint)
        print(f"Resumed training from checkpoint at epoch {epoch}")
    
    # Freeze the teacher model (no training required)
    teacher_model.eval().to(device)
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Training loop
    while epoch < num_epochs:
        running_loss = 0.0
        teacher_model.eval().to(device)
        student_model.train().to(device)

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit='batch') as pbar:
            for batch_idx, (h_est, h_perfect) in enumerate(train_loader):
                h_est, h_perfect = h_est.to(device), h_perfect.to(device)

                optimizer.zero_grad()

                # Get predictions from both the student and the teacher model
                student_output, student_middle, student_features = student_model(h_est)
                with torch.no_grad():
                    teacher_output, teacher_middle, teacher_features = teacher_model(h_est)  # Teacher's output (no gradient)

                # Compute distillation loss
                loss = distillation_loss(student_output=student_output, teacher_output=teacher_output, true_output=h_perfect, \
                    student_middle=student_middle, teacher_middle=teacher_middle, student_features=student_features, teacher_features=teacher_features,\
                    loss_fn=loss_fn, epoch=epoch, alpha=alpha, beta=beta)
                loss.backward()

                optimizer.step()

                nmse_loss = nmse_loss_fn(student_output, h_perfect)
                running_loss += nmse_loss.item()
                writer.add_scalar('Train/Loss', nmse_loss.item(), epoch * len(train_loader) + batch_idx)
                log_training_gradient(student_model, writer, epoch, train_loader, batch_idx)

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
                
        loss, hard_loss, soft_loss, feature_loss = distillation_loss(student_output=student_output, teacher_output=teacher_output, true_output=h_perfect, \
                    student_middle=student_middle, teacher_middle=teacher_middle, student_features=student_features, teacher_features=teacher_features,\
                    loss_fn=loss_fn, epoch=epoch, alpha=alpha, beta=beta, return_loss=True)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss:.4f}, Hard Loss: {hard_loss:.4f}, Soft Loss: {soft_loss:.4f}, Feature Loss: {feature_loss:.4f}")

        # Log average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        loss_record_train.append(epoch_loss)

        # Validation phase
        valid_loss = validate_model(student_model, device, valid_loader, nmse_loss_fn, epoch, save_img, exp_name)
        loss_record_valid.append(valid_loss)
        
        loss_teacher = validate_model(teacher_model, device, valid_loader, nmse_loss_fn, epoch, save_img=False, exp_name=exp_name)

        learning_rate_record.append(optimizer.param_groups[0]['lr'])

        # Log epoch results
        epoch_loss_dB = 10 * np.log10(epoch_loss)
        valid_loss_dB = 10 * np.log10(valid_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Valid Loss: {valid_loss:.4f}, Teacher Loss: {loss_teacher:.4f}, "
              f"Train Loss(dB): {epoch_loss_dB:.4f}, Valid Loss(dB): {valid_loss_dB:.4f}")

        # Save model if validation loss improves
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            no_improve_epoch = 0
            if save_model:
                save_checkpoint(student_model, optimizer, scheduler, epoch, f"{exp_name}/model_checkpoint_best.pth")
        else:
            no_improve_epoch += 1

        # Early stopping
        if no_improve_epoch >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Update learning rate scheduler
        scheduler.step()

        save_txt(loss_record_train, loss_record_valid, learning_rate_record, exp_name)
        if save_img:
            save_plots(loss_record_train, loss_record_valid, learning_rate_record, exp_name)

        epoch += 1

    print("Finished Training")
    
def validate_model(model, device, valid_loader, loss_fn, epoch, save_img, exp_name, split_channel=False):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        
        for h_est, h_perfect in valid_loader:
            h_est = h_est.to(device)
            h_perfect = h_perfect.to(device)
            
            if split_channel:
                h_est_real = h_est[:, 0, :, :].unsqueeze(1)
                h_est_imag = h_est[:, 1, :, :].unsqueeze(1)

                output_real = model(h_est_real)
                output_imag = model(h_est_imag)

                output = torch.stack((output_real, output_imag), dim=1).squeeze(2)
            else:
                output, _, _ = model(h_est)
                
            total_loss += loss_fn(output, h_perfect).item()
            
        if save_img:
            fig_filename = exp_name + f"/epoch_{epoch}_valid.png"
            infer_during_training(h_est, h_perfect, output, fig_filename, index=epoch%16)
            
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
    
def load_checkpoint(model, from_checkpoint):
    checkpoint = torch.load(from_checkpoint, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    return model, epoch

def main():
    file_config = FilesConfig()
    model_config = ModelConfig()
    
    ## Load data
    print("----- Loading data... -----")
    
    folder_path = file_config.processed_folder_path
    
    ## Load processed data
    h_estimated_trains = np.load(folder_path + 'h_estimated_trains_symbol_level_FRE_4_SPA_4.npy').astype(np.float32)
    h_perfect_srs_trains = np.load(folder_path + 'h_perfect_srs_trains_symbol_level.npy').astype(np.float32)
    
    print("Shape of h_estimated_trains: ", h_estimated_trains.shape)
    print("Shape of h_perfect_srs_trains: ", h_perfect_srs_trains.shape)

    ## Model
    model_teacher = SwiftChannelTeacher(2, 2, upscale=4, feature_channels=24, feature_output=True)
    model_student = SwiftChannelStudent(2, 2, upscale=4, middle_channels=8, feature_channels=12, feature_output=True)
    model_used = 'Distillation'
    
    ex_date = "0508"
    ex_time = "1417"
    ex_model = "Teacher"
    ex_loss = "NMSE"
    fre = "FRE_4"
    spa = "SPA_4"
    ex_folder = "Experiments/Experiment_" + ex_date + "_" + ex_time + "_" + ex_model + "_" + ex_loss + "_" + fre + "_" + spa
    model_path = ex_folder + "/model_checkpoint_best.pth"
    model_teacher, epoch = load_checkpoint(model_teacher, model_path)
    print("Loaded model from epoch", epoch)
    
    train_loader, valid_loader = prepare_data(h_estimated_trains, h_perfect_srs_trains, model_config.batch_size, model_config.num_workers)
    
    print("----- Training -----")
    current_date = time.strftime("%m%d")
    current_time = time.strftime("%H%M")
    file_save_name = f"Experiments/Experiment_{current_date}_{current_time}_{model_used}_{ex_loss}_{fre}_{spa}"
    if not os.path.exists(file_save_name):
        os.makedirs(file_save_name)
        
    shutil.copy('config_file/model_config.yaml', file_save_name + '/model_config.yaml')
    shutil.copy('config_file/config.yaml', file_save_name + '/config.yaml')
    shutil.copy('config_file/files_config.yaml', file_save_name + '/files_config.yaml')
    shutil.copy('2_main.py', file_save_name + '/main.py')
    shutil.copy('2_knowledge_distillation.py', file_save_name + '/knowledge_distillation.py')
    shutil.copy('Models/swiftchannel_teacher.py', file_save_name + '/swiftchannel_teacher.py')
    shutil.copy('Models/swiftchannel_student.py', file_save_name + '/swiftchannel_student.py')
    
    alpha = 1
    beta = 10
    gamma = 2
    learning_rate = 0.0002
    num_epochs = 300
    early_stopping_patience = 30
    train_student(model_student, model_teacher, model_config, model_config.device, model_config.random_seed, train_loader, \
        valid_loader, file_save_name, save_model=True, save_img=True, alpha=alpha, beta=beta, gamma=gamma, \
            learning_rate=learning_rate, num_epochs=num_epochs, early_stopping_patience=early_stopping_patience)
        
if __name__ == '__main__':
    main()