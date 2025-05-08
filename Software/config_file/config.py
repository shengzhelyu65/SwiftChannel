import yaml

def get_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

class Config:
    def __init__(self):
        config = 'config_file/config.yaml'
        config = get_config(config)
    
        self.num_BS_antennas = config['parameters']['num_BS_antennas']
        self.num_UE_antennas = config['parameters']['num_UE_antennas']
        self.num_RBs = config['parameters']['num_RBs']
        self.num_subcarriers_RB = config['parameters']['num_subcarrier_RB']
        self.num_symbol_slot = config['parameters']['num_symbol_slot']
        self.subcarrier_spacing_Hz = config['parameters']['subcarrier_spacing_Hz']
        self.num_frame_sample = config['parameters']['num_frame_sample']
        self.num_subframe_frame = config['parameters']['num_subframe_frame']
        self.num_slot_subframe = config['parameters']['num_slot_subframe']
        
        self.special_slot_period = config['parameters']['special_slot_period']
        self.srs_start_symbol = config['parameters']['srs_start_symbol']
        self.srs_end_symbol = config['parameters']['srs_end_symbol']
    
        self.num_subframes = self.num_frame_sample * self.num_subframe_frame
        self.num_slots = self.num_subframes * self.num_slot_subframe
        
        self.spatial_compresion_ratio = config['parameters']['spatial_compresion_ratio']
        self.frequency_compresion_ratio = config['parameters']['frequency_compresion_ratio']
        
        self.dim_antennas = self.num_BS_antennas * self.num_UE_antennas
        self.dim_antennas_compresed = self.dim_antennas // self.spatial_compresion_ratio
        self.dim_frequency = self.num_RBs * self.num_subcarriers_RB
        self.dim_frequency_compresed = self.dim_frequency // self.frequency_compresion_ratio


class FilesConfig:
    def __init__(self):
        config = 'config_file/files_config.yaml'
        config = get_config(config)
        
        self.train_folder_path = config['paths']['training_folder_path']
        self.test_folder_path = config['paths']['testing_folder_path']
        self.processed_folder_path = config['paths']['processed_folder_path']
        
        self.mode_selection = config['mode']['mode_selection']
    
        self.hPerfect_key = config['keys']['hPerfect_key']
        self.txGrid_key = config['keys']['txGrid_key']
        self.rxGrid_key = config['keys']['rxGrid_key']
        self.rxWaveform_key = config['keys']['rxWaveform_key']
   
class ModelConfig:
    def __init__(self):
        config = 'config_file/model_config.yaml'
        config = get_config(config)
    
        self.batch_size = config['training']['batch_size']
        self.num_epochs = config['training']['num_epochs']
        self.learning_rate = config['training']['learning_rate']
        self.lr_step_size = config['training']['lr_step_size']
        self.lr_gamma = config['training']['lr_gamma']
        self.random_seed = config['training']['random_seed']
        self.num_workers = config['training']['num_workers']
        self.model = config['training']['model']
        self.train_loss = config['training']['train_loss']
        self.val_loss = config['training']['val_loss']
        self.optimizer = config['training']['optimizer']
        self.scheduler = config['training']['scheduler']
        self.device = config['training']['device']
        self.early_stopping_patience = config['training']['early_stopping_patience']
        self.data_augmentation = config['training']['data_augmentation']