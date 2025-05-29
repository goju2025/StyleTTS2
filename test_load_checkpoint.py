import os
import sys
import torch
import yaml
import warnings
from munch import Munch

# Add current directory to sys.path to allow imports from local modules
sys.path.insert(0, os.getcwd())

# Now try importing local modules
try:
    import models
    import optimizers
    from Utils.PLBERT.util import load_plbert
    from utils import recursive_munch # Explicitly import recursive_munch
except ImportError as e:
    print(f"Error importing local modules: {e}. Ensure all modules are in the Python path.")
    sys.exit(1)

import logging

# Suppress warnings
warnings.simplefilter('ignore')

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load configuration
    config_path = 'Configs/config.yml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        print(f"Failure: Error loading configuration from {config_path}: {e}")
        sys.exit(1)

    # Construct checkpoint path
    checkpoint_filename = config.get('first_stage_path', 'first_stage.pth') 
    log_dir = config.get('log_dir')
    
    if not log_dir:
        logger.error("'log_dir' not found in config or is empty. Cannot construct checkpoint path.")
        print("Failure: 'log_dir' not found in config.") 
        sys.exit(1)

    checkpoint_path = os.path.join(log_dir, checkpoint_filename)
    logger.info(f"Constructed checkpoint path: {checkpoint_path}")

    # Check if checkpoint file exists
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found at {checkpoint_path}. Cannot test loading.")
        print(f"Failure: Checkpoint file not found at {checkpoint_path}.") 
        sys.exit(1) 

    # Initialize auxiliary models
    logger.info("Initializing auxiliary models...")
    try:
        text_aligner = models.load_ASR_models(config.get('ASR_path'), config.get('ASR_config'))
        logger.info("ASR model loaded.")
        pitch_extractor = models.load_F0_models(config.get('F0_path'))
        logger.info("F0 model loaded.")
        plbert = load_plbert(config.get('PLBERT_dir'))
        logger.info("PL-BERT model loaded.")
    except Exception as e:
        logger.error(f"Error initializing auxiliary models: {e}")
        print(f"Failure: Error initializing auxiliary models: {e}")
        sys.exit(1)

    # Build main model structure
    logger.info("Building main model structure...")
    try:
        model_params_munch = recursive_munch(config['model_params'])
        main_model = models.build_model(model_params_munch, text_aligner, pitch_extractor, plbert)
        logger.info("Main model structure built.")
    except Exception as e:
        logger.error(f"Error building main model structure: {e}")
        print(f"Failure: Error building main model structure: {e}")
        sys.exit(1)

    # Initialize optimizers
    logger.info("Initializing optimizers...")
    try:
        # Use a more robust way to get model keys if model is a list/dict
        model_keys = main_model.keys() if isinstance(main_model, dict) else range(len(main_model))
        
        scheduler_params_dict = {
            key: {"max_lr": 0.0001, "pct_start": 0.0, "epochs": 1, "steps_per_epoch": 1} 
            for key in model_keys
        }
        optimizer = optimizers.build_optimizer(
            {key: main_model[key].parameters() for key in model_keys}, 
            scheduler_params_dict=scheduler_params_dict,
            lr=0.0001
        )
        logger.info("Optimizers initialized.")
    except Exception as e:
        logger.error(f"Error initializing optimizers: {e}")
        print(f"Failure: Error initializing optimizers: {e}")
        sys.exit(1)

    # Attempt to load checkpoint
    logger.info(f"Attempting to load checkpoint from {checkpoint_path}...")
    try:
        # Ensure model components are on CPU for this test
        for key in main_model.keys() if isinstance(main_model, dict) else range(len(main_model)): # Check if main_model is a dictionary
            if isinstance(main_model, dict):
                main_model[key].cpu() 
            # else: # If it's not a dictionary, it might be a list or other iterable of models - build_model returns a Munch (dict-like)
                # This branch might not be strictly necessary if build_model always returns a dict-like object
                # main_model[key].cpu() #This line would cause error if main_model is not a list of modules


        main_model, optimizer, start_epoch, iters = models.load_checkpoint(
            main_model, optimizer, checkpoint_path, load_only_params=True, device='cpu'
        )
        logger.info(f"Checkpoint loaded successfully. Start epoch: {start_epoch}, Iters: {iters}")
        print("Success: Model checkpoint loaded without errors.") 
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}", exc_info=True)
        print(f"Failure: Error during checkpoint loading: {e}") 
        sys.exit(1)

if __name__ == '__main__':
    main()
