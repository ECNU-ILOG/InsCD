import os
import sys
import yaml
import importlib
import datetime
from accelerate.utils import set_seed
from typing import Union, Optional

def init_seed(seed: int, reproducibility: bool):
    """
    Initialize random seeds for reproducibility across random functions in numpy, torch, cuda, and cudnn.

    Args:
        seed (int): Random seed value.
        reproducibility (bool): Whether to enforce reproducibility.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.benchmark = not reproducibility
    torch.backends.cudnn.deterministic = reproducibility


def get_local_time() -> str:
    """
    Get the current local time in a specific format.

    Returns:
        str: Current time formatted as "Month-Day-Year_Hour-Minute-Second".
    """
    return datetime.datetime.now().strftime("%b-%d-%Y_%H-%M-%S")


def get_command_line_args_str() -> str:
    """
    Get the command line arguments as a single string, with '/' replaced by '|'.

    Returns:
        str: The command line arguments.
    """
    return '_'.join(sys.argv).replace('/', '|')


def parse_command_line_args(unparsed: list[str]) -> dict:
    """
    Parse command line arguments into a dictionary.

    Args:
        unparsed (list[str]): List of unparsed command line arguments.

    Returns:
        dict: Parsed arguments as key-value pairs.

    Raises:
        ValueError: If the argument format is invalid.
    """
    args = {}
    for arg in unparsed:
        if '=' not in arg:
            raise ValueError(f"Invalid command line argument: {arg}. Expected format is '--key=value'.")
        key, value = arg.split('=')
        key = key.lstrip('--')
        try:
            value = eval(value)
        except (NameError, SyntaxError):
            pass
        args[key] = value

    return args

def get_config(
        model_name: str,
        config_file: Union[str, list[str], None],
        config_dict: Optional[dict]
) -> dict:
    # print("config_file",config_dict)
    """
    Get the configuration for a model and dataset.

    Args:
        model_name (Union[str, AbstractModel]): The name or instance of the model.
        dataset_name (Union[str, AbstractDataset]): The name or instance of the dataset.
        config_file (Union[str, list[str], None]): Additional configuration file(s).
        config_dict (Optional[dict]): Dictionary of additional configuration options.

    Returns:
        dict: The final configuration dictionary.

    Raises:
        FileNotFoundError: If any of the specified configuration files are missing.
    """
    final_config = {}
    current_path = os.path.dirname(os.path.realpath(__file__))
    config_file_list = [os.path.join(current_path, 'default.yaml')]


    config_file_list.append(os.path.join(current_path, f'configs/{model_name}.yaml'))
    final_config['model'] = model_name

    if config_file:
        if isinstance(config_file, str):
            config_file = [config_file]
        config_file_list.extend(config_file)

    for file in config_file_list:
        with open(file, 'r') as f:
            cur_config = yaml.safe_load(f)
            if cur_config:
                final_config.update(cur_config)

    if config_dict:
        final_config.update(config_dict)

    final_config['run_local_time'] = get_local_time()
    # if final_config['datahub_name']=="Matmat":
    #     final_config['lr']=1e-4
    if final_config['model']=="DCD" and (final_config['datahub_name']=="Assist17" or final_config['datahub_name']=="NeurIPS20"):
        final_config['lr']=1e-3
        final_config['train_batch_size']=128
        final_config['eval_batch_size']=128
    if final_config['model']=="DCD" and (final_config['datahub_name']=="Math1" or final_config['datahub_name']=="Math2"):
        final_config['lr']=1e-3
        final_config['train_batch_size']=64
        final_config['eval_batch_size']=64
    if final_config['model']=="DCD" and final_config['datahub_name']=="FrcSub":
        final_config['lr']=5e-3
        final_config['train_batch_size']=64
        final_config['eval_batch_size']=64
    if final_config['model']=="DCD" and final_config['datahub_name']=="Junyi734":
        final_config['lr']=1e-5
    if final_config['model']=="NCDM" and final_config['datahub_name']=="XES3G5M":
        final_config['lr']=5e-5
        final_config['train_batch_size']=2048
        final_config['eval_batch_size']=2048




    return convert_config_dict(final_config)

def convert_config_dict(config: dict) -> dict:
    """
    Convert configuration values in a dictionary to their appropriate types.

    Args:
        config (dict): The dictionary containing the configuration values.

    Returns:
        dict: The dictionary with converted values.
    """
    for key, value in config.items():
        if isinstance(value, str):
            try:
                new_value = eval(value)
                if new_value is not None and not isinstance(new_value, (str, int, float, bool, list, dict, tuple)):
                    new_value = value
            except (NameError, SyntaxError, TypeError):
                new_value = value.lower() == 'true' if value.lower() in ['true', 'false'] else value
            config[key] = new_value

    return config


def init_device() -> tuple:
    """
    Set the visible devices for training, supporting multiple GPUs.

    Returns:
        tuple: A tuple containing the torch device and whether DDP (Distributed Data Parallel) is enabled.
    """
    import torch

    use_ddp = bool(os.environ.get("WORLD_SIZE"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return device, use_ddp

def get_datahub(datahub_name: str):
    try:
        datahub_class = getattr(importlib.import_module('inscd.datahub'), datahub_name)
    except AttributeError:
        raise ValueError(f'Datahub "{datahub_name}" not found.')

    return datahub_class
def get_model(model_name: str):
    """
    Retrieve the model class based on the provided model name.

    Args:
        model_name (Union[str, AbstractModel]): The name or instance of the model.

    Returns:
        AbstractModel: The model class corresponding to the provided model name.

    Raises:
        ValueError: If the model name is not found.
    """
    try:
        model_class = getattr(importlib.import_module('inscd.models'), model_name)
    except AttributeError:
        raise ValueError(f'Model "{model_name}" not found.')

    return model_class

def get_total_steps(config, train_dataloader):
    """
    Calculate the total number of steps for training based on the given configuration and dataloader.

    Args:
        config (dict): The configuration dictionary containing the training parameters.
        train_dataloader (DataLoader): The dataloader for the training dataset.

    Returns:
        int: The total number of steps for training.

    """
    if config['steps'] is not None:
        return config['steps']
    else:
        return len(train_dataloader) * config['epochs']

def get_file_name(config: dict, suffix: str = ''):
    import hashlib
    config_str = "".join([str(value) for key, value in config.items() if key != 'accelerator'])
    md5 = hashlib.md5(config_str.encode(encoding="utf-8")).hexdigest()[:6]
    command_line_args = get_command_line_args_str()
    logfilename = "{}-{}-{}-{}{}".format(
        config["run_id"], command_line_args, config['run_local_time'], md5, suffix
    )
    return logfilename

def get_split(config: dict, datahub):
    if config['split_manner'] == 'standard':
        datahub.random_split(slice_out=1 - config.get('test_size') - config.get('val_size'), seed=config['rand_seed'])
        datahub.random_split(source="valid", to=["valid", "test"], slice_out=config.get('val_size')/(config.get('val_size')+config.get('test_size')), seed=config['rand_seed'])
    elif config['split_manner'] == 'hold_out':
        datahub.random_split(slice_out=1 - config.get('test_size'), to=["train", "test"], seed=config['rand_seed'])
        datahub._set_type_map['valid'] = datahub._set_type_map['test']
    else:
        raise ValueError("No such manner")
    return datahub