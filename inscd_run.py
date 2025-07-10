import argparse
import os
from inscd import Runner
import warnings
from inscd.utils import parse_command_line_args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DCD', help='Model name')
    parser.add_argument('--datahub_name', type=str, default='XES3G5M')
    return parser.parse_known_args()

def cli():
    warnings.filterwarnings("ignore")
    os.environ["WANDB_MODE"] = 'offline'

    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)
    args_dict = vars(args)
    merged_dict = {**args_dict, **command_line_configs}

    runner = Runner(
        model_name=args.model,
        config_dict=merged_dict
    )
    runner.run()

if __name__ == "__main__":
    cli()
