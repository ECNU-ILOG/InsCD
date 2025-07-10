import torch
from typing import Union
from torch.utils.data import DataLoader
from accelerate import Accelerator
from inscd.utils import *
from inscd import Unifier
import torch.distributed as dist

class Runner:
    def __init__(
            self,
            model_name: str,
            config_dict: dict = None,
            config_file: str = None,
    ):
        self.config = get_config(
            model_name=model_name,
            config_file=config_file,
            config_dict=config_dict
        )
        # print(self.config)

        # Automatically set devices and ddp
        self.config['device'], self.config['use_ddp'] = init_device()
        self.accelerator = Accelerator(log_with='wandb')

        init_seed(self.config['rand_seed'], self.config['reproducibility'])
        
        self.config['accelerator'] = self.accelerator
        self.datahub = get_datahub(self.config['datahub_name'])()
        self.datahub = get_split(self.config, self.datahub)

        self.config['student_num'] = self.datahub.student_num
        self.config['exercise_num'] = self.datahub.exercise_num
        self.config['knowledge_num'] = self.datahub.knowledge_num
        self.config['datahub'] = self.datahub

        with self.accelerator.main_process_first():
            self.model = get_model(model_name)(self.config)

        self.trainer = Unifier(self.config, self.model, self.datahub)
    
    def run(self):

        train_dataloader = self.datahub.to_dataloader(
            batch_size=self.config["train_batch_size"],
            set_type='train',
            label=True
        )
        val_dataloader = self.datahub.to_dataloader(
            batch_size=self.config["eval_batch_size"],
            set_type='valid',
            label=True,
                        shuffle=False
        )
        test_dataloader = self.datahub.to_dataloader(
            batch_size=self.config["eval_batch_size"],
            set_type='test',
            label=True,
            shuffle=False
        )
        self.trainer.train(train_dataloader, val_dataloader)

        self.accelerator.wait_for_everyone()

        target_model = self.accelerator.unwrap_model(self.model)
        if self.accelerator.is_main_process:
            if os.path.exists(self.trainer.saved_model_ckpt):
                state_dict = torch.load(self.trainer.saved_model_ckpt, map_location='cpu')
                target_model.load_state_dict(state_dict)
            else:
                print(f"Checkpoint {self.trainer.saved_model_ckpt} not found.")

        if self.config['use_ddp']:
            dist.barrier()

        self.model, test_dataloader = self.accelerator.prepare(
            self.model, test_dataloader
        )
        if self.accelerator.is_main_process:
            print(f'Loaded best model checkpoint from {self.trainer.saved_model_ckpt}')
        test_results = self.trainer.evaluate(test_dataloader)
        if self.accelerator.is_main_process:
            for key in test_results:
                formatted_value = round(test_results[key] * 100, 2) 
                self.accelerator.log({f'Test_Metric/{key}': formatted_value})

        formatted_metrics = ', '.join(f"{key}: {round(value * 100, 2)}%" for key, value in test_results.items())
        print(f"Test Metrics: {formatted_metrics}")

        self.trainer.end()