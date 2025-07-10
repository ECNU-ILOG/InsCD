import os
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, OrderedDict
from inscd.utils import get_total_steps, get_file_name
from inscd import Ruler

class Unifier():
    def __init__(self, config, model, datahub):
        self.config = config
        self.model = model
        self.datahub = datahub
        self.ruler = Ruler()
        self.accelerator = config['accelerator']
        os.makedirs(os.path.dirname(self.config['ckpt_dir']), exist_ok=True)
        self.saved_model_ckpt = os.path.join(
            self.config['ckpt_dir'],
            get_file_name(self.config, suffix='.pth')
        )

    def _extract_and_compute(self, batch_data):
        if self.config['model'] == 'SymbolCD':
            student_id, question, q_matrix_line, _= batch_data
            # student_id: torch.Tensor = student_id
            # question: torch.Tensor = question
            # q_matrix_line: torch.Tensor = q_matrix_line
            if hasattr(self.model, 'module'):
                interaction_function=self.model.module.extractor.interaction_function
                
                pred= self.model.module.extractor.net(student_id,
                                            question,
                                            q_matrix_line,
                                            interaction_function)
            else:
                interaction_function=self.model.extractor.interaction_function
                
                pred= self.model.extractor.net(student_id,
                                            question,
                                            q_matrix_line,
                                            interaction_function)
            return pred
            
        student_id, exercise_id, q_mask, _ = batch_data

        if hasattr(self.model, 'module'):
            # If the model is wrapped in DataParallel or DistributedDataParallel
            student_ts, diff_ts, disc_ts, knowledge_ts, _ = self.model.module.extractor.extract(student_id, exercise_id, q_mask)
            out = self.model.module.inter_func.compute(student_ts=student_ts, diff_ts=diff_ts, disc_ts=disc_ts, q_mask=q_mask, knowledge_ts=knowledge_ts, other=_)

        else:
            # If the model is not wrapped (single GPU or CPU)
            student_ts, diff_ts, disc_ts, knowledge_ts, _ = self.model.extractor.extract(student_id, exercise_id, q_mask)
            out = self.model.inter_func.compute(student_ts=student_ts, diff_ts=diff_ts, disc_ts=disc_ts, q_mask=q_mask, knowledge_ts=knowledge_ts, other=_)

        return out

    def _train_one_step(self, batch_data, optimizer, loss_func):
        optimizer.zero_grad()
        if self.config['model'] == 'DCD':
            student_id, exercise_id, q_mask, _ = batch_data
            if hasattr(self.model, 'module'):
                loss_dict = self.model.module.get_loss_dict(student_id=student_id, exercise_id=exercise_id, r=_)
            else:
                loss_dict=self.model.get_loss_dict(student_id=student_id, exercise_id=exercise_id, r=_)
            loss = torch.hstack([i for i in loss_dict.values() if i is not None]).sum()
        else:
            out = self._extract_and_compute(batch_data)
            if isinstance(out, list):
                pred_r = out[0]
                aux_loss = out[1].get('extra_loss', 0)
            else:
                pred_r = out
                aux_loss = 0
            loss = loss_func(pred_r, batch_data[3]) + aux_loss
        loss.backward()
        optimizer.step()
        return loss.mean().item()
    
    def train_for_symbolcd(self, epoch_i):
        # for epoch_i in range(0, self.config['epoch']):
        if hasattr(self.model, 'module'):
            print("[Epoch {}]".format(epoch_i + 1))
            self.model.module.extractor.train(self.config['datahub'], 'train', epoch=self.config['para_epoch'], 
            lr=self.config['lr'], init=(self.config['epochs'] == 0), batch_size=self.config['train_batch_size'])
            print(f"The {epoch_i + 1}-th epoch extractor optimization complete")
            arguments = self.model.module.extractor.unpack()
            self.model.module.inter_func.update(*arguments)
            self.model.module.inter_func.train(self.config['datahub'], 'train', self.config['population_size'], self.config['ngen'], self.config['cxpb']
            , self.config['mutpb'], self.config['train_batch_size'])
            print(f"The {epoch_i + 1}-th epoch interaction function complete")
            self.model.module.extractor.update(self.model.module.inter_func.function(), str(self.model.module.inter_func))
        else:
            print("[Epoch {}]".format(epoch_i + 1))
            self.model.extractor.train(self.config['datahub'], 'train', epoch=self.config['para_epoch'], 
            lr=self.config['lr'], init=(self.config['epochs'] == 0), batch_size=self.config['train_batch_size'])
            print(f"The {epoch_i + 1}-th epoch extractor optimization complete")
            arguments = self.model.extractor.unpack()
            self.model.inter_func.update(*arguments)
            self.model.inter_func.train(self.config['datahub'], 'train', self.config['population_size'], self.config['ngen'], self.config['cxpb']
            , self.config['mutpb'], self.config['train_batch_size'])
            print(f"The {epoch_i + 1}-th epoch interaction function complete")
            self.model.extractor.update(self.model.inter_func.function(), str(self.model.inter_func))

            # self.model.score(self.config['datahub'], 'valid', valid_metrics, batch_size=batch_size, **kwargs)


    # def evaluate_for_symbolcd(self):
    #     self.model.score(datahub, valid_set_type, valid_metrics, batch_size=batch_size, **kwargs)
        # print("[Epoch {}]".format(epoch_i + 1))
        # self.extractor.train(self.config['datahub'], 'train', epoch=self.config['para_epoch'], 
        # lr=self.config['lr'], init=(self.config['epoch'] == 0), batch_size=self.config['train_batch_size'])
        # print(f"The {epoch_i + 1}-th epoch extractor optimization complete")
        # arguments = self.extractor.unpack()
        # self.inter_func.update(*arguments)
        # self.inter_func.train(self.config['datahub'], 'train', self.config['population_size'], self.config['ngen'], self.config['cxpb']
        # , self.config['mutpb'], self.config['train_batch_size'])
        # print(f"The {epoch_i + 1}-th epoch interaction function complete")
        # self.extractor.update(self.inter_func.function(), str(self.inter_func))

    def train(self, train_dataloader, val_dataloader):
        loss_func = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])

        self.model, optimizer, train_dataloader, val_dataloader = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader
        )
        self.config.pop('accelerator')
        self.accelerator.init_trackers(project_name="inscd", config=self.config)

        epoch_losses = []

        total_n_steps = get_total_steps(self.config, train_dataloader)
        n_epochs = np.ceil(total_n_steps / (len(train_dataloader) * self.accelerator.num_processes)).astype(int)
        best_epoch, best_val_score = 0, -1

        for epoch in range(n_epochs):
            if self.config['model'] == 'SymbolCD':
                self.train_for_symbolcd(epoch)
            else:
                self.model.train()
                if self.config['model'] == 'ORCDF':
                    if self.config['use_ddp']:
                        self.model.module.extractor.get_flip_graph()
                    else:
                        self.model.extractor.get_flip_graph()
                total_loss = 0.0
                train_progress_bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Training - [Epoch {epoch + 1}]")
                for batch_data in train_progress_bar:
                    loss = self._train_one_step(batch_data, optimizer, loss_func)
                    epoch_losses.append(loss)
                    total_loss += loss
                self.accelerator.log({"Loss/train_loss": total_loss / len(train_dataloader)}, step=epoch + 1)

            # Evaluation
            if (epoch + 1) % self.config['eval_interval'] == 0:
                all_results = self.evaluate(val_dataloader, set_type='valid')
                if self.accelerator.is_main_process:
                    for key in all_results:
                        formatted_value = round(all_results[key] * 100, 2) 
                        self.accelerator.log({f"Val_Metric/{key}": formatted_value}, step=epoch + 1)
                    formatted_metrics = ', '.join(f"{key}: {round(value * 100, 2)}%" for key, value in all_results.items())
                    print(f"Valid Metrics: {formatted_metrics}")

                val_score = all_results[self.config['val_metric']]
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch + 1
                    if self.accelerator.is_main_process:
                        if self.config['use_ddp']:  # unwrap model for saving
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            torch.save(unwrapped_model.state_dict(), self.saved_model_ckpt)
                        else:
                            torch.save(self.model.state_dict(), self.saved_model_ckpt)
                        print(f'[Epoch {epoch + 1}] Saved model checkpoint to {self.saved_model_ckpt}')
                else:
                    print(f'Patience for {epoch + 1 - best_epoch} Times')

                if self.config['patience'] is not None and epoch + 1 - best_epoch >= self.config['patience']:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

        print(f'Best epoch: {best_epoch}, Best val score: {best_val_score}')
        
    def evaluate(self, dataloader, set_type='test'):
        if self.config['model'] != 'SymbolCD':
            self.model.eval()
        #     self.model.extractor.eval()
        # else:
        all_results = defaultdict(list)
        val_progress_bar = tqdm(dataloader, total=len(dataloader), desc=f"Eval - {set_type}")

        for batch in val_progress_bar:
            with torch.no_grad():
                batch_data = [_.to(self.accelerator.device) for _ in batch]
                if self.config['model'] == 'DCD':
                    if hasattr(self.model, "module"):
                        out = self.model.module.get_pred(batch_data[0], batch_data[1])
                    else:
                        out = self.model.get_pred(batch_data[0], batch_data[1])
                else:
                    out = self._extract_and_compute(batch_data)
                    
                preds = out[0].detach() if isinstance(out, list) else out.detach()

                if self.config['use_ddp']:
                    # all_preds, all_labels = self.accelerator.gather_for_metrics((preds, batch[3])) Leaving Issues
                    results = self.ruler(self.model, self.datahub, set_type, preds.cpu(), batch[3].cpu(), self.config['metrics'], ddp=True)
                else:
                    preds = preds.cpu()
                    results = self.ruler(self.model, self.datahub, set_type, preds, batch[3].detach().cpu(), self.config['metrics']) 


                for key, value in results.items():
                    all_results[key].append(value)

        output_results = OrderedDict()
        for metric in self.config['metrics']:
            # print(all_results[metric])
            output_results[metric] = np.mean(all_results[metric])
        return output_results

    def end(self):
        """
        Ends the training process and releases any used resources
        """
        self.accelerator.end_training()
