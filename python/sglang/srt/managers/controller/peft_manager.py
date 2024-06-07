import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from torch.utils.data import DataLoader
from transformers import default_data_collator
from datasets import load_dataset
from tqdm import tqdm
import itertools

class PeftConfig:
    def __init__(self, model_name_or_path, peft_type, task_type, inference_mode, r, lora_alpha, lora_dropout):
        self.model_name_or_path = model_name_or_path
        self.peft_type = peft_type
        self.task_type = task_type
        self.inference_mode = inference_mode
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

    def get_config(self):
        return LoraConfig(
            task_type=self.task_type,
            inference_mode=self.inference_mode,
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
        )

class PeftTask:
    def __init__(self, peft_config, text_column, label_column, max_length, lr, batch_size, num_epochs, train_dataset,eval_dataset,dataset):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model_name_or_path = peft_config.model_name_or_path
        self.peft_config = peft_config

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path)
        self.model = get_peft_model(self.model, self.peft_config.get_config())
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model.to(self.device)
        
        self.current_epoch = 0
        self.current_step = 0
        self.iterator_position = 0

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_dataloader = None
        self.eval_dataloader = None
        self.train_iterator = None
        self.dataset = dataset
        self.optimizer = None
        self.lr_scheduler = None

    # def prepare_data(self):
    #     self.train_dataset, self.eval_dataset, self.train_dataloader, self.eval_dataloader, self.train_iterator, self.dataset = self.prepare_data_function(self)

    # def setup_optimizer_and_scheduler(self):
    #     self.optimizer, self.lr_scheduler = self.setup_optimizer_function(self)

    def post_init(self):
        self.train_dataloader = DataLoader(
            self.train_dataset,shuffle=True, collate_fn=default_data_collator, batch_size=self.batch_size, pin_memory=True
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset, collate_fn=default_data_collator, batch_size=self.batch_size, pin_memory=True
        )
        self.train_iterator = iter(self.train_dataloader)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(self.train_dataloader) * self.num_epochs),
        )


    def save_state(self):
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'iterator_position': self.iterator_position,
            'current_batch': next(self.train_iterator, None)  
        }
        return state

    def load_state(self, state):
        if state is not None:
            self.model.load_state_dict(state['model_state_dict'])
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(state['lr_scheduler_state_dict'])
            self.current_epoch = state['current_epoch']
            self.current_step = state['current_step']
            self.iterator_position = state['iterator_position']
            self.train_iterator = iter(self.train_dataloader)
            if state['current_batch'] is not None:
                self.train_iterator = itertools.chain([state['current_batch']], self.train_iterator)

    def train_step(self, num_steps=1):
        if num_steps <= 0:
            raise ValueError("Number of steps must be a positive integer.")
        
        self.model.train()
        total_loss = 0
        for _ in range(num_steps):
            try:
                batch = next(self.train_iterator)
                self.iterator_position += 1
            except StopIteration:
                self.current_epoch += 1
                self.train_iterator = iter(self.train_dataloader)
                self.iterator_position = 0
                batch = next(self.train_iterator)
                self.iterator_position += 1
                
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.current_step += 1
        return total_loss

    def evaluate(self):
        self.model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(self.eval_dataloader)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                self.tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )
        
        eval_epoch_loss = eval_loss / len(self.eval_dataloader)
        print(f"{eval_epoch_loss=}")
        correct = 0
        total = 0
        for pred, true in zip(eval_preds, self.dataset["validation"]["text_label"]):
            if pred.strip() == true.strip():
                correct += 1
            total += 1
        accuracy = correct / total * 100
        print(f"{accuracy=} % on the evaluation dataset")
        print(f"{self.dataset['validation']['text_label'][:10]=}")

    def unload_from_gpu(self):
        self.model.cpu()
        torch.cuda.empty_cache()
        print("Model and data unloaded from GPU.")

class PeftManager:
    def __init__(self):
        self.task_queue = []

    def add_task(self, peft_task):
        self.task_queue.append(peft_task)
        print(f"Task added. Queue length: {len(self.task_queue)}")

    async def run_next_step(self):
        if self.task_queue:
            task_state = self.task_queue.pop(0)
            task = task_state["task"]
            state = task_state["state"]

            print(f"{task.current_step}, {task.num_epochs}, {len(task.train_dataloader)}")
            
            while task.current_step >= task.num_epochs * len(task.train_dataloader):
                task.evaluate()
                task.unload_from_gpu()
                print(f"Task with current_step {task.current_step} exceeds total steps, skipping.")
                if self.task_queue:
                    task_state = self.task_queue.pop(0)
                    task = task_state["task"]
                    state = task_state["state"]
                else:
                    return

            task.load_state(state)
            try:
                loss = task.train_step(num_steps=1)
                print(f"Step completed. Loss: {loss}")

                state = task.save_state()
                task_state = {
                    "task": task,
                    "state": state
                }

                self.task_queue.insert(0, task_state)
                # task.unload_from_gpu()
                print(f"Task state saved and added back to the queue. Queue length: {len(self.task_queue)}")
                return loss
            except Exception as e:
                print(f"Error during training step: {e}")
                task.unload_from_gpu()
