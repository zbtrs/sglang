import gc

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from importlib.util import find_spec
import warnings
import torch
import datasets
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from peft import LoraConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Mapping
# from utils import print_memory_usage
import queue
import time

def is_peft_available() -> bool:
    return find_spec("peft") is not None
if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

class PeftArgument:
    def __init__(self,output_dir,batch_size,gradient_accumulation_steps,learning_rate,logging_steps,num_train_epochs,max_steps):
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.logging_steps = logging_steps
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        
class PeftTask:
    def __init__(self,model_name,train_dataset,peft_config: LoraConfig,dataset_text_field,max_seq,args: PeftArgument):
        if isinstance(model_name, str):
            self.model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16) # TODO: add model_init_kwargs
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise ValueError(
                "You should pass a model_name using str"
            )
        self.peft_config = peft_config    
        self.device = "cuda"
        if not isinstance(self.model, PeftModel):
            self.model = get_peft_model(self.model, self.peft_config)
            self.model.print_trainable_parameters()
        self.dataset_text_field = dataset_text_field
        self.max_seq = max_seq
        self.args = args
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        self.model = self.model.to(device=self.device)
        self.batch_size = self.args.batch_size
        self.train_dataset = self.__prepare_non_packed_dataloader(
            tokenizer=self.tokenizer,
            dataset=train_dataset,
            dataset_text_field=self.dataset_text_field,
            max_seq_length=self.max_seq,
            batch_size=self.batch_size
        )
        self.data_loader = self.get_train_dataloader()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(self.data_loader) * self.args.num_train_epochs),
        )
        
    def __prepare_non_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        batch_size,
        formatting_func=None,
        add_special_tokens=True,
        remove_unused_columns=True,
    ):
        use_formatting_func = formatting_func is not None and dataset_text_field is None

        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field] if not use_formatting_func else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

        signature_columns = ["input_ids", "labels", "attention_mask"]

        extra_columns = list(set(dataset.column_names) - set(signature_columns))

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with the default collator and yield to errors. If you want to "
                f"inspect dataset other columns (in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the default collator and create your own data collator in order to inspect the unused dataset columns."
            )

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names if remove_unused_columns else None,
            num_proc=1,
            batch_size=batch_size,
        )

        return tokenized_dataset

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self.train_dataset
        else:
            raise ValueError("Trainer: training requires a train_dataset.")

        dataloader_params = {
            "batch_size": self.batch_size,
            "collate_fn": data_collator,
            "num_workers": 1,
            "pin_memory": True,
            "persistent_workers": 1,
        }

        if not isinstance(train_dataset, IterableDataset):
            dataloader_params["sampler"] = RandomSampler(self.train_dataset)
            dataloader_params["drop_last"] = False

        return DataLoader(train_dataset, **dataloader_params)
    
    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            return data.to(**kwargs)
        return data
    
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
            
        return inputs
    
    def compute_loss(self,inputs,return_outputs=False):
        outputs = self.model(**inputs)
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss
    
    def train(self):
        pass
        # print_memory_usage("before train")
        # tr_loss = torch.tensor(0.0).to(self.device)
        # self._total_loss_scalar = 0.0
        # self.model.zero_grad()
        # for epoch in range(self.args.num_train_epochs):
        #     epoch_iterator = self.data_loader
        #     for step,inputs in enumerate(epoch_iterator):            
        #         self.model.train()

        #         inputs = self._prepare_inputs(inputs)
        #         loss = self.compute_loss(inputs=inputs)
        #         print_memory_usage("after train")

        #         del inputs
        #         loss.backward()
        #         print_memory_usage("after backward")

        #         tr_loss += loss.detach()
        #         self.optimizer.step()
        #         print_memory_usage("after optimizer")
        #         self.lr_scheduler.step()
        #         self.optimizer.zero_grad()
        #         print(f"{step}, {tr_loss}, {loss}")
                
class PeftManager:
    def __init__(self):
        self.task_queue = queue.Queue()

    def add_task(self, task: PeftTask):
        self.task_queue.put(task)

    async def train_step(self):
        if self.task_queue.empty():
            print("No tasks in the queue.")
            return

        start_time = time.time()
        task = self.task_queue.get()
        task.model.train()

        try:
            epoch_iterator = iter(task.data_loader)
            inputs = next(epoch_iterator)
            inputs = task._prepare_inputs(inputs)
            forward_start_time = time.time()
            loss = task.compute_loss(inputs=inputs)
            forward_end_time = time.time()
            print(f"forward time:{forward_end_time - forward_start_time}")
            backward_start_time = time.time()
            loss.backward()
            backward_end_time = time.time()
            print(f"backward time:{backward_end_time - backward_start_time}")
            print(f"{loss}")
            task.optimizer.step()
            task.lr_scheduler.step()
            task.optimizer.zero_grad()
            self.task_queue.put(task)
        except StopIteration:
            print("Finished one epoch for a task.")
            self.release_resources(task)
        # except Exception as e:
        #     print(f"Error during training step: {e}")
        end_time = time.time()
        print(f"peft time:{end_time - start_time}")

    def release_resources(self, task: PeftTask):
        del task.model
        del task.optimizer
        del task.lr_scheduler
        del task.data_loader

        gc.collect()
        torch.cuda.empty_cache()

    def has_tasks(self):
        return not self.task_queue.empty()