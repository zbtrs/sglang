import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import warnings
from sglang.srt.managers.controller.peft_manager import PeftArgument,PeftTask,PeftManager
from peft import LoraConfig

warnings.filterwarnings("ignore")

def prepare_non_packed_dataloader(
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

    # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
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

def get_task():
    device_map = "auto"
    df = pd.read_csv("/data02/peft-inference/medquad.csv")

    torch.manual_seed(0)

    data = Dataset.from_pandas(pd.DataFrame(data=df))
    model_name = "/data02/hyf/llama2"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    torch.cuda.empty_cache()

    LORA_ALPHA = 16
    LORA_DROPOUT = 0.2
    LORA_R = 64
    peft_config = LoraConfig(
            lora_alpha= LORA_ALPHA,
            lora_dropout= LORA_DROPOUT,
            r= LORA_R,
            bias="none",
            task_type="CAUSAL_LM",
        )

    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10
    BATCH_SIZE = 4
    WEIGHT_DECAY = 0.001
    MAX_GRAD_NORM = 0.3
    gradient_accumulation_steps = 16
    STEPS = 1
    OPTIM = "adam"
    MAX_STEPS = 2

    OUTPUT_DIR = "./results"

    training_args = PeftArgument(
        output_dir= OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps= gradient_accumulation_steps,
        learning_rate= LEARNING_RATE,
        logging_steps= STEPS,
        num_train_epochs= NUM_EPOCHS,
        max_steps= MAX_STEPS,
    )

    peft_task = PeftTask(
        model_name=model_name,
        train_dataset=data,
        peft_config=peft_config,
        dataset_text_field="question",
        max_seq=500,
        args=training_args
    )
    
    return peft_task