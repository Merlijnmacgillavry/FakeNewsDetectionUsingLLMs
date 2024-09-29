import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import transformers
from datasets import load_dataset, Dataset
import pandas as pd
import utils

model_options = ["gemma-2b", "gemma-2-9b", "llama-3.1-8b-it", "mistral-0.2-7b-it"]


class FineTuner:
    input_path_option = {
        "dest": "input",
        "type": str,
        "nargs": 1,
        "metavar": "<INPUT PATH>",
        "help": "The path to the entries to be prompted",
    }

    model_input = {
        "dest": "model",
        "type": str,
        "nargs": 1,
        "metavar": "<MODEL>",
        "help": f"The LLM model to use, options are: {model_options}",
        "choices": model_options,
    }

    def __init__(self, logger) -> None:
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self.repo = None

    def add_parser(self, sub_parsers):
        prompt_parse = sub_parsers.add_parser(
            "finetune", help="Finetune LLMs using LoRa finetuning"
        )
        prompt_parse.add_argument(**self.input_path_option)
        prompt_parse.add_argument(**self.model_input)
        try:
            prompt_parse.set_defaults(
                func=lambda args: self.main(
                    args.input[0],
                    args.model[0],
                )
            )
        except ValueError as error:
            self.logger.error(f"Value error in finetuning command: {error}")

    def load_model(self, model: str):
        match model:
            case "gemma-2b":
                self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
                self.model = AutoModelForCausalLM.from_pretrained(
                    "google/gemma-2b-it", load_in_4bit=True, device_map="auto"
                )
            case "gemma-2-9b":
                self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
                self.model = AutoModelForCausalLM.from_pretrained(
                    "google/gemma-2-9b-it", load_in_4bit=True, device_map="auto"
                )
            case "llama-3.1-8b-it":
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Meta-Llama-3.1-8B-Instruct"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    load_in_4bit=True,
                    device_map="auto",
                )
            case "mistral-0.2-7b-it":
                self.model = AutoModelForCausalLM.from_pretrained(
                    "mistralai/Mistral-7B-Instruct-v0.2",
                    load_in_4bit=True,
                    device_map="auto",
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "mistralai/Mistral-7B-Instruct-v0.2"
                )
            case _:
                pass
        return

    def main(self, input_path: str, model: str):
        self.load_model(model)
        self.repo = utils.get_repo(model)
        print(self.repo)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.tokenizer.get_vocab().get(
                self.tokenizer.pad_token, self.tokenizer.eos_token_id
            )

        self.model.resize_token_embeddings(len(self.tokenizer))

        for param in self.model.parameters():
            param.requires_grad = False  # freeze the self.model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

        self.model.gradient_checkpointing_enable()  # reduce number of stored activations
        self.model.enable_input_require_grads()

        class CastOutputToFloat(nn.Sequential):
            def forward(self, x):
                return super().forward(x).to(torch.float32)

        self.model.lm_head = CastOutputToFloat(self.model.lm_head)

        config = LoraConfig(
            r=16,  # attention heads
            lora_alpha=32,  # alpha scaling
            # target_modules=["q_proj", "v_proj"],  # if you know the
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",  # set this for CLM or Seq2Seq
        )

        self.model = get_peft_model(self.model, config)
        train_path = os.path.join(input_path, "train.csv")
        val_path = os.path.join(input_path, "val.csv")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)

        # Process the training data
        train_dataset = train_dataset.map(training_data)
        train_data = train_dataset.map(
            lambda samples: self.tokenizer(samples["training_sample"]), batched=True
        )
        train_data = train_data.remove_columns(
            [
                "index",
                "topic",
                "label",
                "length",
                "sentiment",
                "content",
                "Unnamed: 0",
                "training_sample",
            ]
        )

        # Process the validation data
        val_dataset = val_dataset.map(training_data)
        val_data = val_dataset.map(
            lambda samples: self.tokenizer(samples["training_sample"]), batched=True
        )
        val_data = val_data.remove_columns(
            [
                "index",
                "topic",
                "label",
                "length",
                "sentiment",
                "content",
                "Unnamed: 0",
                "training_sample",
            ]
        )

        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=train_data,
            eval_dataset=val_data,  # Add the validation dataset here
            args=transformers.TrainingArguments(
                per_device_train_batch_size=8,
                gradient_accumulation_steps=4,
                warmup_steps=100,
                max_steps=1000,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=10,
                evaluation_strategy="steps",  # Evaluate on the validation set every 'logging_steps'
                output_dir="outputs",
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False
            ),
        )
        self.model.config.use_cache = (
            False  # silence the warnings. Please re-enable for inference!
        )
        trainer.train()
        self.model.push_to_hub(
            self.repo,
            use_auth_token=True,
            commit_message="basic training",
            private=True,
        )


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def label_to_veracity(label):
    if label == 0:
        return "real"
    return "fake"


def training_data(dataset):
    template = f'Instruction: "You are a binary news veracity classifier. Given an article you classify whether it is real or fake answer with "REAL" if it is real and "FAKE" if it is fake.\n\nArticle: %ARTICLE%\n\nClassification:'
    dataset["training_sample"] = template.replace(
        "%ARTICLE%", dataset["content"][0:500]
    ) + label_to_veracity(dataset["label"])
    return dataset
