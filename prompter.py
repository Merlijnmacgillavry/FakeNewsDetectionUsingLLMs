from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import os
from tqdm import tqdm
import utils
from datasets import load_dataset, Dataset
from multiprocess import set_start_method
from codecarbon import EmissionsTracker
from peft import PeftModel, PeftConfig

model_options = ["gemma-2b", "gemma-2-9b", "llama-3.1-8b-it", "mistral-0.2-7b-it"]
discrete_options= ["Pants on Fire", "False", "Mostly False", "Half True", "Mostly True", "True"]
binary_options = ["FAKE", "REAL"]
prompt_formats = {
    "binary": f'Instruction: "You are a binary news veracity classifier. Given an article you classify whether it is real or fake answer with "REAL" if it is real and "FAKE" if it is fake.\n\nArticle: %ARTICLE%\n\nClassification:',
    "discrete": f'Instruction: You are a discrete news veracity classfier. Given an article you classify whether it is real or fake by classifying it as one of the following options (From Fake to Real): ["Pants on Fire", "False", "Mostly False", "Half True", "Mostly True", "True"]\n\nArticle: %ARTICLE%\n\nClassification: ',
    "cot": f'Instruction: You are a binary news veracity classifier. Given an article you classify whether it is real or fake answer with "REAL" if it is real and "FAKE" if it is fake. Before classifying first reason about the 3 following questions regarding the article and take that reasoning into account when deciding on your classification" Questions: 1. Does the claim align with verifiable facts? 2. Can the statement be understood in different ways? 3. Is there evidence provided by the source to substantiate the claim? Article: %ARTICLE% Classification:',
    "percentage": f'Instruction: "You are a percentage based news veracity classifier. Given an article you classify whether it is real or fake answer between 0% and 100%, where 100% is definetely fake and 0% is definetely real.\n\nArticle: %ARTICLE%\n\nPercentage:',
}

class Prompter:
    input_path_option = {
        "dest": "input",
        "type": str,
        "nargs": 1,
        "metavar": "<INPUT PATH>",
        "help": "The path to the entries to be prompted",
    }

    output_path_option = {
        "dest": "output",
        "type": str,
        "nargs": 1,
        "metavar": "<OUTPUT PATH>",
        "help": "The path to save the results to",
    }

    model_input = {
        "dest": "model",
        "type": str,
        "nargs": 1,
        "metavar": "<MODEL>",
        "help": f"The LLM model to use, options are: {model_options}",
        "choices": model_options,
    }

    prompt_type = {
        "dest": "prompt_type",
        "type": str,
        "nargs": 1,
        "metavar": "<PROMPT TYPE>",
        "help": f"Which kind of prompt to use, options are: {list(prompt_formats.keys())}",
        "choices": list(prompt_formats.keys()),
    }

    finetuned = {
        "dest": "finetuned",
        "type": bool,  # Boolean flag
        "help": "Whether to use a finetuned model (optional, default is False)",
        "nargs": 1
    }

    def add_parser(self, sub_parsers):
        prompt_parse = sub_parsers.add_parser(
            "prompt", help="Prompt LLMS to classify veracity of news content"
        )
        prompt_parse.add_argument(**self.input_path_option)
        prompt_parse.add_argument(**self.output_path_option)
        prompt_parse.add_argument(**self.model_input)
        prompt_parse.add_argument(**self.prompt_type)
        prompt_parse.add_argument("--finetuned", action="store_true", help="Whether the model is finetuned")  # Add the action="store_true" flag
        try:
            prompt_parse.set_defaults(
                func=lambda args: self.main(
                    args.input[0],
                    args.output[0],
                    args.model[0],
                    args.prompt_type[0],
                    args.finetuned
                )
            )
        except ValueError as error:
            self.logger.error(f"Value error in preprocessing command: {error}")

    def __init__(self, logger) -> None:
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self.prompt_prefix = None
        self.force_words = False
        self.max_new_tokens = 10
        self.finetuned = False



    def main(self, input_path: str, output_path: str, model: str, prompt_type: str, finetuned):
        if (finetuned):
            self.finetuned = True
        self.prompt_prefix = prompt_formats[prompt_type]
        self.prompt_type = prompt_type
       
        output_path_file_name = f"predictions_{model}_{prompt_type}_{input_path.split('/')[-1][0:-4]}.csv"
        if(finetuned):
            output_path_file_name = f"predictions_{model}_{prompt_type}_{input_path.split('/')[-1][0:-4]}_finetuned.csv"

        output_path = os.path.join(output_path, output_path_file_name)
        # self.logger.info(
        #     f"Starting prompting with input_path: {input_path}, output_path: {output_path}, model: {model}, prompt: {prompt_prefix}"
        # )
        self.logger.info(f"Loading model: {model}")
        self.load_model(model)
       
        if self.model_input == None or self.tokenizer == None:
            self.logger.error(f"Failed to load model: {model} and tokenizer")
            return
        self.setup_force_words()
        self.logger.info(f"Loading dataset from : {input_path}")
        df = pd.DataFrame([])
        input_exists = os.path.exists(input_path)
        if input_path:
            df = pd.read_csv(input_path)
        if df.empty:
            self.logger.error(f"Failed to load dataset from: {input_path}")
        self.logger.info("Succesfully loaded dataset")
        if(self.prompt_type == 'cot'):
            self.max_new_tokens = 100
        dataset = Dataset.from_pandas(df)

        # dataset = dataset.select(range(15000,15100))
        # print(dataset)
        # contents = [row["content"] for i, row in df.iterrows()]
        # predictions = []
        # for content in tqdm(contents, total=len(contents)):
        #     prediction = self.prompt_content(content)
        #     predictions.append(prediction)
        # df[f"prediction_{model}"] = predictions
        set_start_method("spawn")
        tracker = EmissionsTracker(project_name=f"Prompting: {output_path_file_name}")
        tracker.start()
        try: 
            updated_dataset = dataset.map(
                self.gpu_computation,
                batched=True,
                batch_size=1,
                with_rank=True,
                num_proc=torch.cuda.device_count(),  # one process per GPU
            )
            updated_dataset.to_csv(output_path)
        finally: 
            tracker.stop()

        # df.to_csv(output_path, mode="a")

    def gpu_computation(self, batch, rank):
        # Move the model on the right GPU if it's not there already
        device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
        self.model.to(device)

        # Your big GPU call goes here, for example:
        input_texts = [
            self.prompt_prefix.replace("%ARTICLE%", content[0:2000])
            for content in batch["content"]
        ]
        model_inputs = self.tokenizer(
            input_texts, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if self.prompt_type == 'percentage':
            answers = [str(output).split('Percentage:')[1].strip() for output in decoded_outputs]
        else:
            answers = [str(output).split('Classification:')[1].strip() for output in decoded_outputs]
        predictions = [self.answer_to_prediction(answer) for answer in answers ]
        batch['prediction'] = predictions
        return batch

    def answer_to_prediction(self, answer: str, ):
        lowered = answer.lower()
        match self.prompt_type:
            case 'binary':
                real_in = 'real' in lowered
                fake_in = 'fake' in lowered or 'false' in lowered
                if(real_in and not fake_in):
                    return 0

                if(fake_in and not real_in):
                    return 1
            case 'discrete':
                for option in discrete_options:
                    if option.lower() in lowered:
                        return option
                return "-1"
            case 'cot':
                real_in = 'real' in lowered
                fake_in = 'fake' in lowered
                if(real_in and not fake_in):
                    return 0

                if(fake_in and not real_in):
                    return 1
                return -1
            
            case 'percentage':
                if('%' not in lowered):
                    return str(-1)
                percentageString = lowered.split('%')[0]
                if(' ' not in percentageString):
                    return percentageString
                else: 
                    return percentageString.split(' ')[1]
        return -1


    def prompt_content(self, content: str):
        input_text = self.prompt_prefix + content
        input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=250)
        outputs = self.model.generate(**input_ids, max_length=5000)
        return self.tokenizer.decode(outputs[0])

    def load_model(self, model: str):
        match model:
            case "gemma-2b":
                if(self.finetuned):
                    self.logger.info('Using finetuned model...')
                    peft_model_id = utils.get_repo(model)
                    config = PeftConfig.from_pretrained(peft_model_id)
                    self.model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.bfloat16)
                    self.model = PeftModel.from_pretrained(self.model, peft_model_id).eval()
                    self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
                else: 
                    self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
                    self.model = AutoModelForCausalLM.from_pretrained(
                    "google/gemma-2b-it", torch_dtype=torch.bfloat16
                ).eval()
                
            case  "gemma-2-9b":
                if(self.finetuned):
                    self.logger.info('Using finetuned model...')
                    peft_model_id = utils.get_repo(model)
                    config = PeftConfig.from_pretrained(peft_model_id)
                    self.model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.bfloat16)
                    self.model = PeftModel.from_pretrained(self.model, peft_model_id).eval()
                    self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
                else: 
                    self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        "google/gemma-2-9b-it", torch_dtype=torch.bfloat16
                    ).eval()
            case "llama-3.1-8b-it":
                if(self.finetuned):
                    self.logger.info('Using finetuned model...')
                    peft_model_id = utils.get_repo(model)
                    config = PeftConfig.from_pretrained(peft_model_id)
                    self.model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.bfloat16)
                    self.model = PeftModel.from_pretrained(self.model, peft_model_id).eval()
                    self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
                else: 
                    self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
                    self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16)
            case "mistral-0.2-7b-it":
                if(self.finetuned):
                    self.logger.info('Using finetuned model...')
                    peft_model_id = utils.get_repo(model)
                    config = PeftConfig.from_pretrained(peft_model_id)
                    self.model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.bfloat16)
                    self.model = PeftModel.from_pretrained(self.model, peft_model_id).eval()
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
                else: 
                    self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16).eval()
                    self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
            case _:
                pass
        return

    def setup_force_words(self):
        match self.prompt_type:
            case 'binary':
                self.force_words = True
                self.force_ids = self.tokenizer(binary_options, add_special_tokens=False).input_ids
            case 'discrete':
               self.force_words = True
               self.force_ids = self.tokenizer(discrete_options, add_special_tokens=False).input_ids
