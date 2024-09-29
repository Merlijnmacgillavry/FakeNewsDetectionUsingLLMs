import joblib
import pandas as pd
import os
import utils
from codecarbon import EmissionsTracker
from tqdm import tqdm
from trainer import get_features, BERTClassifier
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import BertTokenizer, BertModel


class BaselineClassifier:
    path_option = {
        "dest": "input",
        "type": str,
        "nargs": 1,
        "metavar": "<PATH>",
        "help": "path to folder containing the test dataset to make classifications on",
    }

    model_option = {
        "dest": "model",
        "type": str,
        "nargs": 1,
        "metavar": "<Model>",
        "help": "path to the model used for classification",
    }

    output_option = {
        "dest": "output",
        "type": str,
        "nargs": 1,
        "metavar": "<Output>",
        "help": "path of directory to save the predictions to",
    }

    def __init__(self, logger) -> None:
        self.logger = logger
        classify_options = self.get_classify_functions()
        self.classify_option = {
            "dest": "classify",
            "type": str,
            "nargs": 1,
            "metavar": "<Classify>",
            "help": "The type of baseline to classify (Either 'FeatureBased' or 'PromptBased')",
            "choices": classify_options,
        }

    def get_classify_functions(self):
        methods = [method for method in dir(self) if not method.startswith("_")]
        available_options = [
            method for method in methods if method.startswith("classify_")
        ]
        return map(lambda x: x.split("classify_")[1], available_options)

    def add_parser(self, sub_parsers):
        classify_parse = sub_parsers.add_parser(
            "classify",
            help="Evaluate a baseline classifier by classifying a test dataset",
        )
        classify_parse.add_argument(**self.classify_option)
        classify_parse.add_argument(**self.path_option)
        classify_parse.add_argument(**self.model_option)
        classify_parse.add_argument(**self.output_option)
        classify_parse.set_defaults(
            func=lambda args: self._classify(
                args.classify[0], args.input[0], args.model[0], args.output[0]
            )
        )

    def _classify(self, type: str, path: str, model: str, output_path: str):
        method_name = f"classify_{type}"
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            return method(path, model, output_path)
        else:
            raise ValueError(f"Invalid type: {type}")

    def classify_FeatureBased(self, path: str, model, output_path: str):
        self.logger.info(
            f"Classifying using FeatureBased method with model {model} on dataset {path}"
        )
        tracker = EmissionsTracker(project_name="FeatureBased Classifier")
        tracker.start()
        df_test = pd.read_csv(path)
        predictions = classify_dataset_fb(df_test, model)
        output_path = os.path.join(output_path, "predictions_RF_binary.csv")
        utils.create_file_with_directories(output_path, self.logger)
        predictions.to_csv(output_path, index=False)
        tracker.stop()

    def classify_PromptBased(self, path: str, model, output_path: str):
        self.logger.info(
            f"Classifying using PromptBased method with model {model} on dataset {path}"
        )
        tracker = EmissionsTracker(project_name="PromptBased Classifier")
        tracker.start()
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        df_test = pd.read_csv(path)
        if "content" not in df_test.columns:
            raise ValueError("The dataset must contain a 'content' column")
        texts = df_test["content"].to_list()
        dataset = TextClassificationDataset(texts, tokenizer, max_length=500)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        predictions = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model_bert(model, "bert-base-uncased", 2, device)
        for batch in tqdm(dataloader, total=len(dataloader)):
            inputs_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.no_grad():
                outputs = model(inputs_ids, attention_mask)
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())

        df_test["prediction"] = predictions
        output_path = os.path.join(output_path, "predictions_BERT_binary.csv")
        utils.create_file_with_directories(output_path, self.logger)
        df_test.to_csv(output_path, index=False)
        self.logger.info(f"Predictions saved to {output_path}")
        tracker.stop()


def make_predictions_fb(clf, df_test, final_features):
    X_test = df_test[final_features].values
    predictions = clf.predict(X_test)

    # Save predictions to CSV
    df_test["prediction"] = predictions
    new_columns = {
        "Unnamed: 0": "index",
        "topic": "topic",
        "label": "label",
        "length": "length",
        "sentiment": "sentiment",
        "content_x": "content",
        "smog_index": "smog_index",
        "flesch_reading_ease": "flesch_reading_ease",
        "flesch_kincaid_grade_level": "flesch_kincaid_grade_level",
        "coleman_liau_index": "coleman_liau_index",
        "gunning_fog_index": "gunning_fog_index",
        "ari_index": "ari_index",
        "lix_index": "lix_index",
        "dale_chall_score": "dale_chall_score",
        "dale_chall_known_fraction": "dale_chall_known_fraction",
        "prediction": "prediction",
    }

    # Create the new DataFrame
    new_df = df_test[list(new_columns.keys())].rename(columns=new_columns)
    return new_df


# Example usage
def load_model_bert(model_path, bert_model_name, num_classes, device):
    model = BERTClassifier(bert_model_name, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def classify_dataset_fb(df_test, model_path):
    final_features = get_features(df_test)
    clf_loaded = load_fb_classifier(model_path)

    # Make predictions on a test dataset and save to CSV
    predictions = make_predictions_fb(clf_loaded, df_test, final_features)
    return predictions


def load_fb_classifier(filename):
    return joblib.load(filename)


class TextClassificationDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
