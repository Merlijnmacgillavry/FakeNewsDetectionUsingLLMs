import pandas as pd
import os
import multiprocessing as mp
import utils

# LIWC imports
from codecarbon import EmissionsTracker

from collections import Counter
import nltk
from tqdm import tqdm

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
from nltk.corpus import stopwords

stops = stopwords.words("english")
from nltk.tokenize import RegexpTokenizer
from nltk import sent_tokenize
import re
import liwc

# Remaining Features imports
import feature_based.readability as readability
import feature_based.stylistic_features as stylistic_features
import feature_based.emotion_features as emotion_features

# Classification
from sklearn.ensemble import RandomForestClassifier
import feature_based.step_5_classification as classification

import numpy as np
from numpy.random import seed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
import joblib  # For saving the model
import warnings

# BERT imports
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizer,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


parse, category_names = liwc.load_token_parser("./external_files/LIWC2015_English.dic")


class Trainer:
    path_option = {
        "dest": "input",
        "type": str,
        "nargs": 1,
        "metavar": "<PATH>",
        "help": "path to folder containing the train, validate and test datasets",
    }

    def __init__(self, logger) -> None:
        self.logger = logger
        train_options = self.get_train_functions()
        self.train_option = {
            "dest": "train",
            "type": str,
            "nargs": 1,
            "metavar": "<Train>",
            "help": "The type of the train (Either 'FeatureBased' or 'PromptBased')",
            "choices": train_options,
        }

    def get_train_functions(self):
        methods = [method for method in dir(self) if not method.startswith("_")]
        available_options = [
            method for method in methods if method.startswith("train_")
        ]
        return map(lambda x: x.split("train_")[1], available_options)

    def add_parser(self, sub_parsers):
        train_parse = sub_parsers.add_parser(
            "train",
            help="Train a baseline classifier with a training and validation dataset",
        )
        train_parse.add_argument(**self.train_option)
        train_parse.add_argument(**self.path_option)

        train_parse.set_defaults(
            func=lambda args: self._train(args.train[0], args.input[0])
        )

    def _train(self, type: str, path: str):
        method_name = f"train_{type}"
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            return method(path)
        else:
            raise ValueError(f"Invalid type: {type}")

    def train_FeatureBased(self, path: str):
        tracker = EmissionsTracker(project_name=f"Training FeatureBased")
        tracker.start()
        path_train = os.path.join(path, "train.csv")
        path_val = os.path.join(path, "val.csv")
        path_test = os.path.join(path, "test.csv")

        trained_dataset = self.prepare_featurebased_dataset(path_train, "train")
        validated_dataset = self.prepare_featurebased_dataset(path_val, "validate")
        test_dataset = self.prepare_featurebased_dataset(path_test, "test")
        self.save_classifier(trained_dataset, validated_dataset, test_dataset)

        tracker.stop()
        # tracker = EmissionsTracker(project_name=f"Classifying FeatureBased")
        # tracker.start()
        # classify_dataset(test_dataset, trained_dataset)
        # tracker.stop()

    def train_PromptBased(self, path: str):
        tracker = EmissionsTracker(project_name=f"Training PromptBased")
        tracker.start()
        train_texts, train_labels, val_texts, val_labels = (
            self.prepare_promptbased_dataset(path)
        )
        bert_model_name = "bert-base-uncased"
        num_classes = 2
        max_length = 128
        batch_size = 16
        num_epochs = 4
        learning_rate = 2e-5

        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        train_dataset = TextClassificationDataset(
            train_texts, train_labels, tokenizer, max_length
        )
        val_dataset = TextClassificationDataset(
            val_texts, val_labels, tokenizer, max_length
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BERTClassifier(bert_model_name, num_classes).to(device)

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            train(model, train_dataloader, optimizer, scheduler, device)
            accuracy, report = evaluate(model, val_dataloader, device)
            print(f"Validation Accuracy: {accuracy:.4f}")
            print(report)

        output_path = os.path.join("./models", "BERT_classifier.pth")
        utils.create_file_with_directories(output_path, self.logger)
        torch.save(model.state_dict(), output_path)

        tracker.stop()

    def prepare_promptbased_dataset(self, path: str):
        path_train = os.path.join(path, "train.csv")
        path_val = os.path.join(path, "val.csv")
        train_texts, train_labels = load_data(path_train)
        val_texts, val_labels = load_data(path_val)
        return train_texts, train_labels, val_texts, val_labels

    def prepare_featurebased_dataset(self, path: str, name: str):
        intermediate_steps_base_path = (
            f"./data/intermediate_steps/train/FeatureBased/{name}"
        )

        self.logger.info(
            f"starting train evaluation featuere based... with path: {path}"
        )
        self.logger.info("Starting step 1: LIWC features:")
        self.logger.info("Checking whether step is already calculated before...")
        file_exists = os.path.exists(path)
        if not file_exists:
            self.logger.error(f"File at location: {path}, does not exist!")
            return
        df = pd.read_csv(path)
        df_liwc = pd.DataFrame([])
        liwc_intermediate_path = os.path.join(intermediate_steps_base_path, "liwc.csv")
        liwc_exists = os.path.exists(liwc_intermediate_path)
        if liwc_exists:
            self.logger.info("Liwc step already completed")
            df_liwc = pd.read_csv(liwc_intermediate_path)
        if df_liwc.empty:
            df_liwc = compute_liwc_from_dict(df, "content")
            df_liwc.insert(loc=0, column="index", value=range(len(df)))
            utils.create_file_with_directories(liwc_intermediate_path, self.logger)
            df_liwc.to_csv(liwc_intermediate_path, mode="a", index=False)
        self.logger.info("Succesfully completed Liwc step")
        self.logger.info(
            "Starting step 2: Complexity, Emotion and Stylistic features..."
        )
        df_remaining_features = pd.DataFrame([])
        remaining_features_intermediate_path = os.path.join(
            intermediate_steps_base_path, "remaining_features.csv"
        )
        no_liwc_exists = os.path.exists(remaining_features_intermediate_path)
        if no_liwc_exists:
            self.logger.info("Remaining features already completed")
            df_remaining_features = pd.read_csv(remaining_features_intermediate_path)
        if df_remaining_features.empty:
            df_remaining_features = extract_complexity_emotion_stylistic_features(
                df, "content", self.logger
            )
            utils.create_file_with_directories(
                remaining_features_intermediate_path, self.logger
            )
            df_remaining_features.to_csv(
                remaining_features_intermediate_path, mode="a", index=False
            )
        self.logger.info("Succesfully completed Remaining features step")
        self.logger.info("Starting Step 3: Combining no_liwc and liwc dataframes...")
        df_merged = pd.DataFrame([])
        merged_intermediate_path = os.path.join(
            intermediate_steps_base_path, "merged.csv"
        )
        merged_exists = os.path.exists(merged_intermediate_path)
        if merged_exists:
            self.logger.info("Merging dataframe already completed")
            df_merged = pd.read_csv(merged_intermediate_path)
        if df_merged.empty:
            df_merged = merge_liwc_and_remaining_features(
                df_liwc, df_remaining_features
            )
            utils.create_file_with_directories(merged_intermediate_path, self.logger)
            df_merged.to_csv(merged_intermediate_path, mode="a", index=False)
        return df_merged

    def save_classifier(self, df_train, df_validate, df_test):
        # Define the features you want to use
        final_features = get_features(df_train)  # Assuming get_features is defined

        # Initialize the RandomForest classifier
        clf = RandomForestClassifier(class_weight="balanced", random_state=0)

        # Classify and get metrics
        metrics = classify(
            df_train, df_validate, final_features, feature_selection=False, clf=clf
        )
        output_path = os.path.join("./models", "RF_classifier.joblib")
        utils.create_file_with_directories(output_path, self.logger)
        # Save the trained classifier
        joblib.dump(clf, output_path)

        # # Load the classifier later
        # clf_loaded = load_classifier('random_forest_classifier.joblib')

        # # Make predictions on a test dataset and save to CSV
        # predictions = make_predictions(clf_loaded, df_test, final_features)
        # print(predictions)


def tokenize_feature_based(text):
    """
    tokenizer to tokenize input text
    """
    for match in re.finditer(r"\w+", text, re.UNICODE):
        yield match.group(0)


def compute_liwc_from_dict(df, col, num_cpus=None, offset=0):
    """
    Extract LIWC features from dictionary using multiprocessing
    """
    parse, category_names = liwc.load_token_parser(
        "./external_files/LIWC2015_English.dic"
    )  # path of LIWC dictionary

    # Use num_cpus or available CPUs
    num_cpus = num_cpus or os.cpu_count()
    contents = [str(row.content) for x, row in df.iterrows()]
    contents = contents[offset:]
    frames = []
    with mp.Pool(processes=num_cpus) as pool:
        frames += pool.imap(process_text, tqdm(contents, total=len(contents)))
    df_liwc = pd.concat(frames)
    return df_liwc


def process_text(content):
    text_tokens = tokenize_feature_based(content)
    text_counts = Counter(
        category for token in text_tokens for category in parse(token)
    )

    liwc_value_dic = {}
    for k, v in text_counts.items():
        liwc_value_dic["content"] = content
        word_count = len([word for word in content.split(" ")])
        liwc_value_dic["WC"] = word_count
        liwc_value_dic["WPS"] = sum(
            [len(sent.split(" ")) for sent in sent_tokenize(content)]
        ) / len(sent_tokenize(content))
        liwc_value_dic[k.split(",")[0].split(" ")[0]] = (v / word_count) * 100
    return pd.DataFrame([liwc_value_dic])


def extract_complexity_emotion_stylistic_features(df, text_key, logger):
    """
    Extract all the features used in paper for text in input filename.
    """
    # df = df.sample(frac=0.01)
    # Extract complexity features
    tqdm.pandas()
    logger.info("Computing readability...")
    df = readability.compute_readability(df, text_key)
    df = readability.compute_syntactic(df, text_key)
    logger.info("Computing stylistic features...")
    df["lexical_diversity"] = df[text_key].progress_apply(
        stylistic_features.lexical_diversity
    )
    df["wlen"] = df[text_key].progress_apply(stylistic_features.average_word_length)

    # Extract stylistic features
    df = stylistic_features.part_of_speech(df, text_key)
    df = stylistic_features.numeric_features(df, text_key)
    logger.info("Computing emotion features")
    # Extract emotion features
    emo_dic_path = "./external_files/emotion_intensity.txt"  # path to the emotion intensity lexicon dictionary
    df = emotion_features.emotion_NRC(df, emo_dic_path, text_key)
    df = emotion_features.sentiment_strength_vader(df, text_key)
    logger.info("Saving extracted features completed!")
    return df


def merge_liwc_and_remaining_features(df_liwc, df_remaining_features):
    """
    function to merge LIWC features and all other features.

    path: path to the folder where Generated_features folder is saved
    filename_remaining: name of pickle file that contains all other features except LIWC
    filename_liwc: name of file that contains LIWC features

    """

    # merge both dfs by index
    df_remaining_features["index"] = range(0, len(df_remaining_features))
    df_merged_features = df_remaining_features.merge(df_liwc, on="index", how="inner")
    return df_merged_features


def classify(df_train, df_validate, final_features, feature_selection, clf):
    # Prepare training data

    # Seed the random number generator
    seed(1)
    warnings.filterwarnings("ignore")
    y_train = df_train["label"].values
    X_train = df_train[final_features].values

    # Train the classifier
    clf.fit(X_train, y_train)

    # Validate the classifier
    y_validate = df_validate["label"].values
    X_validate = df_validate[final_features].values
    y_pred = clf.predict(X_validate)

    # Calculate metrics
    accuracy = accuracy_score(y_validate, y_pred)
    precision = precision_score(y_validate, y_pred)
    recall = recall_score(y_validate, y_pred)
    f1 = f1_score(y_validate, y_pred)

    try:
        proba = clf.predict_proba(X_validate)
        AP = average_precision_score(y_validate, proba[:, 1])
        AUROC = roc_auc_score(y_validate, proba[:, 1])
    except:
        AP = average_precision_score(y_validate, y_pred)
        AUROC = roc_auc_score(y_validate, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "AP": AP,
        "AUROC": AUROC,
    }


def load_classifier(filename):
    return joblib.load(filename)


# Ensure you have your dataframes df_train, df_validate, and df_test ready before calling main
def get_features(df_final):
    """
    function to get only those features used in paper of Horne and Adali plus emotion features
    """

    features_used_in_paper = [
        # "Analytic",
        "insight",
        "cause",
        "discrep",
        "tentat",
        "certain",
        "differ",
        "affiliation",
        "power",
        "reward",
        "risk",
        "work",
        "leisure",
        "money",
        "relig",
        # "Tone",
        "affect",
        "WC",
        "WPS",
        "num_nouns",
        "num_propernouns",
        "num_personalnouns",
        "num_ppssessivenouns",
        "num_whpronoun",
        "num_determinants",
        "num_whdeterminants",
        "num_cnum",
        "num_adverb",
        "num_interjections",
        "num_verb",
        "num_adj",
        "num_vbd",
        "num_vbg",
        "num_vbn",
        "num_vbp",
        "num_vbz",
        "focuspast",
        "focusfuture",
        "i",
        "we",
        "you",
        "shehe",
        "quant",
        "compare",
        # "Exclam",
        "negate",
        "swear",
        "netspeak",
        "interrog",
        "count_uppercased",
        "percentage_stopwords",
        # "AllPunc",
        # "Quote",
        "lexical_diversity",
        "wlen",
        "gunning_fog_index",
        "smog_index",
        "flesch_kincaid_grade_level",
    ]

    features_used_in_paper_ = [x for x in features_used_in_paper]

    emotion_features = [
        x
        for x in [
            "Anger",
            "Anticipation",
            "Disgust",
            "Fear",
            "Joy",
            "Sadness",
            "Surprise",
            "Trust",
            "neg",
            "pos",
            "posemo",
            "negemo",
            "anx",
        ]
    ]

    return features_used_in_paper_ + emotion_features


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
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
            "label": torch.tensor(label),
        }


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits


def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()


def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(
        actual_labels, predictions
    )


def predict_veracity(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        return preds.item()


def load_data(data_file):
    df = pd.read_csv(data_file)
    texts = df["content"].tolist()
    labels = [1 if label == "1" else 0 for label in df["label"].tolist()]
    return texts, labels
