# FakeNewsDetectionUsingLLMs

## Introduction
This repo presents the complete pipeline for reproducing the experiments made in my Thesis available at [link](https://example.com). To encourage reproducibility and open science we have made a pipeline consisting of several parts that are either possible to run sequentially or individually. The main way how the pipeline parts can be run is in the following way:
```shell
python3 main.py [Pipeline Part] [...Pipeline Part arguments]
```
The several parts are described as follows:

| Command | Description |
|---------|------------- |
| `python3 main.py preprocess` | Preprocess the data from multiple fake news net datasets | 
| `python3 main.py merge ` | Merges all preprocessed data into one file | 
| `python3 main.py split`  | Splits the merged dataset into a training, validation, and test set (60/20/20) for training/finetuning and evaluation | 
| `python3 main.py train` | Trains the baselines (Random Forest/BERT) on a given training and validation set for binary veracity classification | 
| `python3 main.py finetune` | Finetunes a given LLM on a given training and validation set for binary veracity classification |
| `python3 main.py baseline_classify` | Given a baseline model and dataset, queries the model to make predictions for each article in the dataset |
| `python3 main.py prompt` | Given a LLM and dataset, queries the model to make predictions for each article in the dataset |
| `python3 main.py analyze_data` | Analyzes characteristics of the dataset to compare with later predictions for multi-lens evaluation |
| `python3 main.py analyze_predictions` | Analyzes folder of predictions made by baselines and LLMs  | 


## Setup and requirements

These steps describe the process for Linux based operating systems. The commands/installation may be different for different environments, but the general steps remain the same. 

0. Clone the repository.
1. Make sure that you have a CUDA enabled machine and that it is installed. (tips on how to install CUDA are described [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
2. Setup a virtual environment:
    ```shell
    python3 -m venv  venv
    ```
3. Activate the virtual environment:
    ```shell
    python3 ./venv/bin/activate
    ```
4. Install the required python packages using pip
    ```shell
    pip install -r requirements.txt
    ```

999. Some LLMs are gated by default on huggingface and you need access to use them. This [guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli) describes how to use a huggingface token (when access is granted) to leverage them. 

## Pipeline
This section describes the different pipeline parts, how to use, and how to extend them.

### Preprocess
The preprocess step takes as input various labeled fake news datasets and prepares them for merging into on combined dataset. The arguments needed to run the preprocess steps are the following:

| Argument | Description |
|---------|------------- |
| dataset | One of the types of dataset that we support, as of writing these are: **FakeNewsNet**, **FakeHealth**, and **MOCHEG** |
| input | Relative input path location of the dataset |

for example, if you want to preprocess FakeNewsNet the preprocess command would look like this:

```shell
python3 main.py preprocess FakeNewsNet ./data/in/FakeNewsNet
```

The output file would then be generated in  `/data/out/preprocess/FakeNewNet.csv`.

As of writing we support three datasets: 
- FakeNewsNet
- FakeHealth
- MOCHEG

For your particular use-case you can choose which ones to use. Others can also be added but Python code needs to be added in the [preprocess](./preprocessor.py) file. This can be done by adding three new functions similar to the existing ones:

- **\_preprocess_NEW_DATASET**
- **\_normalize_NEW_DATASET**
- **\_normalize_NEW_DATASET_entry**

From there you can preprocess the new dataset by running:

```shell
python3 main.py preprocess [NEW_DATASET] [PATH_TO_NEW_DATASET]
```

**Note**: It's important to keep in mind that when adding a new dataset you can also add a corresponding topic to each article. in the **\_preprocess_NEW_DATASET** function.

### Merge
The merge step takes as input a directory containing preprocessed files and outputs one combined csv file containing all articles. The arguments needed to run the merge step are the following:

| Argument | Description |
|---------|------------- |
| input | Relative input path location of the directory containing preprocessed datasets |
| output | Relative path to the merged csv file. |

for example, if you want to merge the files from the preprocess steps into one you could run:

```shell
python3 main.py merge ./data/out/preprocess ./data/out/merge/combined.csv
```

### Split
The split step takes as input a combined dataset file and outputs a directory containing the stratified split files (train,val,test). If you want to change the split to something else change it in [split](./splitter.py). The arguments needed to run the split step are the following:

| Argument | Description |
|---------|------------- |
| input | Relative path to the merged csv file. |
| output | Relative outputh path location of directory to store the split sets. |

for example, if you want to split the file from the merge step could run:

```shell
python3 main.py split ./data/out/merge/combined.csv ./data/out/split/combined/
```

### Train
The train step takes as input a directory containing the train and validation set and outputs a model in the ./models directory.

| Argument | Description |
|---------|------------- |
| train type | Type of baseline to train currently that is **PromptBased** or **FeatureBased** . |
| input | Relative input directory containing the test and validation set |

for example, if you want to train a FeatureBased classifier on the split dataset we generated in the split set you can:

```shell
python3 main.py train FeatureBased ./data/out/split/combined/
```

This will the following model in the ./models directory: RF_classifier.joblib

**Note**: It's important to keep in mind that to train the FeatureBased classifier you need an additional file in the ./external_files directory containing the liwc data. This file is not freely available online and should be bought from the official website. It is not a good idea to push this file online in your personal github repo and then make the github repo public. This way other people can just search: LIWC2015_English.dic on github click on Code and see your pushed dictionary. DO NOT DO THIS

For your particular use-case you can choose which ones to use. Others can also be added but Python code needs to be added in the [preprocess](./preprocessor.py) file. This can be done by adding three new functions similar to the existing ones:

- **\_train_[NEW CLASSIFIER]**


From there you can train the news classifier by running:

```shell
python3 main.py preprocess [NEW CLASSIFIER] [PATH_TO_SPLITS]
```

### Finetune

The finetune step takes as input a directory containing the train and validation set and uploads a finetuned model to your huggingface page (by huggingface token).

| Argument | Description |
|---------|------------- |
| input | Relative input directory containing the test and validation set |
| base LLM |LLM to finetune, right now, possible options are: ["gemma-2b", "gemma-2-9b", "llama-3.1-8b-it", "mistral-0.2-7b-it"] |


for example, if you want to finetune gemma-2b LLM on the split dataset we generated in the split set you can:

```shell
python3 main.py finetune ./data/out/split/combined/ gemma-2b
```

For your particular use-case you can choose which models to use. These can be added by changing the model_options (in line 14), and adding a case in the load_model function (in line 57) in [finetune](./finetune.py) 

From there you can finetune the new model by runnning:

```shell
python3 main.py preprocess [NEW LLM] [PATH_TO_SPLITS]
```

### Classify

The classify step takes as input a baseline type, a test set to classify, a classifier model and a output directory

| Argument | Description |
|---------|------------- |
| train type | Type of baseline to train currently that is **PromptBased** or **FeatureBased** . |
| input | Relative input file of the dataset to classify |
| model | Relative path to model to use |
| output | Relative path to directory for predictions |

for example, if you want to use the FeatureBased classifier we trained in the Train step on the test set we created in the Split step and want to output the predictions in ./data/out/predictions you can run:

```shell
python3 main.py finetune ./data/out/split/combined/ gemma-2bpython3 main.py classify PromptBased ./data/out/split/combined/test.csv ./models/BERT_classifier.pth ./data/out/predictions/baselines/
```


### Prompt

The classify step takes as input a baseline type, a test set to classify, a classifier model and a output directory

| Argument | Description |
|---------|------------- |
| input | Relative input file of the dataset to classify |
| output | Relative path to directory for predictions |
| model | possible options are: ["gemma-2b", "gemma-2-9b", "llama-3.1-8b-it" |
| detection strategy | possible options are: ["binary", "discrete", "percentage", "cot"] |
| (optional) finetuned | Whether to use the finetuned adapter (can be left blank) |

for example, if you want to use the Gemma2b model with the finetuned adapter to predict binary predictions for the test set created in the Split step you can run:

```shell
python3 main.py prompt ./data/out/split/combined/test.csv ./data/out/predictions/llms gemma-2b binary --finetuned
```
**Note**: It's important to keep in mind that right now we use the finetuned adapters that we created (which are open to use) if you want to use your own models finetuned in the Finetune step you can change the utils.get_repo function (line 39) in [utils](./utils.py).

### Analyze data

With this step you can find basic characteristics of a dataset before you classify/prompt models on them. 

| Argument | Description |
|---------|------------- |
| input | Relative input file of the dataset to analyze |
| output | Relative path to directory to store analyzed results |

for example, analyze the test set in we created in the Split step you can run the following: 


```shell
python3 main.py analyze_data ./data/out/split/combined/test.csv ./data/out
```

### Analyze Predictions

With this step you can analyze the predictions made in the Prompt or Classify step.

| Argument | Description |
|---------|------------- |
| input | Relative input file of the dataset to analyze |
| output | Relative path to directory to store analyzed results |

for example, to analyze the predictions of the baselines in the Classify step we can run the following: 


```shell
python3 main.py analyze_predictions ./data/out/predictions/baselines/ ./data/out/results
```
