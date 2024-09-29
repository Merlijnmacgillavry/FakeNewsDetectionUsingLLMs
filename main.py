# import external libraries
import pandas as pd
import logging
import argparse

# import internal libraries
import utils

# Pipeline parts
import preprocessor

import merger

import splitter

import trainer

import baseline_classify

import prompter

import finetune

# import finetune
import analyze_data

import analyze_predictions


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    green = "\x1b[92m"
    format = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
        # logging.SUCCESS: green + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


if __name__ == "__main__":
    logger = logging.getLogger("main.py")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    logger.info("Logger initialized")
    logger.info("Started Main")
    parser = argparse.ArgumentParser(
        description="\x1b[33;20m"
        + "Thesis program entry point: Decide whether to run the pre-processing pipeline or the prompter"
        + "\x1b[0m",
        prog="main",
    )
    subparsers = parser.add_subparsers(
        title="subcommands", help="different subcommands of the repo"
    )
    # Add preprocess subcommand
    preprocessor = preprocessor.Preprocessor(logger=logger)
    preprocessor.add_parser(subparsers)

    merger = merger.Merger(logger=logger)
    merger.add_parser(subparsers)

    splitter = splitter.Splitter(logger=logger)
    splitter.add_parser(subparsers)

    trainer = trainer.Trainer(logger=logger)
    trainer.add_parser(subparsers)

    baselineClassifier = baseline_classify.BaselineClassifier(logger=logger)
    baselineClassifier.add_parser(subparsers)

    prompter = prompter.Prompter(logger=logger)
    prompter.add_parser(subparsers)

    analyze_data = analyze_data.AnalyzeData(logger=logger)
    analyze_data.add_parser(subparsers)

    analyze_predictions = analyze_predictions.AnalyzePredictions(logger=logger)
    analyze_predictions.add_parser(subparsers)

    finetuner = finetune.FineTuner(logger=logger)
    finetuner.add_parser(subparsers)

    args = parser.parse_args()
    args.func(args)
