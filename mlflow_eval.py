# -*- coding: utf-8 -*-
import os
import json

from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate

import openai
import pandas as pd
import mlflow

from mlflow.metrics.genai import make_genai_metric


def get_eval(transcript, report, config_data):

    config = config_data.get('llm3', {})
    model_name = config.get('model_name')
    openai_api_key = config.get('api_key')
    temperature = config.get('tmp')
    top_p = config.get('top_p')


    # Set OpenAi API KEY
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Define Custom LLM-judged metric
    factual_accuracy_metric = make_genai_metric(
        name="factual_accuracy",
        definition=(
            "Answer correctness is evaluated on the factual accuracy of the provided output based on the provide ground truth. Score can be assigned based on the factual correctness of the provided output to the provided input, where a higher score indicates factual accurate output."
        ),
        grading_prompt=(
            "Answer Correctness: Below are the details for different scores:"
            "- Score 0: The output is incorrect. There is no justification for the output in the provided ground truth."
            "- Score 1: The output is correct. It is factualy accurate according to the provided ground truth."
        ),
        version="v1",
        model=model_name,
        parameters={"temperature": temperature, "top_p": top_p},
        grading_context_columns=['ground_truth'],
        aggregations=["mean", "variance", "p90"],
        greater_is_better=True,
    )

    # Get Evaluation
    eval_df = pd.DataFrame(
        {
            "inputs": report['criteria'].values,
            "ground_truth": transcript,
            "outputs": report['results'].values
        }
    )

    with mlflow.start_run() as run:
        results = mlflow.evaluate(
            data=eval_df,
            targets="ground_truth",
            predictions="outputs",
            extra_metrics=[factual_accuracy_metric],
        )

    eval_table = results.tables["eval_results_table"]

    print(eval_table['factual_accuracy/v1/score'].values)
    print(eval_table['factual_accuracy/v1/justification'].values)
