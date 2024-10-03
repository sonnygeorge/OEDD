import json
import os
import re
import time
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

import scipy.stats as stats
from dotenv import load_dotenv
from jinja2 import Template

# from langchain_ai21 import ChatAI21
# from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.models import LlmChoice, Test, TestRun
from src.utils import get_data_for_inference

load_dotenv()

with open("templates/user_prompt.jinja2") as f:
    user_prompt_template_str = f.read()

with open("templates/system_prompt.jinja2") as f:
    system_prompt_template_str = f.read()


RESULTS_CSV_PATH = "results.csv"
USER_PROMPT_TEMPLATE = Template(user_prompt_template_str)
SYSTEM_PROMPT_TEMPLATE = Template(system_prompt_template_str)
TEMPERATURE = 0.5
MIN_RUNS_PER = 30
MAX_RUNS_PER = 50
CONFIDENCE_INTERVAL = 0.90
CONFIDENCE_INTERVAL_WIDTH_THRESHOLD = 0.1


class Config(BaseModel):
    include_red_herring: bool
    require_intermediate_inference: bool
    length_class: Literal["short", "medium", "long"]


MODELS_TO_TEST = [
    ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=TEMPERATURE),
    ChatOpenAI(model="gpt-4o-2024-05-13", temperature=TEMPERATURE),
    ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=TEMPERATURE),
    # ChatAnthropic(model="claude-3-opus-20240229", temperature=TEMPERATURE),
    # ChatNVIDIA(
    #     model="meta/llama3-70b-instruct",
    #     temperature=TEMPERATURE,
    # ),
    # ChatNVIDIA(
    #     model="mistralai/mixtral-8x22b-instruct-v0.1",
    #     temperature=TEMPERATURE,
    # ),
]

PRINT_MSG = "{i} | {model}  |  LEN: {len}  |  ELAPSED: {elapsed}  |  TITLE: {title}  |"
PRINT_MSG += "  RH: {rh}  |  II: {ii}  |  CUR CI WIDTH: {ci:.3f}  |  CORRECT: {avg:.3f}"


class Config(BaseModel):
    include_red_herring: bool
    require_intermediate_inference: bool
    length_class: Literal["short", "medium", "long"]


CONFIGS_TO_RUN = [
    Config(
        include_red_herring=True,
        require_intermediate_inference=True,
        length_class="short",
    ),
    Config(
        include_red_herring=True,
        require_intermediate_inference=True,
        length_class="medium",
    ),
    Config(
        include_red_herring=True,
        require_intermediate_inference=True,
        length_class="long",
    ),
    Config(
        include_red_herring=False,
        require_intermediate_inference=True,
        length_class="short",
    ),
    Config(
        include_red_herring=False,
        require_intermediate_inference=True,
        length_class="medium",
    ),
    Config(
        include_red_herring=False,
        require_intermediate_inference=True,
        length_class="long",
    ),
    Config(
        include_red_herring=True,
        require_intermediate_inference=False,
        length_class="short",
    ),
    Config(
        include_red_herring=True,
        require_intermediate_inference=False,
        length_class="medium",
    ),
    Config(
        include_red_herring=True,
        require_intermediate_inference=False,
        length_class="long",
    ),
    Config(
        include_red_herring=False,
        require_intermediate_inference=False,
        length_class="short",
    ),
    Config(
        include_red_herring=False,
        require_intermediate_inference=False,
        length_class="medium",
    ),
    Config(
        include_red_herring=False,
        require_intermediate_inference=False,
        length_class="long",
    ),
]


def extract_action_choice_from_response_content(
    response_content: str,
) -> Optional[Tuple[str, str]]:
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, response_content, re.DOTALL)

    for match in matches[::-1]:
        try:
            action_choice_object = LlmChoice.model_validate_json(match)
            return action_choice_object.chosen, action_choice_object.reasoning
        except Exception:
            print(f"Error decoding JSON")

    print("No JSON found in response content:", response_content[:90], "...")
    return None, None


def calculate_confidence_interval(data, confidence_level=0.95) -> Tuple[float, float]:
    sample_size = len(data)
    sample_mean = np.mean(data)
    standard_error = stats.sem(data)
    margin_of_error = standard_error * stats.t.ppf(
        (1 + confidence_level) / 2.0, sample_size - 1
    )

    return sample_mean - margin_of_error, sample_mean + margin_of_error


def get_model_string_from_model(model: BaseChatModel) -> str:
    if hasattr(model, "model_name"):
        return model.model_name
    elif hasattr(model, "model"):
        return model.model
    else:
        raise NotImplementedError


def get_temperature_from_model(model: BaseChatModel) -> float:
    if hasattr(model, "temperature"):
        return model.temperature
    else:
        raise NotImplementedError


def infer_action_choice(
    chat_model: BaseChatModel,
    test_run_data: TestRun,
) -> Optional[Tuple[str, str]]:
    model_string = get_model_string_from_model(chat_model)
    if model_string == "mistralai/mixtral-8x22b-instruct-v0.1":
        # Hotfix for this NVIDIA API model since requests get bounced otherwise
        single_prompt = test_run_data.system_prompt + test_run_data.user_prompt
        messages = [single_prompt]
    else:
        messages = [
            SystemMessage(test_run_data.system_prompt),
            HumanMessage(test_run_data.user_prompt),
        ]
    response = chat_model.invoke(messages)
    return extract_action_choice_from_response_content(response.content)


def run_test_config_until_confidence_interval_small_enough(
    chat_model: BaseChatModel,
    test: Test,
    include_red_herring: Optional[bool] = None,
    require_intermediate_inference: Optional[bool] = None,
    length_class: Optional[Literal["short", "medium", "long"]] = None,
    get_bias: Optional[bool] = False,
) -> pd.DataFrame:
    inference_runs_correctness = []
    runs: List[TestRun] = []

    start = time.perf_counter()
    while True:
        # Get fresh test run object with new random shuffling of random-order components
        test_run_data = get_data_for_inference(
            test=test,
            user_prompt_template=USER_PROMPT_TEMPLATE,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            include_red_herring=include_red_herring,
            require_intermediate_inference=require_intermediate_inference,
            length_class=length_class,
            get_bias=get_bias,
        )

        # Get inference
        chosen_char, given_reasoning = infer_action_choice(
            chat_model=chat_model, test_run_data=test_run_data
        )

        # Finalize run data
        test_run_data.chosen_char = chosen_char
        test_run_data.given_reasoning = given_reasoning
        test_run_data.was_correct = chosen_char == test_run_data.better_choice_char
        test_run_data.chat_model_string = get_model_string_from_model(chat_model)
        test_run_data.temperature = get_temperature_from_model(chat_model)
        test_run_data.iteration_num = len(inference_runs_correctness) + 1

        inference_runs_correctness.append(test_run_data.was_correct)
        runs.append(test_run_data)

        # Exit loop if confidence interval converged enough
        if len(inference_runs_correctness) > 1:
            lower_bound, upper_bound = calculate_confidence_interval(
                inference_runs_correctness, CONFIDENCE_INTERVAL
            )
            confidence_interval_width = upper_bound - lower_bound
            elapsed = time.perf_counter() - start
            print_msg = PRINT_MSG.format(
                i=test_run_data.iteration_num,
                model=test_run_data.chat_model_string,
                len=test_run_data.length_class,
                elapsed=str(elapsed).split(".")[0],
                title=test_run_data.test_title,
                rh=include_red_herring,
                ii=require_intermediate_inference,
                ci=confidence_interval_width,
                avg=sum(inference_runs_correctness) / len(inference_runs_correctness),
            )
            print(print_msg)
            if (
                len(inference_runs_correctness) >= MIN_RUNS_PER
                and confidence_interval_width < CONFIDENCE_INTERVAL_WIDTH_THRESHOLD
            ) or len(inference_runs_correctness) >= MAX_RUNS_PER:
                break

    return pd.DataFrame([run.model_dump() for run in runs])


def add_runs_to_csv(df: pd.DataFrame) -> None:
    df = df.drop(columns=["system_prompt", "user_prompt"])  # Hotfix - file IO too slow
    if os.path.exists(RESULTS_CSV_PATH):
        temp_csv_path = "temp.csv"
        old_df = pd.read_csv(RESULTS_CSV_PATH)
        new_df = pd.concat([old_df, df], ignore_index=True)
        new_df.to_csv(temp_csv_path, mode="w", index=False)
        os.replace(temp_csv_path, RESULTS_CSV_PATH)
    else:
        df.to_csv(RESULTS_CSV_PATH, mode="w", index=False)


def run_tests_for_model(
    chat_model: BaseChatModel, get_bias: bool = False
) -> Optional[pd.DataFrame]:
    # Skip if model results already in .csv
    model_string = get_model_string_from_model(chat_model)
    if os.path.exists(RESULTS_CSV_PATH):
        df = pd.read_csv(RESULTS_CSV_PATH)
        if model_string in df["chat_model_string"].values:
            print(f"Skipping '{model_string}' - already exists in results .csv.")
            return None

    try:
        all_model_runs = pd.DataFrame()
        for fname in os.listdir("tests"):
            if not fname.endswith(".json"):
                continue

            with open(f"tests/{fname}", "r") as f:
                data = json.load(f)
            test = Test(**data)

            if get_bias:
                df = run_test_config_until_confidence_interval_small_enough(
                    chat_model=chat_model,
                    test=test,
                    include_red_herring=None,
                    require_intermediate_inference=None,
                    length_class=None,
                    get_bias=True,
                )
                all_model_runs = pd.concat([all_model_runs, df], ignore_index=True)
            for config in CONFIGS_TO_RUN:
                df = run_test_config_until_confidence_interval_small_enough(
                    chat_model=chat_model,
                    test=test,
                    include_red_herring=config.include_red_herring,
                    require_intermediate_inference=config.require_intermediate_inference,
                    length_class=config.length_class,
                )
                all_model_runs = pd.concat([all_model_runs, df], ignore_index=True)

        add_runs_to_csv(all_model_runs)
    except Exception as e:
        print(f"Error running tests for '{model_string}': {e}")

    return all_model_runs


def run_tests(get_a_priori_bias: bool = False) -> None:
    start = time.perf_counter()
    last_model_completed = start
    dfs = []
    for chat_model in MODELS_TO_TEST:
        df = run_tests_for_model(chat_model=chat_model, get_bias=get_a_priori_bias)
        if df is not None:
            dfs.append(df)
        elapsed = time.perf_counter() - last_model_completed
        print(f"### Elapsed for model: {str(elapsed).split('.')[0]} ###")
        last_model_completed = time.perf_counter()
    print(
        f"### Elapsed for all models: {str(time.perf_counter() - start).split('.')[0]} ###"
    )


if __name__ == "__main__":
    run_tests(get_a_priori_bias=False)
