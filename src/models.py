from enum import StrEnum
from typing import List, Dict, Union, Optional, Literal

from pydantic import BaseModel


class IntermediateInference(BaseModel):
    """Two independent premises that, only collectively, lead to a rightfully assumed
    conclusion in the universe of a hypothetical Language Model Agent. This conclusion is
    the basis for the betterness of the "better" action choice in a test.

    Attributes:
        premise_one (str): The first premise.
        premise_two (str): The second premise.
        conclusion (str): The unambiguously makeable assumption that follows from both
            premises collectively.
    """

    premise_one: str
    premise_two: str
    conclusion: str


class PivotalTestConstituents(BaseModel):
    """The pivotal constituents of a step in a Markov Decision Process that tests a
    Language Model Agent's ability choose the better of two action paths using the
    necessary deduction from two previously-gathered facts, despite the presence of a red
    herring.

    Attributes:
        agent_goal (str): A short description of the agent's universe and goal.
        test_scenario (str): A short summary of the environment circumstance that begets the
            decision to be made.
        better_choice (str): The better action choice.
        worse_choice (str): The worse action choice.
        intermediate_inference (NecessaryDeduction): A deduction that makes the better choice
            unambiguously superior to the worse choice.
        red_herring (str): A previously-discoverable persistent universe truth that,
            assuming the agent fails to make the `intermediate_inference` serves to skew
            their judgement away from the better choice.
    """

    agent_goal: str
    test_scenario: str
    better_choice: str
    worse_choice: str
    intermediate_inference: IntermediateInference
    red_herring: str


class Fence(StrEnum):
    RED_HERRING = "$$$RH$$$"
    INTERMEDIATE_INFERENCE_CONCLUSION = "$$$IIC$$$"
    INTERMEDIATE_INFERENCE_PREMISE_ONE = "$$$IIP1$$$"
    INTERMEDIATE_INFERENCE_PREMISE_TWO = "$$$IIP2$$$"


class HistoricalStep(BaseModel):
    """A triple of the agent's observation, the choices they were presented with, and the
    index of the choice they made.

    NOTE: At test time, the order in which the choices are presented should be randomized.
    """

    observation: str
    choices: List[str]
    chosen_idx: int


class TestStep(BaseModel):
    """The current step in which the agent is tested to see if they make the better of
    two action choices.

    NOTE: At test time, the order in which the choices are presented should be randomized.
    """

    observation: str
    better_choice: str
    worse_choice: str


class PromptTemplateHistoricalStep(BaseModel):
    """Historical step data that has been prepped/cleaned for prompt template injection."""

    observation: str
    options: List[str]
    chosen: str


class PromptTemplateData(BaseModel):
    """All data prepped, cleaned, and ready to be injected into a prompt template."""

    historical_steps: List[PromptTemplateHistoricalStep]
    final_observation: str
    final_options: List[str]


HistoricalEpisode = List[HistoricalStep]
TestEpisode = List[Union[HistoricalStep, TestStep]]  # Must have only 1 TestStep at [-1]
HistoricalEpisodesDict = Dict[str, HistoricalEpisode]
TestEpisodesDict = Dict[str, TestEpisode]


class TestConfiguration(BaseModel):
    """A test setup defined by its inclusion of episodes from the `Test`'s `episodes`
    dictionary. The `final_episode` is always presented as the last episode. The
    `historical_episodes` are always presented beforehand in a randomized order.

    Attributes:
        historical_episode_uids (List[str]): A set of historical episode uid strings.
        final_episode_uid (str): The uid of the final test episode.
    """

    historical_episode_uids: List[str]
    final_episode_uid: str


class LengthConfigurations(BaseModel):
    """The different levels of context length for test setups."""

    short: TestConfiguration
    medium: TestConfiguration
    long: TestConfiguration


class ReasoningConfigurations(BaseModel):
    """The different combinations of reasoning required by a test setup."""

    no_intermediate_inference_no_red_herring: LengthConfigurations
    intermediate_inference_no_red_herring: LengthConfigurations
    no_intermediate_inference_and_red_herring: LengthConfigurations
    intermediate_inference_and_red_herring: LengthConfigurations


class Test(BaseModel):
    """All of the data associated with a test.

    Attributes:
        canary (str): Marks tests with a unique pattern that if completable by an LLM,
            indicates that all test data has likely leaked into the LLM's training data.
        title (str): The title of the test.
        pivotal_constituents (PivotalTestConstituents): The pivotal constituents of the test.
        system_prompt (str): A system prompt describing important system-level information
            to the LLM agent.
        historical_episodes (HistoricalEpisodesDict): a map of all historical episodes that
            have been written for this test indexed by their uid strings.
        test_episodes (TestEpisodesDict): a map of all test episodes that have been written
            for this test indexed by their uid strings.
        configurations (ReasoningConfigurations): The different setups of required reasoning
            and context length.
    """

    canary: str
    title: str
    pivotal_constituents: PivotalTestConstituents
    system_prompt: Optional[str]
    historical_episodes: HistoricalEpisodesDict
    test_episodes: TestEpisodesDict
    configurations: ReasoningConfigurations


class TestRun(BaseModel):
    """The data model associated with a single run of a test configuration. These are the
    rows to the results .csv file.

    Attributes:
    # FIXME
    """

    test_title: str
    system_prompt: str
    user_prompt: str
    better_choice_char: str
    chosen_char: Optional[str] = None  # Populated once LLM inference made
    was_correct: Optional[bool] = None  # Populated once LLM inference made
    chat_model_string: Optional[str] = None  # Populated after init
    temperature: Optional[float] = None  # Populated after init
    iteration_num: Optional[int] = None  # Populated after init
    total_length: int
    total_length: int
    length_class: Literal["short", "medium", "long"]
    include_red_herring: bool
    require_intermediate_inference: bool
    red_herring_prompt_span_start: Optional[int]
    red_herring_prompt_span_end: Optional[int]
    intermediate_inference_premise_one_prompt_span_start: Optional[int]
    intermediate_inference_premise_one_prompt_span_end: Optional[int]
    intermediate_inference_premise_two_prompt_span_start: Optional[int]
    intermediate_inference_premise_two_prompt_span_end: Optional[int]
    intermediate_inference_conclusion_prompt_span_start: Optional[int]
    intermediate_inference_conclusion_prompt_span_end: Optional[int]
