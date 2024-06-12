import random
from typing import List, Tuple, Literal

from jinja2 import Template

from src.models import (
    PromptTemplateData,
    HistoricalStep,
    PromptTemplateHistoricalStep,
    TestStep,
    Test,
    TestConfiguration,
    Fence,
    TestRun,
)

ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _get_randomly_ordered_option_strings_and_char_of_specified_idx(
    options: List[str], remember_idx: int
) -> Tuple[List[str], str]:
    assert remember_idx < len(options)
    new_order = random.sample(range(len(options)), len(options))
    new_remember_idx = new_order.index(remember_idx)
    new_options = [options[i] for i in new_order]
    option_strings = []
    for i, option in enumerate(new_options):
        char = ALPHA[i]
        option_strings.append(f"{char}. {option}")
        if i == new_remember_idx:
            remember_alpha_char = char
    return option_strings, remember_alpha_char


def _get_prompt_ready_data(
    test: Test,
    test_configuration: TestConfiguration,
) -> Tuple[PromptTemplateData, str]:
    """Returns a prompt-template-ready data object and better choice alpha character given
    a test and a test configuration."""
    # Randomly sample an ordered list of all of the historical episodes uids
    ordered_historical_episode_uids = random.sample(
        list(test_configuration.historical_episode_uids),
        len(test_configuration.historical_episode_uids),
    )
    historical_episodes = [  # Get the historical episodes from mapping in this order
        test.historical_episodes[uid] for uid in ordered_historical_episode_uids
    ]
    # Get the final episode from mapping
    final_episode = test.test_episodes[test_configuration.final_episode_uid]
    # Verify that final episode contains only one TestStep at the end
    assert isinstance(final_episode[-1], TestStep)
    assert all(isinstance(step, HistoricalStep) for step in final_episode[:-1])
    # Flatten all of the historical episodes into one list of steps
    historical_steps = sum(historical_episodes, [])
    # Increment the historical steps from the final episode to this list
    historical_steps.extend(final_episode[:-1])
    final_test_step = final_episode[-1]  # Isolate the final TestStep
    # Convert historical steps into template-ready data objects
    template_historical_steps = []
    for historical_step in historical_steps:
        assert isinstance(historical_step, HistoricalStep)
        option_strings, chosen_alpha_char = (
            _get_randomly_ordered_option_strings_and_char_of_specified_idx(
                historical_step.choices, historical_step.chosen_idx
            )
        )
        template_historical_steps.append(
            PromptTemplateHistoricalStep(
                observation=historical_step.observation,
                options=option_strings,
                chosen=chosen_alpha_char,
            )
        )
    # Randomly sample the final options and return the better choice alpha char
    randomized_final_options, better_choice_char = (
        _get_randomly_ordered_option_strings_and_char_of_specified_idx(
            [final_test_step.better_choice, final_test_step.worse_choice], 0
        )
    )
    # Return the final data object and the correct choice alpha char
    prompt_template_data = PromptTemplateData(
        system_prompt=test.system_prompt,
        historical_steps=template_historical_steps,
        final_observation=final_test_step.observation,
        final_options=randomized_final_options,
    )
    return prompt_template_data, better_choice_char


def get_span_location_given_fence(text: str, fence: Fence) -> Tuple[int, int]:
    """Returns the index after the first fence instance and the index before the second
    fence instance in a string"""
    assert text.count(fence) == 2
    first_fence_end = text.find(fence) + len(fence)
    second_fence_start = text.rfind(fence)
    return first_fence_end, second_fence_start


def get_data_for_inference(
    test: Test,
    user_prompt_template: Template,
    system_prompt_template: Template,
    include_red_herring: bool,
    require_intermediate_inference: bool,
    length_class: Literal["short", "medium", "long"],
) -> TestRun:
    assert length_class in ("short", "medium", "long")
    assert isinstance(include_red_herring, bool)
    assert isinstance(require_intermediate_inference, bool)

    # Get the test configuration (episode uids)
    if include_red_herring is True and require_intermediate_inference is True:
        if length_class == "short":
            config = test.configurations.intermediate_inference_and_red_herring.short
        elif length_class == "medium":
            config = test.configurations.intermediate_inference_and_red_herring.medium
        else:
            config = test.configurations.intermediate_inference_and_red_herring.long
    elif include_red_herring is False and require_intermediate_inference is True:
        if length_class == "short":
            config = test.configurations.intermediate_inference_no_red_herring.short
        elif length_class == "medium":
            config = test.configurations.intermediate_inference_no_red_herring.medium
        else:
            config = test.configurations.intermediate_inference_no_red_herring.long
    elif include_red_herring is True and require_intermediate_inference is False:
        if length_class == "short":
            config = test.configurations.no_intermediate_inference_and_red_herring.short
        elif length_class == "medium":
            config = (
                test.configurations.no_intermediate_inference_and_red_herring.medium
            )
        else:
            config = test.configurations.no_intermediate_inference_and_red_herring.long
    else:
        if length_class == "short":
            config = test.configurations.no_intermediate_inference_no_red_herring.short
        elif length_class == "medium":
            config = test.configurations.no_intermediate_inference_no_red_herring.medium
        else:
            config = test.configurations.no_intermediate_inference_no_red_herring.long

    # Get the prompt-ready (except for lingering fences) data and better choice alpha char
    prompt_template_data, better_choice_char = _get_prompt_ready_data(test, config)
    # Render prompt templates
    user_prompt = user_prompt_template.render(
        historical_steps=prompt_template_data.historical_steps,
        final_observation=prompt_template_data.final_observation,
        final_options=prompt_template_data.final_options,
    )
    system_prompt = system_prompt_template.render(system_prompt=test.system_prompt)
    sys_prompt_length = len(system_prompt)

    # Handle and validate red herring fences
    if include_red_herring is True:
        red_herring_span_start, red_herring_span_end = get_span_location_given_fence(
            user_prompt, Fence.RED_HERRING
        )
        user_prompt = user_prompt.replace(Fence.RED_HERRING, "")
        red_herring_span_start += sys_prompt_length
        red_herring_span_end += sys_prompt_length
    else:
        assert Fence.RED_HERRING not in user_prompt
        red_herring_span_start = None
        red_herring_span_end = None

    # Handle and validate intermediate inference fences
    if require_intermediate_inference is True:
        (
            intermediate_inference_premise_one_span_start,
            intermediate_inference_premise_one_span_end,
        ) = get_span_location_given_fence(
            user_prompt, Fence.INTERMEDIATE_INFERENCE_PREMISE_ONE
        )
        user_prompt = user_prompt.replace(Fence.INTERMEDIATE_INFERENCE_PREMISE_ONE, "")
        intermediate_inference_premise_one_span_start += sys_prompt_length
        intermediate_inference_premise_one_span_end += sys_prompt_length

        (
            intermediate_inference_premise_two_span_start,
            intermediate_inference_premise_two_span_end,
        ) = get_span_location_given_fence(
            user_prompt, Fence.INTERMEDIATE_INFERENCE_PREMISE_TWO
        )
        user_prompt = user_prompt.replace(Fence.INTERMEDIATE_INFERENCE_PREMISE_TWO, "")
        intermediate_inference_premise_two_span_start += sys_prompt_length
        intermediate_inference_premise_two_span_end += sys_prompt_length

        assert Fence.INTERMEDIATE_INFERENCE_CONCLUSION not in user_prompt
        intermediate_inference_conclusion_span_start = None
        intermediate_inference_conclusion_span_end = None
    else:
        (
            intermediate_inference_conclusion_span_start,
            intermediate_inference_conclusion_span_end,
        ) = get_span_location_given_fence(
            user_prompt, Fence.INTERMEDIATE_INFERENCE_CONCLUSION
        )
        user_prompt = user_prompt.replace(Fence.INTERMEDIATE_INFERENCE_CONCLUSION, "")
        intermediate_inference_conclusion_span_start += sys_prompt_length
        intermediate_inference_conclusion_span_end += sys_prompt_length

        assert Fence.INTERMEDIATE_INFERENCE_PREMISE_ONE not in user_prompt
        assert Fence.INTERMEDIATE_INFERENCE_PREMISE_TWO not in user_prompt
        intermediate_inference_premise_one_span_start = None
        intermediate_inference_premise_one_span_end = None
        intermediate_inference_premise_two_span_start = None
        intermediate_inference_premise_two_span_end = None

    for fence in Fence:
        assert fence not in user_prompt

    # Return test run object
    return TestRun(
        test_title=test.title,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        better_choice_char=better_choice_char,
        total_length=sys_prompt_length + len(user_prompt),
        length_class=length_class,
        include_red_herring=include_red_herring,
        require_intermediate_inference=require_intermediate_inference,
        red_herring_prompt_span_start=red_herring_span_start,
        red_herring_prompt_span_end=red_herring_span_end,
        intermediate_inference_premise_one_prompt_span_start=intermediate_inference_premise_one_span_start,
        intermediate_inference_premise_one_prompt_span_end=intermediate_inference_premise_one_span_end,
        intermediate_inference_premise_two_prompt_span_start=intermediate_inference_premise_two_span_start,
        intermediate_inference_premise_two_prompt_span_end=intermediate_inference_premise_two_span_end,
        intermediate_inference_conclusion_prompt_span_start=intermediate_inference_conclusion_span_start,
        intermediate_inference_conclusion_prompt_span_end=intermediate_inference_conclusion_span_end,
    )
