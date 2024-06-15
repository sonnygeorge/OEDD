import random
from typing import List, Literal, Optional, Tuple

from jinja2 import Template

from src.models import (
    Fence,
    HistoricalStep,
    PromptTemplateData,
    PromptTemplateHistoricalStep,
    Test,
    TestConfiguration,
    TestRun,
    TestStep,
)

ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _get_randomly_ordered_option_strings_and_char_of_specified_idx(
    options: List[str], desired_option_idx: int
) -> Tuple[List[str], str]:
    assert desired_option_idx < len(options)

    new_order = random.sample(range(len(options)), len(options))
    idx_to_original_desired_option = new_order.index(desired_option_idx)
    reordered_options = [options[i] for i in new_order]
    reordered_option_strings = []

    for i, option in enumerate(reordered_options):
        char = ALPHA[i]
        reordered_option_strings.append(f"{char}. {option}")
        if i == idx_to_original_desired_option:
            desired_option_alpha_char = char

    return reordered_option_strings, desired_option_alpha_char


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
    # Add the historical steps from the final episode to this list
    historical_steps.extend(final_episode[:-1])
    # Isolate the final TestStep
    final_test_step = final_episode[-1]

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


def remove_fences_from_prompt(
    prompt: str, fence: str, starting_idx: int = 0
) -> Tuple[str, int, int]:
    """Removes the specified fence from the prompt and returns the updated prompt and
    the start and end indices of the fence."""
    (
        fence_span_start,
        fence_span_end,
    ) = get_span_location_given_fence(prompt, fence)
    prompt = prompt.replace(fence, "")
    return (
        prompt,
        fence_span_start + starting_idx,
        fence_span_end + starting_idx,
    )


def get_data_for_inference(
    test: Test,
    user_prompt_template: Template,
    system_prompt_template: Template,
    include_red_herring: Optional[bool],
    require_intermediate_inference: Optional[bool],
    length_class: Optional[Literal["short", "medium", "long"]],
    get_bias: Optional[bool] = False,
) -> TestRun:
    assert length_class in ("short", "medium", "long", None)
    assert isinstance(include_red_herring, bool) or include_red_herring is None
    assert (
        isinstance(require_intermediate_inference, bool)
        or require_intermediate_inference is None
    )

    # Get the test configuration (episode uids)
    if get_bias is True:
        # Only use the final test episode with no red herring

        episodes = test.test_episodes.keys()
        matches = [e for e in episodes if "_TEST" in e and "RH" not in e]
        assert len(matches) == 1
        config = TestConfiguration(
            historical_episode_uids=[],
            final_episode_uid=matches[0],
        )
    elif include_red_herring is True and require_intermediate_inference is True:
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
        user_prompt, red_herring_span_start, red_herring_span_end = (
            remove_fences_from_prompt(user_prompt, Fence.RED_HERRING, sys_prompt_length)
        )
    else:
        assert Fence.RED_HERRING not in user_prompt
        red_herring_span_start = None
        red_herring_span_end = None

    # Handle and validate intermediate inference fences
    if require_intermediate_inference is True:
        (
            user_prompt,
            intermediate_inference_premise_one_span_start,
            intermediate_inference_premise_one_span_end,
        ) = remove_fences_from_prompt(
            user_prompt, Fence.INTERMEDIATE_INFERENCE_PREMISE_ONE, sys_prompt_length
        )

        (
            user_prompt,
            intermediate_inference_premise_two_span_start,
            intermediate_inference_premise_two_span_end,
        ) = remove_fences_from_prompt(
            user_prompt, Fence.INTERMEDIATE_INFERENCE_PREMISE_TWO, sys_prompt_length
        )

        assert Fence.INTERMEDIATE_INFERENCE_CONCLUSION not in user_prompt
        intermediate_inference_conclusion_span_start = None
        intermediate_inference_conclusion_span_end = None
    elif get_bias is not True:
        (
            user_prompt,
            intermediate_inference_conclusion_span_start,
            intermediate_inference_conclusion_span_end,
        ) = remove_fences_from_prompt(
            user_prompt, Fence.INTERMEDIATE_INFERENCE_CONCLUSION, sys_prompt_length
        )

        assert Fence.INTERMEDIATE_INFERENCE_PREMISE_ONE not in user_prompt
        assert Fence.INTERMEDIATE_INFERENCE_PREMISE_TWO not in user_prompt
        intermediate_inference_premise_one_span_start = None
        intermediate_inference_premise_one_span_end = None
        intermediate_inference_premise_two_span_start = None
        intermediate_inference_premise_two_span_end = None
    else:
        intermediate_inference_premise_one_span_start = None
        intermediate_inference_premise_one_span_end = None
        intermediate_inference_premise_two_span_start = None
        intermediate_inference_premise_two_span_end = None
        intermediate_inference_conclusion_span_start = None
        intermediate_inference_conclusion_span_end = None

    for fence in Fence:
        assert fence not in user_prompt

    # Return test run object
    return TestRun(
        test_title=test.title,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        better_choice_char=better_choice_char,
        total_length=sys_prompt_length + len(user_prompt),
        get_bias=get_bias,
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
