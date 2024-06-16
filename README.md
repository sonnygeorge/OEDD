# OEDD

Official codebase for the paper _Probing the Capacity of Language Model Agents to Operationalize Disparate Experiential Context Despite Distraction_

## OEDD Corpus of Reasoning Tests for LLM Agents

The OEDD (Operationalize Experience Despite Distraction) corpus is a collection of reasoning tests designed to evaluate the capacity of language model agent systems to make smart action-inferences despite plausible distractions.

The different versions of the corpus can be downloaded here: [omitted for review]

Our `results.csv` from our initial experiments using v1.0.0 of the corpus can be downloaded here: [omitted for review]

## Code Structure

```text
.
├── assets                    # contains .svgs for app.py
├── src                       # main source code
│   ├── models.py             #   Pydantic data objects
│   └── utils.py              #   helper functions
├── app.py                    # runs NiceGUI app on local machine to visualize corpus
├── figures.py                # script to generate matplotlib figures from
└── run_tests.py              # script to run tests with GPT-3.5-Turbo, GPT-4o, and Gemini 1.5 Pro
```

## Continual Updates

We consider this a living corpus and encourage community scrutiny, feedback, and contributions.

Corpus updates and justifications will be documented here:

| Date | Version | Comments |
|-|-|-|
| [release date] | 1.0.0 | Initial release |

To suggest changes to the corpus, please contact the repository owner directly with your suggestions/changes/additions.

In general, we ask that people refrain from discussing the contents of the corpus in public forums to avoid biasing future evaluations.

## Corpus Visualization App

We provide a custom [NiceGUI](https://github.com/zauberzeug/nicegui) application that allows users to more intuitively visualize the contents of the OEDD tests.

It can be run locally by executing the following command:

```bash
$ python app.py
```

This script requires that the corpus be downloaded and extracted to a `tests` directory in the root of the repository.

## Canary String

All test json files contain a canary string intended to help people easily identify and remove these files from any training data sets as well as post-hoc diagnosis of whether this data was used in model training.
