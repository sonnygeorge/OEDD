# OEDD

Official codebase for the paper _Probing the Capacity of Language Model Agents to Operationalize Disparate Experiential Context Despite Distraction_ (published at EMNLP Findings 2024)

## OEDD Corpus of Reasoning Tests for LLM Agents

The OEDD (Operationalize Experience Despite Distraction) corpus is a collection of reasoning tests designed to evaluate the capacity of language model agent systems to make smart action-inferences despite plausible distractions.

## Code Structure

```text
.
├── assets                    # contains .svgs for app.py
├── src                       # main source code
│   ├── models.py             #   Pydantic data objects
│   └── utils.py              #   helper functions
├── templates                 # prompt templates
├── tests                     # OEDD tests (omitted from version control, downloadable from links below)
├── app.py                    # runs NiceGUI app on local machine to visualize corpus
├── figures.py                # generates figures and statistical significance test results
└── run_tests.py              # script to run tests with GPT-3.5-Turbo, GPT-4o, and Gemini 1.5 Pro
```

## Downloads

The following are download links to different versions of the test corpus. Please download this `tests` directory and add it to the root of the repository before running anything.

- [v1.0.0](https://drive.google.com/drive/folders/1uO7KggJr2HZ9c6KcGfrrgekYcrypKPY0?usp=sharing)


Our `results.csv` from our initial experiments using v1.0.0 of the corpus can be downloaded [here](https://drive.google.com/file/d/105Ravdk8Rbhi1UT15p5fj0MB0tIlCway/view?usp=sharing).

## Version History

We consider this a living corpus and encourage community scrutiny, feedback, and contributions.

Corpus updates and justifications will be documented here:

| Date | Version | Comments |
|-|-|-|
| 10/3/2024 | 1.0.0 | Initial release |

To suggest changes to the corpus, please contact the repository owner **_privately_** with your suggestions, additions, etc.

Please refrain from discussing the contents of the corpus or potentional additions to the corpus in public forums (including Github Issues) to avoid leaking content into LLM training sets and biasing future evaluations.

## Canary String

All test json files contain a canary string intended to help people easily identify and remove these files from any training data sets as well as post-hoc diagnosis of whether this data was used in model training.

## Corpus Visualization App

We provide a custom [NiceGUI](https://github.com/zauberzeug/nicegui) application that allows users to more intuitively explore the content of the OEDD tests.

It can be run locally by executing the following command (after installing dependencies in `requirements.txt`):

```bash
$ python app.py
```

This script requires that the corpus be downloaded and extracted to a `tests` directory in the root of the repository.

![app demo](https://media3.giphy.com/media/p19gMWjWZIDVWJOMSb/giphy.gif)
