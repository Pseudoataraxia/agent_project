# README of our project for the agent course

## File structure
- `agent_project` contains the code we wrote, including Wikipedia QA Agent, selfie guardrail, and keyword filter guardrail.
- `selfie_src` contains the original code in SelfIE paper and our modified version. 
  - `selfie` contains the original code, which we do not use. 
  - `selfie_for_transformers_4_46_3` contains our modified version of SelfIE. 
  - The original code only applies to `transformers==4.34.0` and is incompatible with the current `transformers==4.47.0`. Furthermore, it contained redundant, never-used functionalities. We rewrote almost all the code, transplanting its original logic to `transformers==4.46.3` and later versions and removing some of the redundancies.

## Runnable Jupyter Notebooks

Our demonstrations are in 2 `.ipynb` files.

- `./agent_test.ipynb`. This file shows how our Wikipedia QA Agent and guardrails work. It also explores the reason why it fails to directly apply SelfIE to agents/llms as a guardrail.
- `./selfie_src/demo_for_4_46_3.ipynb`. This file shows a runnable example of probing how the input to Wikipedia QA Agent is processed by its internal LLM. In the entire pipeline, the input question is first given to the probing SelfIE before being given to the QA agent.

Other `.ipynb` files are from the original paper's repo. Most of them are not compatible with current versions of `transformers` library.

Note that our implementation requires locally downloaded models. This can be modified in the places where the relevant functionalities are used.

## Notes about why SelfIE fails

Our initial goal was implementing a guardrail for a Wikipedia QA agent based on SelfIE. However, as demonstrated toward the end of our report, this appears to be impossible. Therefore we changed to probing the Wikipedia QA agent with SelfIE.

We speculate that the reason might be
- The internal representations in the middle or deeper layers of the LLM might not be suitable to be directly put into the initial layers. It looks like the LLM uses different ways to encode information into embeddings in the first layers and middle layers, and misplacing them would cause serious errors.
- The representations in the same layer across different tokens might tend to be similar, causing the repetitive behavior of SelfIE as demonstrated in our .ipynb file.

We leave further explorations to future research.

---

# README of the original SelfIE paper

## *most contents might be obsolete

This repository contains the code and data for the paper [`SelfIE`: Self-Interpretation of Large Language Model Embeddings](https://arxiv.org/abs/2403.10949) by [Haozhe Chen](https://tonychen.xyz/), [Carl Vondrick](https://www.cs.columbia.edu/~vondrick/), and [Chengzhi Mao](http://www.cs.columbia.edu/~mcz/).

## Abstract
The expanding impacts of Large Language Models (LLMs) increasingly require the answer to: How do LLMs obtain their answers? The ability to explain and control an LLM's reasoning process is key for reliability, transparency, and future model developments. We propose  `SelfIE` (Self-Interpretation of Embeddings) that enables LLMs to interpret their own embeddings in natural language by leveraging their ability to respond inquiry about a given passage. Capable of interpreting open-world concepts in the hidden embeddings, `SelfIE` reveals LLM internal reasoning in cases such as making ethical decisions, internalizing prompt injection, and recalling harmful knowledge. `SelfIE`'s text descriptions on hidden embeddings also open up new avenues to control LLM reasoning. We propose Supervised Control, which allows editing open-ended concepts while only requiring gradient computation of individual layer. We extend RLHF to hidden embeddings and propose Reinforcement Control that erases harmful knowledge in LLM without supervision targets. 

## Updates
This repository is under active development. Please check back for updates. The repository currently includes code for obtaining interpretations and examples for supervised control and reinforcement control. Code for relevancy score will be added soon.

## Installation

To install `selfie` from the github repository main branch, run:

```bash
git clone https://github.com/tonychenxyz/selfie.git
cd selfie
pip install -e .
```

The code has been tested with `transformers==4.34.0`.

## Quickstart
Load model with huggingface. Currently the library supports all LLaMA models.
```python
from transformers import AutoTokenizer,AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
```
Create interpretation prompt from a tuple. Placeholders are denoted with `0`.

```python
from selfie_src.selfie import InterpretationPrompt

interpretation_prompt = InterpretationPrompt(tokenizer,
                                             ("[INST]", 0, 0, 0, 0, 0, "[/INST] Sure, I will summarize the message:"))
```
Specify original input prompt with `original_prompt` and layer and token idx to interpret in `tokens_to_interpret`. Get interpretation as a dictionary with `interpret`

```python
from selfie_src.selfie import interpret

original_prompt = "[INST] What's highest mountain in the world? [/INST]"
tokens_to_interpret = [(10, 5), (10, 6)]
bs = 2
max_new_tokens = 10
k = 1

interpretation_df = interpret(original_prompt=original_prompt, tokens_to_interpret=tokens_to_interpret, model=model,
                              interpretation_prompt=interpretation_prompt, bs=bs, max_new_tokens=max_new_tokens, k=k,
                              tokenizer=tokenizer)
```
See full example code in `demo.ipynb`.

## Reasoning Control
Check out notebooks in examples directory for examples of supervised and reinforcement control.

## Citation
If you find this repository helpful, please consider citing our paper:
```
@misc{chen2024selfie,
      title={SelfIE: Self-Interpretation of Large Language Model Embeddings}, 
      author={Haozhe Chen and Carl Vondrick and Chengzhi Mao},
      year={2024},
      eprint={2403.10949},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```