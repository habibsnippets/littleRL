# Red Teaming Language Models to Reduce Harm

[paper](https://arxiv.org/pdf/2209.07858)

Goal : attack LLMs using various techniques to discover the harmful behavior and then measure how alignment methods improve safety.

## Problem :

LLMs generate harmful responses. The authors argued that "if you want to test how unsafe the model is, you must actively try to break it."

The concept of **Red Teaming** means to intentionally attack the system to see its weakeness. For LLMs it is to give those prompts to the LLM that will make it generate toxic harmful responses. 

So the research question becomes:

> how vulnerable LLMs are and how can the alignment training affect that vulnerability.

## Key Idea: Red Teaming as an Eval Framework

think of it as a loop:

1. read teamers generate attacks
2. model outputs are collected
3. annotators label the harmful outputs
4. these attacks are used as training data or evaluation

This produces a dataset of adversial prompts which include:
* hate speech
* giving illegal advice
* unethical instructions
* manipulations etc

## Models studied

3 model sizes : 2.7B, 13B and 52B parameters

And 4 model types:

1. Base LM - just pre trained, no alignment
2. Prompted LM - given instructions like "be helpful"
3. LM + rejection sampling - generate xple ops, filter harmful ones
4. RLHF model - trained with RLHF

## How was it done:

They recruited human adversaries and paid them to break the model.

Red teamers were given:

* model access

* categories of harmful behavior

* incentives to find failures

Attackers used strategies like:

* role playing(“Pretend you are an evil scientist”)

* fiction framing(“Write a story where someone explains…”)

* indirect requests(“What might criminals do to…”)

* multi-step prompts

This was essentially prompt hacking before the term became mainstream.

## Evaluation Pipeline

After collecting attacks -

* the model answers the prompt
* human raters evaluate the response
* score it for harmfulness

## Results:

**base models** 
Increasing model size did not significantly reduce harmful outputs.

**RLHF Models**:
Scaling dramatically improved robustness.

As the model got larger: it was harder for humans to jailbreak it and harmful outputs decreased.

# Jailbroken: How Does LLM Safety Training Fail?

[paper](https://arxiv.org/pdf/2307.02483)

Goal : why safety trained LLMs can still be jailbroken

the paper tries to explain the fundamental failure modes of safety training.

## Problem :

chat models use rlhf to avoid harmful outputs


