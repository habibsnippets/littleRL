# Scaling Laws for Reward Model

notes from [Scaling Laws for Reward Model](https://arxiv.org/pdf/2210.10760)


## Introduction

to train ai models we need a teacher which can tell what is right and what is wrong. 

thinking about it the best teacher to teach our ai model about it is a human - but there are constraints - humans are slow and inefficient. to solve this we use a "proxy reward model" which behaves like a mini teacher and tries to mimic the human behavior. 

the problem with this approach is that this reward model is never perfect and it has its loopholes. 

when we try to optimize a policy it tries to learn what the reward model is trying to make it learn. for instance if it is told that generating concise code is ideal and will be highly rewarded then it will foucs only on that while ignoring the fact that quality of code also matters. it implies the fact that instead of gaining knowledge(gold reward) it tries to learn about the shortcuts so that it can maximise the reward from the proxy reward model. 

the authors in the paper try to use the scaling laws to make predictions about when the system will fail. 

**Methodolody**: 
so they made a gold reward model (perfect truth) and the proxy reward model and observed how the gold score changes as we optimized it further (KL D)

**best of n sampling** as discussed in the paper

suppose you ask ai a question and instead of generating a single response it starts generating n different versions of the answer.  the proxy rm acts like a judge and gives each answer a score and then finally you pick the one with highest score. but what happens during "over-optimization" is that n gets very large and the chance of getting an answer that might trick the RM increases. 

**RL**:
unlike BoN where the model stays the same and we just pick the best ouput as the final answer, here in RL we change the weights of the model thus updating it. 

the process is simple: ai generates an answer, the RM gives it a score and through algos like PPO the AI is told whether the answer is good or bad and wrt the model is made to update its weights. over many steps the entire personality of the model changes and it gets inclined to the reward model's behavior. 

problem during over-optimztion: model becomes good at generating responses that the RM will like but since this RM is just a proxy (an imperfect copy of the human taste) the ai starts to drift away from the human preferences. 

thus in both cases, if you start to over-optimize (large n values in BoN or too many steps in RL) the proxy score keeps on going up but the gold score (actual model quality) starts crashing.

## Methodology

the setup - instructGPT like

* input : prompts - instructions or questions
* output : response gen by the model
* goal : use RM to grade these responses and use these grades to improve the AI

the model used - 
* base model : gpt3
* initial training (sft): model is trained on high qulaity answers written by humans to give a starting point
* RM architecture : also a GPT 3 model  but has a scalar head (meaning instead of words we generate a number - score )

to study over-optimzation without spending millions on human labels the authors created a synthetic ground truth. 

* gold RM : take a 6B param RM and treat its score as the absolute truth (act like a human)
* proxy RM : train smaller models to copy the gold RM
* logic : if the proxy RM is optimized too much we can check the results against the gold RM to see if the quality dropped.

    