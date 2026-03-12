# Problem statement
LLM alignment requires a two step process - SFT followed by RLHF/PPO or DPO which is computationally expensive as it requires training of a reference model to prevent the agent or the policy form drifting away from its natural distribution that is learned during training.

[paper](https://arxiv.org/pdf/2403.07691)
## Dig deep

* SFT teaches the model how to talk. llms are just next token generators but how should they generate these tokens, what should be the format of these responses is something that is learnt during SFT. Preference alignment like PPO DPO teaches the model what to prefer -which response is good and which is bad is learnt during preference alignment.

* Preference alignment methods like PPO and DPO require for us to keep the weights of the reference model present ( in a frozen state )in VRAM. This becomes a very big memory bottleneck. Also it is to be noted that during SFT there is no mathematical mechanism to penalise the rejected responses. sft just makes sure that the model learns the style of response, it is just copying what it needs to output - it will learn the style of the rejected response as much as it will learn the style of the accepted one. 

## historical context

* PPO uses a reward model and rl. the main bottleneck with ppo was that it required 4 models to be present in memory - reference model, policy, reward model and value model.

* DPO replaces reward model with a closed form solution. more stable than ppo but required reference model to calculate KL Divergence which prevented the model from drifting much away.

# Paper Mentality
> "SFT role is to adapt the model to the target domain... however, it also inadvertently increases the probability of generating undesired tokens in the rejected responses."

SFT though teaches the model the style of response it wants it to generate but it also validates bad behaviour as there is no penalty given to such responses.

>"DPO relies heavily on the reference model... which is computationally expensive and memory-intensive."

> "ORPO is a monolithic preference optimization... it does not require a separate reference model."

they wanted a one and done loss function term and monolithic means that they see the current multi stage pipeline as unnecessarily decoupled.

# Novel Contribution

## one liner

adds the preference alignment directly into the SFT objective by adding a Odds Ratio penalty that discourages the model from favouring rejected responses relative to the chosen ones.

## conceptual leap

well you do not need a reference model to define a good direction. the authors used an internal contrast between the chosen and rejected pair in same batch to provide the signal. they realised that "odds" of a signal are much stable signal than raw probabilities.

# derivation

## the odds:

what are odds of an event x? ratio of its probability to the probability of not occuring

$$
\text{odds}_{\theta}(y|x) = \frac{P_{\theta}(y|x)}{1 - P_{\theta}(y|x)}
$$

If $P = 0.8$, the odds are $0.8/0.2 = 4$. If the model is very confident, the odds are high.

## defining ratio 

we want the odds of the correct response to be higher than the odds of the wrong response 

for this we define the odds ratio (OR) : 

$$
OR_{\theta}(y_w, y_l) = \frac{\text{odds}_{\theta}(y_w|x)}{\text{odds}_{\theta}(y_l|x)}
$$

## logit shift

to make the above differentiable we take the log

$$
\log OR_{\theta}(y_w, y_l) = \log \left( \frac{P_{\theta}(y_w|x)}{1 - P_{\theta}(y_w|x)} \right) - \log \left( \frac{P_{\theta}(y_l|x)}{1 - P_{\theta}(y_l|x)} \right)
$$

## orpo loss component

so to maximise the ratio we will minimize the negative log-sigmoid of this ratio and push the correct further away from the loser

$$
\mathcal{L}_{OR} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \log \frac{\text{odds}_{\theta}(y_w|x)}{\text{odds}_{\theta}(y_l|x)} \right) \right]
$$

## final monolithic objective

orpo combines standard sft loss with the odd ratio loss so that we are teaching both style and preference to the model

$$
\mathcal{L}_{ORPO} = \mathbb{E} [ \mathcal{L}_{SFT} + \lambda \cdot \mathcal{L}_{OR} ]
$$

* If $\lambda = 0$: It's just standard SFT.

* If $\mathcal{L}_{SFT}$ is removed: The model might learn preferences but forget how to construct coherent sentences (the "drift" problem).

# methodology 

## algorithm - plain english

1. Take a dataset where each prompt has a "good" and "bad" answer.

2. Feed both answers into the model.

3. Calculate how much the model likes the good answer vs. the bad answer (Odds).

4. Update the model weights so that:

    * The good answer's probability stays high (SFT part).

    * The gap between the "odds" of the good and bad answer grows wider (OR part).

5. Do this in one single training run.

## pseudocode

```python
def orpo_loss(logits_w, logits_l, labels_w, labels_l, alpha=0.1):
    # 1. Standard SFT Loss (Negative Log Likelihood)
    l_sft = cross_entropy(logits_w, labels_w)
    
    # 2. Calculate Log Odds for both
    # log_odds = log(P / (1-P))
    log_p_w = get_log_prob(logits_w, labels_w)
    log_p_l = get_log_prob(logits_l, labels_l)
    
    log_odds_w = log_p_w - torch.log1p(-torch.exp(log_p_w))
    log_odds_l = log_p_l - torch.log1p(-torch.exp(log_p_l))
    
    # 3. Odds Ratio Loss
    odds_ratio = log_odds_w - log_odds_l
    l_or = -log_sigmoid(odds_ratio)
    
    # 4. Combine
    return l_sft + alpha * l_or

```

# edge cases and limitations

1. Length Bias: Like DPO, ORPO can be tricked by longer responses. If "chosen" is always longer, it might just learn to be wordy.

2. Memory: While it saves the Reference Model memory, it requires processing two sequences ($y_w$ and $y_l$) in the same batch, which increases the activation memory.