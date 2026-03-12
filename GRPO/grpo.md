# DeepSeek V3 - Engineering Deep Dive

optimizes the 2 main bottleneck found in the earlier architectures - kv cache and training throughput

## 1. Architecture - MLA

At the time of inference the main bottleneck is loading the KV Cache which poses a memory bandwith and not compute. If we use standard MHA - the size of the KV Cache will be huge and it will kill the throughput. 

### The innovation:
The paper uses multi head latent attention in which the KV Cache is compressed into a latent vector (low rank compression).

**Maths**
Instead of storing the d-dim keys and values we project them into a much smaller latent space 

$$h_t = W_{DKV} \cdot c_t^{KV}$$

Example:

Let us say you have a hidden dimension of 512. You store a 512 dimension key vector and 512 dimension value vector thus making the total stored to be 1024. This is what happens in standard MHA.

DeepSeek MLA compresses the 512-dim vector into a latent vector of size 64. So during your inference you just carry this 64-dim vector and when you need to do any math you "up-project" it back to 512.

How does it happen (in detail):

1. The Compression (Down-Projection):
In a standard transformer, for every token, you generate a Key ($K$) and a Value ($V$). If your model dimension ($d_{model}$) is 512, you're looking at:
$K$ vector: 512 elements
$V$ vector: 512 elements
Total: 1024 floating-point numbers stored in GPU VRAM for every single token in the context.
MLA says: "That's redundant." Most of those 512 dimensions are highly correlated. So, instead of storing them, MLA projects the input into a compressed latent vector ($c_t^{KV}$).$$c_t^{KV} = W_{DKV} \cdot x_t$$Where $W_{DKV}$ is a down-projection matrix. If the latent dimension is 64, we just squeezed 1024 units of information into a 64-dimensional bottleneck. This 64-dim vector is the only thing that stays in the KV cache.

2. The Reconstruction (Up-Projection):
When the model needs to actually perform the Attention mechanism (Query $\times$ Key), it can't do it with a 64-dim vector if the Query is 512-dim. It needs the dimensions to match. At inference time, right before the math happens, MLA "unpacks" the latent vector using an up-projection matrix ($W_{UK}$ for Keys and $W_{UV}$ for Values):Key Reconstruction: $K_t = W_{UK} \cdot c_t^{KV}$Value Reconstruction: $V_t = W_{UV} \cdot c_t^{KV}$
The "aha!" moment here is that these up-projection matrices are part of the model weights, not the KV cache. They stay static on the GPU. You only "pay" the memory price for the 64-dim vector per token.

3. Absorbing the Weights (The Inference Hack):
DeepSeek takes this a step further to avoid the compute overhead of up-projecting every single time. In the Attention equation:$$Score = Q \cdot K^T$$$$Score = Q \cdot (W_{UK} \cdot c_t^{KV})^T$$Because of the associative property of matrix multiplication, we can pre-multiply the Query ($Q$) by the up-projection matrix ($W_{UK}$) before looking at the KV cache.Project the Query: $Q' = Q \cdot W_{UK}$
Direct Match: $Score = Q' \cdot c_t^{KV}$
Now, you are doing the attention math directly against the compressed 64-dim latent vector. You never actually "materialize" the full 512-dim Key in memory.

## 2. The MOE Strategy

In standard architectures like GPT-3 (dense models) every single parameter is involved in calculating every single token. MoE is a "sparse" architecture which splits the one giant FFN into smaller layers called "experts".

So think like this, you have many small experts that will handle tokens. So before the tokens arrive to these experts they should go somewhere from where they will be alloted an expert right? For this we have "router". It is a tiny learnable neural network who has one job : "Look at the token and decide - which expert is good enough to handle this". 
Now for the experts, these are normal FFNs which are good at different things. Some might be good at solving C++ code, while some might be good at writing english passages. 
And last but not the least - what makes MoE sparse? So for each token only some of these experts are fired up, not all of them and that is why it is considered sparse in nature.

In DeepSeek-V3 we have 671B parameters but only 37B are activated.

### The Problem : Route Collapsing

Think of it like this, initially everything is random so all the experts are random, this means that for the first few tokens "expert 1" might be better than all the other experts and the router will keep on sending the tokens to it and since expert 1 gets more data it trains and becomes even more smarter. This leads to overloading on expert 1 while the other experts are idle and become wasted VRAM. 

Traditional fix for the above problem was simple. If you think about it, the most basic way to counter this issue was to see if the model is doing this (giving tokens to just one expert and not equally distributing tokens) then at that time you will penalize the model. But this fix is dumb because you are trying to 

DeepSeek's Aux-Loss-Free Strategy

Very simple. Do not force the balance between the experts using the loss function but through "Dynamic Routing Bias". Fancy term simple explanation. Instead of using a fixed penalty use an affinity score for each expert that adjusts during fly during training. 

Normally, the Router calculates a score ($s$) for each expert $i$:$$s_i = \text{Softmax}(\text{Router}(x))$$

DeepSeek adds a bias term ($b_i$) to this score:$$\hat{s}_i = \text{Router}(x) + b_i$$

* If an expert is overloaded (taking too much traffic), the system automatically decreases its $b_i$.
* If an expert is underutilized, the system increases its $b_i$ to make it more "attractive" to the router.

Why is this beautiful? 
Because now the model just focuses on being smart. Previously adding a loss term the model's gradient (learning process) used to focus on both being smart and being fair. But now since a bias term the balancing happens on the routing level and not during the training.

## 3. The Training Objective : Multi-Token Prediction(MTP):

Again as the name suggest - don't go for the "next token prediction" but for "multiple token prediction". What good is in this? Well, we make the model a bit far sighted.

In a standard transformer the hidden state $h_i$ at position $i$ is used to predict token $t_{i+1}$. In DeepSeek-V3, they add extra "MTP modules."

For each additional token you want to predict (let's say we want to predict 2 tokens at once), the model does the following:
Main Path: Calculates $h_i$ and predicts $t_{i+1}$ (Standard NTP).
MTP Path: Takes that same hidden state $h_i$, combines it with the embedding of the predicted token $t_{i+1}$, and runs it through an additional MTP Module (a shared transformer block) to predict $t_{i+2}$.

The loss function becomes a weighted sum:$$\mathcal{L} = \mathcal{L}_{NTP} + \lambda \sum_{k=1}^{K} \mathcal{L}_{MTP}^{(k)}$$Where $K$ is the number of future tokens being predicted. 

Think of this as "Information Compression." 

 Standard NTP: The model might learn that after "The cat sat on...", the next word is probably "the". It doesn't need to know what the cat sat on yet; it just needs to get the next word right.
 
 MTP: The model is forced to predict "the" AND "mat" simultaneously. To do this, the internal representation (the vector $h_i$) must encode the concept of the "mat" much earlier in the computation.

 The most genius part of MTP isn't just that it makes the model smarter—it makes it faster. Usually, LLMs are autoregressive, meaning they generate one token at a time. To get 10 tokens, you have to run the model 10 times. This is the "Memory Wall" problem you see in inference optimization.
 
 Speculative Decoding with MTP works like this:
 * The model generates token $n+1$.
 * Simultaneously, the MTP modules "guess" tokens $n+2, n+3$, etc.
 * Because these guesses are produced in a single forward pass, they are "free" (computationally speaking).
 * In the next step, the model verifies these guesses. If they are correct, you just generated 3 tokens for the price of 1.


# 4. The Infra Layer

In a 671B model, memory bandwidth is the killer. Standard (BF16) takes 2 bytes per number while DeepSeek used FP8 which takes 1 byte per number.

So again think like this, you reduced the precision by half - now you have saved memory thus doubled your compute throughput but if you just reduce the bits the model's grads will vanish or explode. To solve this DeepSeek uses  "fine-grained scaling". Instead of using a single scale factor for an entire layer, they divided the tensor into many small blocks and each of these blocks receive their own scaling factor.

The intuition behind this is very simple. Each part of the weight matrix has different value ranges. If one region has numbers between -0.1 and 0.1 while the other has between -20 and 20, a single global scaling factor cannot do justice to both of them. Block wise scaling allows each region of the tensor to use the FP8 range efficiently. 

The next part is : DualPipe

A 671B param model cannot fit on a single GPU. It needs to be split acorss many GPUs. One common strategy that is used is pipeline parallelism. 

In pipeline parallelism the model is divided into stages across the GPUs. A micro batch flows throught the pipeline:

GPU1 → GPU2 → GPU3 → GPU4

First the forward pass moves through it followed by the backward pass. 
The inefficiency appears because GPUs often sit idle while waiting for data from another stage. For example, GPU1 may finish the forward computation and send the result to GPU2. While GPU2 processes the data, GPU1 may have nothing to do. These idle periods are called pipeline bubbles.

In large systems these bubbles can waste 30–50% of available compute.

DeepSeek addresses this with DualPipe, a bidirectional pipeline strategy.

Instead of running only forward passes in sequence and then backward passes, the system overlaps different operations. While one micro-batch is moving forward through later pipeline stages, earlier GPUs begin processing the forward pass of another micro-batch or the backward pass of a previous one.

As a result, forward and backward computations are happening simultaneously across the pipeline. Work flows in both directions. The idle gaps are filled with useful computation, and the GPUs remain busy almost all the time.

The result is much higher utilization of the hardware.

DeepSeek engineers also noticed that waiting for NVLink (GPU-to-GPU communication) is another bottleneck. In DualPipe, they hide the communication behind the computation.

While the GPU is crunching the numbers for the "Attention" layer, the "All-to-All" communication for the next "MoE" layer is already happening in the background.

By the time the math is done, the data for the next step is already there.

# GRPO - PPO Killer

1. The First Principle: Why do we need RL?

    In standard training (SFT), we tell the model: "Here is the prompt, and here is exactly what you should say." In Reinforcement Learning (RL), we tell the model: "Here is the prompt. Try a few things, and I will give you a score (Reward) based on how well you did."

    RL is essential for reasoning (like in DeepSeek-R1) because there isn't always one "correct" path to a solution; the model needs to explore different ways of "thinking" to find the most efficient one.


2. The PPO Bottleneck (The "Critic" Problem)

    In standard PPO, you have to maintain at least two massive models in VRAM:

    The Actor (Policy): The model you are actually training (e.g., DeepSeek-V3 671B).

    The Critic (Value Function): A separate model of the same size that tries to predict how much reward the Actor will get.

    Why do we need the Critic? To calculate the Advantage.

    If the model gets a reward of 0.8, is that good? We don't know unless the Critic tells us, "Normally, for this prompt, you only get 0.5." Now we know 0.8 is great ($0.8 - 0.5 = +0.3$ advantage).

    The Cost: Keeping a 671B Critic in memory alongside a 671B Actor is practically impossible for most labs. It's a "Memory Tax" that doubles your hardware requirements.

3. The GRPO Solution: Intelligence via Comparison

    GRPO's "Aha!" moment is realizing that you don't need a Critic model to tell you what's "normal." You can just look at what the model is currently doing.

    How it works:

    Group Generation: For a single prompt, you ask the model to generate a group of outputs (let’s say $G=8$ different responses).

    The Reward: You pass all 8 responses through your Reward Model (or a rule-based checker).

    You get 8 scores: $\{r_1, r_2, \dots, r_8\}$.

    The Advantage (The Math): Instead of comparing a response to a Critic's prediction, you compare it to its peers.$$A_i = \frac{r_i - \text{mean}(r_1, \dots, r_8)}{\text{std}(r_1, \dots, r_8)}$$Where:$r_i$ is the reward of the current output.

    $\text{mean}$ is the average of the group.

    $\text{std}$ is the standard deviation (to keep the numbers stable).

4. Why this works

    Self-Correction: If the model generates 8 solutions to a math problem and 2 are correct while 6 are wrong, the 2 correct ones will have a much higher reward than the group average. The model is forced to "shift its weight" toward the logic used in those 2 responses.

    VRAM Efficiency: By deleting the Critic model, DeepSeek saved nearly 50% of the VRAM required for RLHF. This allowed them to allocate that memory to larger batch sizes or higher-resolution training.

    Reduced Training Noise: Because the "baseline" is derived from the actual current outputs of the model (not a lagging Critic model's prediction), the training signal is often much cleaner and more stable.

5. GRPO in the Context of DeepSeek-R1 (Reasoning)GRPO is what made the "Aha!" moment possible for DeepSeek-R1. When training for reasoning, the "Reward" isn't a human's opinion—it's a Rule-Based Reward.Accuracy Reward: Does the code run? Is the math answer correct?Format Reward: Did the model put its thinking process inside <think> tags?Because GRPO allows for very large group sizes (sampling 16 or 32 versions of the same thought process), the model can "see" a wide variety of reasoning paths and quickly learn which ones lead to the correct answer.
