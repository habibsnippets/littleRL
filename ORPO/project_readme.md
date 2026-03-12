# what is orpo 

link

# components of audio qwen chat

## audio encoder - whisper large v2

job is to convert waveform -> semantic audio features

Pipeline :

waveform
   ↓
log-mel spectrogram
   ↓
Whisper encoder transformer
   ↓
audio feature vectors

ouput is a sequence:

A=(a1​,a2​,...,aT​)

where each a_i is a representation of a small time window of audio

and - encoder is frozen , reasons:

* Whisper is already extremely good
* training audio encoders is expensive
* freezing stabilizes multimodal training

## cross attention fusion layer

inject audio into the llm, but llms expect tokens and not audio vectors right so the trick will be to treat audio features like pseudo tokens. the arch adds cross attention layers where 

• queries = LLM hidden states
• keys/values = audio embeddings

so that the text representation can attend to audio as well. 

## qwen 7b langauge model

normal decoder only tf, once audio embeddings are fused it behaves just like a normal LLM. 