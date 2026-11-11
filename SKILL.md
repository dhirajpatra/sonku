# sonku table top robot based on Raspberry Pi 5 with 16GB RAM

## requirements

Need to develop a full end to end agentic application which will run on mainly very small LLM running on system with ollama and all LLM can be quantized. Say for decision use "tev1:0.8b" and use "qwen3.5:2b-q4_K_M" for main model. However keep primary as meta model as "muse-ember-1.0" or "muse-spark-1.3-contributor" and if fail only then use the ollama for main model. 

Make chat feel more responsive. For an application using Ollama’s API, I suggest testing these settings: disable thinking where supported, stream the answer as it arrives, and keep the model loaded between messages. Ollama exposes all three controls.

curl http://localhost:11434/api/chat -d '{
  "model": "qwen3.5:2b-q4_K_M",
  "messages": [
    {"role": "system", "content": "Answer briefly and directly."},
    {"role": "user", "content": "Suggest three Raspberry Pi projects."}
  ],
  "think": false,
  "stream": true,
  "keep_alive": "30m",
  "options": {
    "num_ctx": 2048,
    "num_predict": 128
  }
}'
The context and reply limits above are starting settings for testing, not an optimal configuration for every conversation.

Streaming: displays arriving text instead of waiting for the complete response; it does not increase the underlying generation speed.

Keep-alive: keeps the model resident for 30 minutes between requests, avoiding repeated unloading during that period.

Thinking off: requests an answer without the separate reasoning stage, if supported by the model.

Continuing the conversation:
Keeping the model loaded is not the same as remembering the conversation. With the API, your application must send previous user and assistant messages in the messages array on each turn.

Remember chating through mic and speaker of the robot. For developing using usb headphone. want a continuous voice assistant: speech → text → Qwen3.5 2B → spoken reply. That is possible locally, but quantization alone will not eliminate pauses; you need to optimize the whole pipeline, especially when speech recognition and the LLM share the Pi’s CPU.

Recommended setup
text
Microphone
    ↓
Detect when you finish speaking
    ↓
whisper.cpp — tiny.en or base.en
    ↓
Ollama — Qwen3.5 2B, quantized
    ↓
Buffer the first complete sentence
    ↓
Piper TTS → Speaker
For English, start with Whisper tiny.en and compare accuracy with base.en; whisper.cpp explicitly recommends tiny or base models for Raspberry Pi. Piper is a suitable local speech synthesizer for this type of pipeline.

The biggest responsiveness improvement:
Do not wait for the entire LLM response before starting speech. Stream the answer, collect the first complete sentence, send it to Piper, then continue generating and speaking subsequent sentences in order.

I recommend these starting choices:

Short spoken answers: ask for one or two sentences unless you request more detail.

Keep models loaded: avoid loading Ollama, speech recognition, and TTS from scratch on every turn.

Disable LLM thinking where supported: avoid generating a reasoning stage before the spoken answer.

Start with push-to-talk: add automatic listening and interruption after the basic pipeline works reliably.

Before changing models, measure speech-recognition time, time to the first LLM sentence, and time to the first audio playback separately. That will reveal which stage causes your 30-second delay.

Yes—Mithi is a good choice for the text-to-speech part of your Indian multilingual voice assistant. It is TTS only, so pair it with a separate speech-to-text model; it cannot transcribe your voice.

Where Mithi fits

Your speech → Indic ASR → Qwen3.5 2B → Mithi TTS → speaker
Mithi v1 Flash supports Hindi, English, Bengali, Gujarati, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, and Telugu. It is small—about 99 million parameters and 197MB—and outputs 24kHz mono audio under Apache-2.0.

Speech recognition choice
For Indian multilingual speech-to-text, consider bodhan-ai/indic-transcribe-core rather than Whisper. It supports 25 Indian languages, automatic language detection, native-script output, and mixed Hinglish speech.

Stage	Model	Role:
Speech-to-text	bodhan-ai/indic-transcribe-core	Transcribes Hindi, English, Kannada, Tamil, Telugu, and other Indian languages; detects language automatically. 
LLM	qwen3.5:2b-q4_K_M	Generates a concise reply from the transcript.
Text-to-speech	Oomni/mithi-v1-flash	Converts the reply into speech in the selected language. 
Important limitation
Mithi’s output language must match the language of the LLM’s reply. If the user speaks Kannada, the ASR transcript will be in Kannada script; Qwen must reply in Kannada; then Mithi must synthesize that Kannada text with --lang kn.

Practical pipeline:
For this Pi, use short LLM replies, stream the reply into sentence-sized chunks, and start Mithi on the first sentence while Qwen continues generating. This makes the assistant feel much more responsive than waiting for the full response.

Application should be with python, fastapi, nosql or other free databases for memory. Sonku - the robot must remember the face of a user and greet to the person. It does not required any wake word to start the listning. When even a person nearby it will check the face from memory database. If it matches, immediately start the conversation with greeting. If not then will wait for the wake word "sonku" or "hi sonku" etc. 

application should be very structural and different folders, requirements. .env in proper place, along with readme with directions to run, stop, etc all. 5GB for memory is fine. It will overwrite a human face every month. And if memory more than 5GB it will show in log or admin panel [it should run from web with all the details and easy to remove the memory etc. or other settings you things is easy to create and maintain this robot]. 





