# Use a pipeline as a high-level helper
from transformers import pipeline
import sounddevice as sd
import numpy as np
import wavio
import pdb
from scipy.io import wavfile
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def record_audio(file_path, duration, samplerate=44100):
    print("Recording...")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    print("Recording done.")
    file_path = "recorded_audio.wav"
    wavio.write(file_path, audio_data, samplerate, sampwidth=3)



def main():
    print(30*'-')
    print(sd.query_devices())
    print(30*'-')
    record_audio("/AudoFiles/my_voice.wav",duration=10)

def huggingface(dataset_file):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("inference on: ",device)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-small-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    if dataset_file=="":
        dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
        sample = dataset[0]["audio"]
    else:
        sample = dataset_file
    

    result = pipe(sample)
    print(result["text"])


if __name__=="__main__":
    #main() recorded_audio.mp3
    file='recorded_audio.wav'
    # pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2")
    # samplerate, data = wavfile.read(file)
    huggingface(dataset_file=file)
    pdb.set_trace()
