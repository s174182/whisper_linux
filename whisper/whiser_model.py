from transformers import pipeline
import sounddevice as sd
import numpy as np
import pdb
from scipy.io import wavfile
import os
import torch, torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, TextStreamer
from pydub import AudioSegment



class transcriber:
    def __init__(self, model_id):
        self.model_id = model_id
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("inference on: ",self.device)
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.streamer = TextStreamer(self.processor)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
            streamer=self.streamer
        )
    def transcribe(self,data):
        result = self.pipe(data)
        print(result["text"])
class transcriber_live:
    def __init__(self, model_id):
        self.model_id = model_id
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("inference on: ",self.device)
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.whisper_sample_rate=16000
        # generate token ids
        self.streamer=TextStreamer(self.processor,skip_special_tokens=True)
    def transcribe(self,file):
        print("began transcription")
        data, sample_rate = torchaudio.load(file)
        data=np.array(data).squeeze()
        input_features = self.processor(data, sampling_rate=sample_rate, return_tensors="pt").input_features
        input_features = input_features.to(self.device).half()
        print("input to model")
        predicted_ids = self.model.generate(input_features,streamer=self.streamer)
        # transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        # print(transcription)
def resample(input_file_path):
    output_file_path = "resampled_"+input_file_path

    # Load the audio file
    audio = AudioSegment.from_wav(input_file_path)

    # Set the target sample rate (16000 Hz)
    target_sample_rate = 16000

    # Resample the audio
    resampled_audio = audio.set_frame_rate(target_sample_rate)

    # Export the resampled audio to a new file
    resampled_audio.export(output_file_path, format="wav")

if __name__=="__main__":
    model_id="openai/whisper-small"
    whisper_model=transcriber_live(model_id=model_id)
    resample('recorded_audio.wav')
    file="resampled_recorded_audio.wav"

    whisper_model.transcribe(file=file)
    pdb.set_trace()
