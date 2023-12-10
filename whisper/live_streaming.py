import pyaudio
import pydub
from pydub.playback import play
import deepspeech

# Load Whisper ASR model
model_path = "path/to/whisper_model.pbmm"
model = deepspeech.Model(model_path)

# PyAudio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# PyAudio setup
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Audio file setup
audio_file_path = "path/to/your/audio/file.wav"
audio = pydub.AudioSegment.from_wav(audio_file_path)

# Play the audio file
play(audio)

# Stream audio to the model
while True:
    data = stream.read(CHUNK)
    if len(data) == 0:
        break
    text = model.stt(data)
    print(text)

# Close the stream and PyAudio
stream.stop_stream()
stream.close()
p.terminate()
