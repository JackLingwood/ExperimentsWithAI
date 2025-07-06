# Embedding text and upserting to Pinecone Vector Database
# https://www.pinecone.io
# https://sbert.net/#
# Setup PineCone Basic Vector Database 2
# ---------------------------------------------------------------------------------------------
from dotenv import load_dotenv, find_dotenv
import os
import sys

# Add Shared folder to sys.path
sys.path.append(os.path.abspath("Shared"))

from utils import heading, heading2, heading3, clearConsole, note, print_ascii_table , highlight_differences_diff_based

from openai import OpenAI

# Load environment variables
load_dotenv(find_dotenv(), override = True)

# Clear console and set working directory
clearConsole()

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Current working directory:", os.getcwd())

# Headings
name = os.path.basename(__file__)
heading(f"{name}")
api_key = os.environ.get("api_key")
pinecone_vector_database_key = os.environ.get("pinecone_vector_database_key")
pinecone_environment = os.environ.get("pinecone_environment", "gcp-starter")
# ---------------------------------------------------------------------------------------------





import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import speech_recognition as sr
from jiwer import wer, cer
from IPython.display import Audio
import whisper
import csv
import os
import tempfile
import wave
#from gtts import gtts
from gtts import gTTS


audio_signal, sample_rate = librosa.load('speech_01.wav', sr=None) # sr=None to preserve the original sample rate

heading2("Audio Signal", "Loaded audio signal from 'speech_01.wav'")
heading3("Sample Rate", sample_rate)

plt.figure(figsize=(12, 4)) # Plot the waveform
librosa.display.waveshow(audio_signal, sr=sample_rate) # Plot the waveform
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()


# Play the audio in the notebook
# Audio('speech_01.wav') # Load the audio file -- This is for Jupyter Notebook

file_path = 'speech_01.wav'


def play_audio(file_path):
    os.system(f"start {file_path}") # This is for Windows to play the audio file

#play_audio(file_path) # Play the audio file using the function defined above

recognizer = sr.Recognizer()

def transcribe_audio(file_path):
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source) # record captures the entire audio file
        note("Transcribing audio file...")
        text = recognizer.recognize_google(audio_data)
        heading2("Transcript", text)
        return text    
    
transcribed_text = transcribe_audio(file_path)

ground_truth = """My name is Ivan and I am excited to have you as part of our learning community! 
Before we get started, I’d like to tell you a little bit about myself. I’m a sound engineer turned data scientist,
curious about machine learning and Artificial Intelligence. My professional background is primarily in media production,
with a focus on audio, IT, and communications"""

def report_transcription_accuracy(ground_truth, transcribed_text):
    calculated_wer = wer(ground_truth, transcribed_text)    
    calculated_cer = cer(ground_truth, transcribed_text)
    data = [
        ["WER", "CER"],
        [f"{calculated_wer:.4f}", f"{calculated_cer:.4f}"]
    ]
    print_ascii_table(data)
    highlight_differences_diff_based(ground_truth, transcribed_text)

note("Transcription Accuracy Report:")
report_transcription_accuracy(ground_truth, transcribed_text)

# ground truth is the expected text that the audio should be transcribed to.
# transcribed_text is the text that was actually transcribed from the audio file.   


# WER stands for Word Error Rate, which measures the accuracy of the transcription by comparing the number of words in the ground truth to the number of words in the transcribed text.
# CER stands for Character Error Rate, which measures the accuracy of the transcription by comparing the number of characters in the ground truth to the number of characters in the transcribed text.

# WER = (Substitutions + Insertions + Deletions) / Total Words in Ground Truth -- All counts are case-sensitive and per word, including punctuation and spaces.

# CER = (Substitutions + Insertions + Deletions) / Total Characters in Ground Truth -- All counts are case-sensitive and per character, including punctuation and spaces.

plt.figure(figsize=(12, 4))
librosa.display.waveshow(audio_signal, sr=sample_rate)
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

#play_audio(file_path) # Play the audio file using the function defined above

heading("Spectrogram - Visualizing the frequency content of the audio signal")

# Compute the spectrogram
S = librosa.stft(audio_signal)
# using Short-Time Fourier Transform (STFT) to compute the spectrogram
# STFT is a method to analyze the frequency content of a signal over time

S_dB = librosa.amplitude_to_db(abs(S), ref=np.max) # convert the amplitude to decibels (dB) for better visualization

heading2("np.max(S_dB)", np.max(S_dB))
#np.max(S_dB)

# Plot the spectrogram
plt.figure(figsize=(12, 4))
librosa.display.specshow(data = S_dB, sr=sample_rate, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB') # colorbar to show the dB scale
plt.title('Spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()


heading("REMOVING NOISE - Pre-emphasis Filter - Applying a pre-emphasis filter to the audio signal")

signal_filtered = librosa.effects.preemphasis(audio_signal, coef=0.97)
sf.write('filtered_speech_01.wav', signal_filtered, sample_rate)
output_file = 'filtered_speech_01.wav'

# Emphasizing high frequencies helps to reduce noise and improve the clarity of the speech signal.
# The pre-emphasis filter is a high-pass filter that amplifies the high-frequency components of the signal.
# The coefficient (coef) controls the amount of emphasis applied to the high frequencies.
# Save the filtered audio signal to a new file


# Play the original audio file
#print("Playing original audio:")
#play_audio(file_path)

# Play the filtered audio file
print("Playing filtered audio:")
#play_audio(output_file)

# Compute the spectrogram
Sb = librosa.stft(signal_filtered)

S_dBb = librosa.amplitude_to_db(abs(Sb), ref=np.max)

# Plot the spectrogram
plt.figure(figsize=(12, 4))
librosa.display.specshow(data = S_dBb, sr=sample_rate, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()

transcribed_text_preemphasis = transcribe_audio('filtered_speech_01.wav')

note("Transcription after applying pre-emphasis filter:")
report_transcription_accuracy(ground_truth, transcribed_text_preemphasis)

heading("Whisper ASR - Using OpenAI's Whisper model for automatic speech recognition")

model_size = "tiny"  # You can choose from "tiny", "base", "small", "medium", or "large"
# "tiny" is the fastest and requires the least memory, but has lower accuracy.
# "base" is a good starting point for most tasks, but you can also use "small", "medium", or "large" models for better accuracy at the cost of speed and memory usage.



model = whisper.load_model(model_size) # base model is a good starting point for most tasks, # but you can also use "small", "medium", or "large" models for better accuracy at the cost of speed and memory usage.
result = model.transcribe(file_path)
transcribed_text_whisper = result["text"]

heading2("Whisper Transcription Result", transcribed_text_whisper)
heading2("result['language']",result['language'])

note("Transcription using Whisper ASR:")
report_transcription_accuracy(ground_truth, transcribed_text_whisper)

directory_path = os.path.join(os.getcwd(), "Recordings")

def transcribe_directory_whisper(directory_path):
    transcriptions = []
    for file_name in os.listdir(directory_path):
        heading3("Processing file ", file_name)
        if file_name.endswith(".wav"):
            files_path = os.path.join(directory_path, file_name)
            # Transcribe the audio file
            result = model.transcribe(files_path)
            transcription = result["text"]
            transcriptions.append({"file_name": file_name, "transcription": transcription})
    return transcriptions

heading2("Transcribing all audio files in directory", directory_path)

transcriptions = transcribe_directory_whisper(directory_path)

output_file = "transcriptions.csv"

note(f"Saving transcriptions to {output_file}")

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Track Number", "File Name", "Transcription"])  # Write the header
    for number, transcription in enumerate(transcriptions, start=1):
        print(f"Transcription {number}: {transcription['file_name']} - {transcription['transcription']}")
        writer.writerow([number, transcription['file_name'], transcription['transcription']])

text1 = """You are the most handsome man I have ever seen. Please, tell me more about yourself. This is a test of the text-to-speech functionality."""
text2 = """당신은 제가 본 사람 중 가장 잘생긴 남자예요. 당신에 대해 더 자세히 말씀해 주세요. 텍스트 음성 변환 기능을 테스트하는 것입니다."""
text3 = """Jy is die aantreklikste man wat ek nog ooit gesien het. Vertel my asseblief meer oor jouself. Hierdie is 'n toets van die teks-na-spraak-funksionaliteit."""
text4 = """Je bent de knapste man die ik ooit heb gezien. Vertel me alsjeblieft meer over jezelf. Dit is een test van de tekst-naar-spraakfunctie."""
text5 = """Uyindoda enhle kunazo zonke engake ngazibona. Ngicela ungitshele okwengeziwe ngawe. Lokhu ukuhlola komsebenzi wokuguqula umbhalo ube inkulumo."""
text6 = """Eres el hombre más guapo que he visto. Por favor, cuéntame más sobre ti. Esta es una prueba de la función de texto a voz."""
tts1 = gTTS(text=text1, lang='en', tld="co.za", slow=False)  # Use 'co.za' for South African English accent
tts2 = gTTS(text=text2, lang='ko', slow=False) # Korean
tts3 = gTTS(text=text3, lang='af', slow=False)  # Afrikaans
tts4 = gTTS(text=text4, lang='nl', slow=False)  # Dutch
#tts5 = gTTS(text=text5, lang='zu', slow=False)  # Zulu  --> Not supported by gTTS
tts6 = gTTS(text=text6, lang='es', slow=False)  # Spanish --> Not supported by gTTS
# Save the TTS audio files
print("Saving the TTS audio to 'final_part_X.mp3'")
tts1.save("final_part_English.mp3")  
tts2.save("final_part_Korean.mp3")
tts3.save("final_part_Afrikaans.mp3")
tts4.save("final_part_Dutch.mp3")
#tts5.save("final_part_Zulu.mp3")
tts6.save("final_part_Spanish.mp3")