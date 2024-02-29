import os
from scipy.ndimage import gaussian_filter1d
import librosa
import soundfile as sf
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import whisperx
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast





#dependencies
def extract_features(audio_file, sample_rate=22050, mfcc=True, chroma=True, mel=True):
    audio, sample_rate = librosa.load(audio_file, sr=sample_rate)
    features = []
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        features.append(mfccs)
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        features.append(chroma)
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        features.append(mel)
    return np.concatenate(features)


def predict_audio(audio_file):
    features = extract_features(audio_file)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    if prediction == 0:
        return "not fear - 0"
    elif prediction > 0 and prediction <= 0.1:
        return "fear - 1"
    elif prediction > 0.1 and prediction <= 0.2:
        return "fear - 2"
    elif prediction > 0.2 and prediction <= 0.3:
        return "fear - 3"
    elif prediction > 0.3 and prediction <= 0.4:
        return "fear - 4"
    elif prediction > 0.4 and prediction <= 0.5:
        return "fear - 5"
    elif prediction > 0.5 and prediction < 0.6:
        return "fear - 6"
    elif prediction > 0.6 and prediction < 0.7:
        return "fear - 7"
    elif prediction > 0.7 and prediction < 0.8:
        return "fear - 8"
    elif prediction > 0.8 and prediction < 0.9:
        return "fear - 9"
    elif prediction > 0.9 and prediction < 1:
        return "fear - 10"

#components
def preprosessor(noisy_audio,sr):
    denoised_audio = gaussian_filter1d(noisy_audio, sigma=2)
    sf.write('denoised_audio_hardlevel.mp3', denoised_audio, sr)

def sentiment_score(audio_file_path):
    prediction = predict_audio(audio_file_path)
    return prediction

def annotate(audio_file,senti):
    senti_score=senti
    device = "cpu"
    audio_file = "audio.mp3"
    batch_size = 2
    compute_type = "int8"
    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(model_name="Harveenchadha/vakyansh-wav2vec2-tamil-tam-250",language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    combined_dialogue = ''
    current_speaker = None
    current_dialogue = ''
    for segment in result['segments']:
        speaker = segment['speaker']
        text = segment['text']
        words = segment['words']
        dialogue = ' '.join([word['word'] for word in words])

        if current_speaker == speaker:
            current_dialogue += ' ' + dialogue
        else:
            if current_speaker is not None:
                combined_dialogue += f"{current_speaker}: {current_dialogue}\n"
            current_speaker = speaker
            current_dialogue = dialogue
    if current_speaker is not None:
        combined_dialogue += f"{current_speaker}: {current_dialogue}"

    translator(combined_dialogue,senti_score)

def translator(text, senti):
    source_lang="ta_IN"
    target_lang="en_XX"
    model_name = "facebook/mbart-large-50-many-to-one-mmt"
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[source_lang], forced_eos_token_id=tokenizer.lang_code_to_id[target_lang])
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    text_sentiment(translated_text,senti)
  
def text_sentiment(text,audio_senti):
    # Load sentiment analysis pipeline
    sentiment_analysis = pipeline('sentiment-analysis')
    # Example usage
    result = sentiment_analysis(text)
    # Print sentiment and confidence score
    sentiment = result[0]['label']
    confidence_score = result[0]['score']
    summarizer(text,sentiment, confidence_score,audio_senti)


def summarizer(text,sentiment,confidence_score,audio_senti):
    classifier = pipeline("summarization")
    summarized_text=classifier(f"{text}",max_length=56)
    result = f"TEXT-SENTIMENT: {sentiment}\nCONFIDENCE-SCORE: {confidence_score}\nAUDIO-SENTIMENT-SCORE: {audio_senti}\nSUMMARIZED-TEXT: {summarized_text}"
    print(result)


#preprocessing
noisy_audio, sr = librosa.load(r'audio.mp3', sr=None)
preprosessor(noisy_audio,sr)
#sentiment analysis
model = load_model("fear_detection_model3.h5")
audio_path="denoised_audio_hardlevel.mp3"
senti_score=sentiment_score(audio_path)
#print(senti_score)
#speaker segmentation
annotate(audio_path,senti_score)
