import numpy as np
import librosa
import whisper
from transformers import pipeline as hf_pipeline
import soundfile as sf
from typing import Any

speech_emotion_order = ["angry", "happy", "neutral", "sad"]
LABEL_MAP = {
    "ang": "angry",
    "hap": "happy",
    "neu": "neutral",
    "sad": "sad"
}

ACOUSTIC_FEATURE_ORDER = [
    "speech_rate",
    "pause_ratio",
    "avg_pause_length",
    "pitch_mean",
    "pitch_std",
    "rms_mean",
    "rms_std",

    "wav2vec2_angry",
    "wav2vec2_happy",
    "wav2vec2_neutral",
    "wav2vec2_sad"
]

ACOUSTIC_DIM = len(ACOUSTIC_FEATURE_ORDER)

_whisper_model = None
_emotion_classifier = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model...")
        _whisper_model = whisper.load_model("base")
    return _whisper_model

def get_emotion_classifier():
    global _emotion_classifier
    if _emotion_classifier is None:
        print("Loading Wav2Vec2 emotion model...")
        _emotion_classifier = hf_pipeline(
            task  = "audio-classification",
            model = "superb/wav2vec2-base-superb-er",
            top_k = None
        )
    return _emotion_classifier


def transcribe_audio(audio_path:str)-> dict[str,Any]:
    try:
        info = sf.info(audio_path)
        duration = info.duration
    except Exception as e:
        raise ValueError(f"Cannot read audio file '{audio_path}': {e}")

    if duration < 0.5:
        raise ValueError(
            f"Audio too short ({duration:.2f}s). Minimum is 0.5 seconds.")

    model  = get_whisper_model()    
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        verbose=False
    )

    result["duration"] = sf.info(audio_path).duration

    text = result.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "Whisper produced no transcript. "
            "Audio may be silent or contain only background noise.")

    return result

def extract_acoustic_measurements(audio_path: str,transcript: dict) -> np.ndarray:
    duration = sf.info(audio_path).duration

    if duration <= 0:
        return np.zeros(7, dtype=np.float32)

    y, sr = librosa.load(audio_path, sr=16000)

    words   = transcript["text"].split()
    n_words = len(words)
    speech_rate = (n_words / duration * 60) if duration > 0 else 0.0

    all_words = []
    for segment in transcript.get("segments", []):
        words_in_segment = segment.get("words", [])
        if not words_in_segment:
            continue
        for word_info in words_in_segment:
            if "start" in word_info and "end" in word_info:
                all_words.append(word_info)

    pauses = []
    if len(all_words) > 1:
        for i in range(1, len(all_words)):
            gap = all_words[i]["start"] - all_words[i-1]["end"]
            if gap > 0.25:   
                pauses.append(gap)

    if not all_words:
        pause_ratio      = 0.0
        avg_pause_length = 0.0            
    else:
        total_pause_time = sum(pauses)
        pause_ratio      = total_pause_time / duration if duration > 0 else 0.0
        avg_pause_length = float(np.mean(pauses)) if pauses else 0.0

    try:
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        # only use frames where magnitude is strong
        positive_mags = magnitudes[magnitudes > 0]
        if len(positive_mags) == 0:
            pitch_mean = 0.0
            pitch_std  = 0.0
        else:
            threshold  = float(np.median(positive_mags))
            pitch_vals = pitches[magnitudes > threshold]
            pitch_vals = pitch_vals[pitch_vals > 0] 

            if len(pitch_vals) == 0:
                pitch_mean = 0.0
                pitch_std  = 0.0
            else:
                pitch_mean = float(np.mean(pitch_vals))
                pitch_std  = float(np.std(pitch_vals))
    except Exception:
        pitch_mean = 0.0
        pitch_std  = 0.0

    try:
        rms      = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms)) if len(rms) > 0 else 0.0
        rms_std  = float(np.std(rms))  if len(rms) > 0 else 0.0
    except Exception:
        rms_mean = 0.0
        rms_std  = 0.0

    result = np.array([
        speech_rate, pause_ratio, avg_pause_length,
        pitch_mean, pitch_std, rms_mean, rms_std
    ], dtype=np.float32)

    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result

def extract_speech_emotions(audio_path: str) -> np.ndarray:
    try:
        classifier = get_emotion_classifier()
        results    = classifier(audio_path)
        score_map = {
            LABEL_MAP[r["label"]]: r["score"]
            for r in results
        }
        emotion_vec = np.array(
            [score_map.get(e, 0.0) for e in speech_emotion_order],
            dtype=np.float32
        )
        return emotion_vec
    except Exception as e:
        print(f"Warning: speech emotion extraction failed: {e}")
        return np.full(len(speech_emotion_order), 
                       1.0 / len(speech_emotion_order), 
                       dtype=np.float32)

def acoustic_readable(transcript: dict,acoustic_vec: np.ndarray,emotion_vec: np.ndarray) -> dict:
    rate = acoustic_vec[0]
    if   rate < 100: rate_label = "very slow"
    elif rate < 130: rate_label = "slow"
    elif rate < 170: rate_label = "normal"
    elif rate < 210: rate_label = "fast"
    else:            rate_label = "very fast"
    
    emotion_scores = dict(zip(speech_emotion_order, emotion_vec.tolist()))
    dominant_vocal_emotion = max(emotion_scores.items(), key=lambda x:x[1])[0]
    top3_vocal = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return {
        "transcript":           transcript.get("text", ""),
        "detected_language":       transcript.get("language", "unknown"),
        "duration_seconds":        round(transcript.get("duration", 0), 2),
        "speech_rate_wpm":         round(float(acoustic_vec[0]), 1),
        "speech_rate_label":       rate_label,
        "pause_ratio":             round(float(acoustic_vec[1]), 4),
        "avg_pause_length_sec":    round(float(acoustic_vec[2]), 4),
        "pitch_mean_hz":           round(float(acoustic_vec[3]), 1),
        "pitch_variability":       round(float(acoustic_vec[4]), 1),
        "loudness_mean":           round(float(acoustic_vec[5]), 4),
        "loudness_variability":    round(float(acoustic_vec[6]), 4),
        "dominant_vocal_emotion":  dominant_vocal_emotion,
        "top3_vocal_emotions":     [(e, round(s, 4)) for e, s in top3_vocal],
        "vocal_emotion_scores":    {e: round(s, 4) for e, s in emotion_scores.items()},
    }

def audio_block(audio_path: str) -> tuple:
    transcript= transcribe_audio(audio_path)
    acoustic_vec= extract_acoustic_measurements(audio_path, transcript)  # (7,)
    speech_emotion_vec= extract_speech_emotions(audio_path)

    audio_feature_vec= np.concatenate([acoustic_vec, speech_emotion_vec])          # (11,)
    audio_feature_mask= np.ones(ACOUSTIC_DIM, dtype=np.float32)

    readable =acoustic_readable(transcript,acoustic_vec, speech_emotion_vec)

    return audio_feature_vec, audio_feature_mask, transcript, readable

def empty_audio_block() -> tuple:
    audio_feature_vec= np.zeros(ACOUSTIC_DIM, dtype=np.float32)
    audio_feature_mask= np.zeros(ACOUSTIC_DIM, dtype=np.float32)
    readable= None
    transcript= None
    return audio_feature_vec, audio_feature_mask, transcript, readable    
