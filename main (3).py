import numpy as np
import torch
import nltk
nltk.download('punkt_tab')
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import re
from datetime import datetime
import textstat

emotion_classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vader_analyzer   = SentimentIntensityAnalyzer()

audio_path="/teamspace/studios/this_studio/audio_clip.ogg"


emotion_order= [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral'
]

def get_vader(text) ->np.ndarray:
    sentences=nltk.sent_tokenize(text)

    if not sentences:
        return np.zeros(7)
    all_scores = [vader_analyzer.polarity_scores(s) for s in sentences]
    neg  = np.mean([s['neg']      for s in all_scores])
    neu  = np.mean([s['neu']      for s in all_scores])
    pos  = np.mean([s['pos']      for s in all_scores])
    comp = np.mean([s['compound'] for s in all_scores])
    min_comp=np.min([s['compound'] for s in all_scores])
    max_comp=np.max([s['compound'] for s in all_scores])
    std_comp= np.std([s['compound'] for s in all_scores])

    return np.array([neg, neu, pos, comp,min_comp, max_comp, std_comp],dtype=np.float32)


def readability_features(text)-> np.ndarray:
    fre=textstat.textstat.flesch_reading_ease(text)     
    fkgl=textstat.textstat.flesch_kincaid_grade(text)
    ari=textstat.textstat.automated_readability_index(text) 

    return np.array([fre,fkgl,ari], dtype=np.float32)


def first_person(text)->np.ndarray:
    words    = re.findall(r'\b\w+\b', text.lower())
    total    = len(words) or 1
    singular = ['i','me','my','mine','myself']
    plural   = ['we','us','our','ours','ourselves']
    first_person_singular = sum(w in singular for w in words) / total
    first_person_plural = sum(w in plural   for w in words) / total
    
    return np.array([first_person_singular,first_person_plural],dtype=np.float32)

def length_features(text) ->np.ndarray:
    sentences  = nltk.sent_tokenize(text)
    words      = nltk.word_tokenize(text.lower())
    words=[w for w in words if w.isalpha()]
    n_sents    = len(sentences) or 1
    n_words    = len(words)
    avg_sent_l = n_words / n_sents
    return np.array([avg_sent_l, n_sents, n_words],dtype=np.float32)    

def punctuation_features(text) ->np.ndarray:
    sentences  = nltk.sent_tokenize(text)
    n_sents    = len(sentences) or 1
    words    = text.split()
    n_words  = len(words) or 1
    q_ratio  = text.count('?')  / n_sents
    ex_ratio = text.count('!')  / n_sents
    ellipsis = text.count('...') / n_sents
    caps_r   = sum(1 for w in words if w.isupper() and len(w)>1) / n_words
    return np.array([q_ratio, ex_ratio, ellipsis, caps_r],dtype=np.float32)

def lexical_diversity(text) ->np.ndarray:
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words)==0:
        return np.array([0.0,0.0], dtype=np.float32)

    ttr=len(set(words))/len(words) 
    segments,running=[],[]
    for w in words:
        running.append(w)
        if len(set(running)) / len(running) < 0.72:
            segments.append(len(running))
            running = []
    mtld = float(np.mean(segments)) if segments else float(len(words))

    return np.array([ttr, mtld], dtype=np.float32)   



def metadata_features(timestamp: datetime, prev_timestamp: datetime | None = None) -> np.ndarray:
    hour = timestamp.hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    if prev_timestamp is None:
        days_gap = 0.0
    else:
        gap = (timestamp - prev_timestamp).days
        gap = min(max(gap, 0), 30)
        days_gap = gap / 30.0

    return np.array([hour_sin, hour_cos, days_gap],dtype=np.float32)

def health_features(sleep_hours=None, sleep_quality=None, 
                    activity_level=None, music_mood_score=None) -> np.ndarray:
    values = np.array([
        sleep_hours        if sleep_hours        is not None else 0.0,
        sleep_quality      if sleep_quality      is not None else 0.0,
        activity_level     if activity_level     is not None else 0.0,
        music_mood_score   if music_mood_score   is not None else 0.0,
    ], dtype=np.float32)
    
    masks = np.array([
        1.0 if sleep_hours        is not None else 0.0,
        1.0 if sleep_quality      is not None else 0.0,
        1.0 if activity_level     is not None else 0.0,
        1.0 if music_mood_score   is not None else 0.0,
    ], dtype=np.float32)
    
    return np.concatenate([values, masks])  
    

def readables(text,emotion_vec,vader_vec,lexical_vec,readability_vec,first_person_vec,length_vec,health_vec):
    emotion_scores = {
        e: float(emotion_vec[i])
        for i, e in enumerate(emotion_order)
    }

    top3 = sorted(
        emotion_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    compound = float(vader_vec[3])

    if compound <= -0.5:
        sentiment_label = "very negative"
    elif compound <= -0.05:
        sentiment_label = "negative"
    elif compound < 0.05:
        sentiment_label = "neutral"
    elif compound < 0.5:
        sentiment_label = "positive"
    else:
        sentiment_label = "very positive"

    avg_sent = float(length_vec[0])
    n_words = int(length_vec[2])

    if n_words < 20:
        style = "very short entry"
    elif avg_sent < 7:
        style = "fragmented"
    elif avg_sent < 15:
        style = "normal"
    else:
        style = "elaborate"

    _va = SentimentIntensityAnalyzer()

    raw_words = re.findall(r'\b\w+\b', text.lower())

    scored = [
        (w, _va.lexicon.get(w, 0))
        for w in raw_words
    ]

    negative_keywords = [
        w
        for w, s in sorted(scored, key=lambda x: x[1])
        if s < -1.5
    ][:3]

    internal_distress = float(np.mean([
        emotion_scores.get("sadness", 0.0),
        emotion_scores.get("fear", 0.0),
        emotion_scores.get("nervousness", 0.0),
        emotion_scores.get("grief", 0.0),
        max(0.0, -compound)
    ]))

    external_aggression = float(np.mean([
        emotion_scores.get("anger", 0.0),
        emotion_scores.get("annoyance", 0.0),
        emotion_scores.get("disgust", 0.0),
        emotion_scores.get("disapproval", 0.0)
    ]))

    return {

        "raw_display_metrics": {
            "sentiment_score": compound,
            "dominant_emotion": top3[0][0],
            "top3_emotions": [
                (e, round(s, 4))
                for e, s in top3
            ],

            "word_count": n_words,

            "avg_sentence_length":
                round(avg_sent, 2),

            "reading_grade":
                round(float(readability_vec[1]), 2),    

            "i_pronoun_ratio": round(float(first_person_vec[0]), 4)   
        },

        "contextual_metadata": {
            "sentiment_label": sentiment_label,
            "writing_style": style,
            "dominant_emotion": top3[0][0],
            "negative_keywords":
                negative_keywords or ["none detected"],
        },

        "data_integrity_masks": {
            "text_present": True,

            "sleep_provided":
                bool(health_vec[4]),

            "sleep_quality_provided":
                bool(health_vec[5]),

            "activity_provided":
                bool(health_vec[6]),

            "music_provided":
                bool(health_vec[7]),

            "entry_too_short":
                n_words < 20,
        },

        "clinical_scalars": {
            "internal_distress_risk":
                round(internal_distress, 4),

            "external_aggression_score":
                round(external_aggression, 4),
        }
    }


def extract_features(text, timestamp=None, prev_timestamp=None,audio_path= None,sleep_hours=None, sleep_quality=None,
                     activity_level=None, music_mood_score=None):
    if timestamp is None:
        timestamp=datetime.now()
    sbert_embedding=model.encode(text,convert_to_numpy=True).astype(np.float32)
    emotions=emotion_classifier(text)[0] 
    score_map={r['label']: r['score'] for r in emotions}
    emotion_vec=np.array([score_map.get(e,0.0) for e in emotion_order],dtype=np.float32)  
    vader_vec=get_vader(text)
    lexical_diversity_vec=lexical_diversity(text)
    readability_feature_vec=readability_features(text)
    first_person_vec=first_person(text)
    length_features_vec=length_features(text)
    punctuation_features_vec=punctuation_features(text)
    metadata_vec = metadata_features(timestamp,prev_timestamp)
    health_features_vec= health_features(sleep_hours, sleep_quality,
                                          activity_level, music_mood_score)

    if audio_path is not None:
        from audio import audio_block
        audio_feature_vec, audio_feature_mask, transcript, acoustic_readable = \
           audio_block(audio_path)
    else:
        from audio import empty_audio_block
        audio_feature_vec, audio_feature_mask, transcript, acoustic_readable = \
            empty_audio_block()        


    feature_vec = np.concatenate([sbert_embedding, emotion_vec,vader_vec,lexical_diversity_vec,readability_feature_vec,
    first_person_vec,length_features_vec,punctuation_features_vec, metadata_vec, health_features_vec,audio_feature_vec,audio_feature_mask])  

    readables_vec = readables(
    text,
    emotion_vec,
    vader_vec,
    lexical_diversity_vec,
    readability_feature_vec,
    first_person_vec,
    length_features_vec,
    health_features_vec)

    if acoustic_readable is not None:
        readables_vec["acoustic_profile"] = acoustic_readable  

    return feature_vec, readables_vec #466


if __name__ == "__main__":

    feature_vec, readable = extract_features(
        text="Today was a good day.",
        audio_path="/teamspace/studios/this_studio/audio_clip.ogg"
    )

    print("Vector shape:", feature_vec.shape)

    print("\nTranscript:")
    print(readable["acoustic_profile"]["transcript"])

    print("\nAudio features:")
    print(readable["acoustic_profile"])

    print("\nTop emotions:")
    print(readable["raw_display_metrics"]["top3_emotions"])