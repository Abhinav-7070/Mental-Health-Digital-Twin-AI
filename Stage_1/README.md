# Encoder Module Overview

The encoder converts raw user inputs into a fixed-size feature vector that can be used by the baseline model, anomaly detector and temporal tracker.

---

## Inputs

* Journal text (**required**)
* Audio recording (**optional**)
* Sleep hours *(optional)*
* Sleep quality *(optional)*
* Activity level *(optional)*
* Music mood score *(optional)*

---

# Text Pipeline

| Feature                 | Dimensions |
| ----------------------- | ---------: |
| SBERT Embedding         |        384 |
| GoEmotions Model        |         28 |
| VADER Sentiment         |          7 |
| Lexical Diversity       |          2 |
| Readability             |          3 |
| First Person Pronouns   |          2 |
| Length Features         |          3 |
| Punctuation Features    |          4 |
| Time Metadata           |          3 |
| Health Features + Masks |          8 |

**Total Text Features = 444**

---

# Audio Pipeline (Optional)

### Whisper

* Speech transcription
* Language detection
* Word timestamps

### Acoustic Features (7)

* Speech Rate
* Pause Ratio
* Average Pause Length
* Mean Pitch
* Pitch Variability
* Mean Loudness
* Loudness Variability

### Speech Emotion Recognition (4)

Model: **superb/wav2vec2-base-superb-er**

* Angry
* Happy
* Neutral
* Sad

An **11-dimensional audio mask** is also appended to indicate whether audio features are present.

---

# Final Feature Vector

| Component          | Dimensions |
| ------------------ | ---------: |
| Text Features      |        444 |
| Audio Features     |         11 |
| Audio Feature Mask |         11 |
| **Total**          |    **466** |

If no audio is provided, the audio feature vector is filled with zeros and the corresponding mask is set to 0, ensuring that the output vector always remains **466 dimensions**.
