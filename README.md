**Social Media Analytics** — README + Notebook
README (concise project plan)
1 — Case study (summary)

Problem: Content teams and creators need quick, data-driven estimates of how a planned post will perform (reach, likes, engagement) and automated hashtag & media descriptions to speed publishing and improve results.

Goal: Build an end-to-end prototype: Frontend dashboard → AI assistant (hashtags + descriptions) → ML engine (predict reach, likes, engagement) → return suggestions & predictions to user.

Users / Value: Social media managers, content creators, community managers. Value: faster publishing, better-optimized posts, measurable expectations.

2 — Features

Frontend: create post (text + optional image/video), schedule time, set follower count.

AI assistant: generate hashtags and short image/video description.

Prediction engine: predict reach, likes, engagement rate.

Explainability: show top features that influenced prediction (basic).

API: FastAPI endpoints for suggestion + prediction.

Notebook: training, inference, and saving models.

3 — Data schema (inputs & outputs)

Request payload (from UI):

{
  "post_text": "string",
  "media_type": "none|image|video",
  "num_media_items": 0,
  "scheduled_time": "ISO8601 string (UTC)",
  "user_followers": 1000,
  "user_avg_engagement_rate": 0.01,
  "num_hashtags": 0,
  "num_mentions": 0,
  "num_emojis": 0
}


Response (inference):

{
  "suggested_hashtags": ["#tag1","#tag2"],
  "suggested_description": "short description for media",
  "predicted_reach": 12345,
  "predicted_likes": 678,
  "predicted_engagement_rate": 0.054,
  "explanations": {
    "top_features": ["user_followers","media_type","hour"]
  }
}

4 — Derived features (ML)

post_text_len (chars / words)

daypart / hour bucket

day_of_week

num_hashtags, num_mentions, num_emojis

has_link (bool)

media_type (categorical)

user_followers (numeric)

user_avg_engagement_rate (numeric)

Targets: reach (impressions), likes, engagement_rate (likes+comments+shares / impressions).

5 — Tech stack

Python 3.10+

Data + ML: pandas, numpy, scikit-learn, joblib

API: fastapi, uvicorn

Optional: LLMs for better hashtag/description (OpenAI, Anthropic) — fallback: TF-IDF / heuristics

Storage: Postgres / MongoDB (post records), S3 for media

Containerization: Docker

6 — Project structure
social-analytics/
├── api/
│   └── main.py                # FastAPI app serving /predict and /suggest
├── notebooks/
│   └── train_predictor.ipynb  # the notebook cells below
├── src/
│   ├── features.py
│   ├── nlp_utils.py
│   ├── train.py
│   └── serve.py
├── models/
│   ├── gbm_reach.joblib
│   └── gbm_likes.joblib
├── data/
│   └── synthetic_data.csv
├── requirements.txt
└── README.md

7 — Production suggestions (next steps)

Replace heuristic hashtag generator with LLM or trained extractor.

Add calibration/uncertainty (quantile regression or prediction intervals).

Persist predictions + ground-truth after post completes; retrain periodically.

Add monitoring (data drift, distribution changes).

Rate-limit LLM calls; provide offline fallback.

Add authentication, logging, and CI/CD + Docker.

Jupyter Notebook — Cells (copy each cell to a notebook)

Notes before running:

Create directory models/ next to your notebook so joblib saving works.

The notebook produces synthetic data and trains two scikit-learn regressors (reach & likes).

All imports are standard pip packages (pandas, numpy, scikit-learn, joblib, fastapi). See requirements.txt.

Cell 1 — Title & imports
# Cell 1: Title & imports
# Social Media Analytics — training notebook
import os
import math
from datetime import datetime
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction import _stop_words
import joblib

# reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ensure models directory
os.makedirs('models', exist_ok=True)

print("Imports ready.")


Explanation: Standard imports for data, modeling, and saving models. Create models/ folder so saved models go there.

Cell 2 — Synthetic data generator
# Cell 2: generate synthetic dataset
def generate_synthetic_data(n=6000, seed=RANDOM_SEED):
    np.random.seed(seed)
    d = {}
    d['user_followers'] = np.random.randint(50, 500000, size=n)
    d['user_avg_engagement_rate'] = np.random.beta(2, 50, size=n)  # small baseline rates
    
    d['media_type'] = np.random.choice(['none', 'image', 'video'], size=n, p=[0.3, 0.5, 0.2])
    d['num_media_items'] = np.random.poisson(1, size=n)
    d['post_text_len'] = np.random.randint(5, 800, size=n)
    d['num_hashtags'] = np.random.poisson(2, size=n)
    d['num_mentions'] = np.random.poisson(0.3, size=n)
    d['num_emojis'] = np.random.poisson(1, size=n)
    d['has_link'] = np.random.binomial(1, 0.15, size=n)
    d['hour'] = np.random.randint(0, 24, size=n)
    d['day_of_week'] = np.random.randint(0, 7, size=n)

    df = pd.DataFrame(d)

    # Synthetic target logic (hidden to models)
    media_boost = df['media_type'].map({'none': 0.6, 'image': 1.0, 'video': 1.6})
    time_boost = np.where((df['hour'] >= 18) & (df['hour'] <= 22), 1.2, 1.0)

    # base reach proportional to followers (and a tiny dependence on avg eng rate)
    base_reach = df['user_followers'] * (0.02 + 0.2 * df['user_avg_engagement_rate'])
    df['reach'] = (base_reach * media_boost * time_boost * (1 + 0.05 * df['num_hashtags'])).clip(min=5)

    # Add multiplicative noise
    df['reach'] = (df['reach'] * (1 + 0.5 * np.random.randn(n))).clip(min=5).round().astype(int)

    # engagement_rate depends on user's baseline and some randomness
    df['engagement_rate'] = (df['user_avg_engagement_rate'] * (0.8 + 0.6 * np.random.rand(n))).round(4)
    df['likes'] = (df['reach'] * df['engagement_rate'] * (0.2 + 0.8 * np.random.rand(n))).clip(min=0).round().astype(int)

    return df

df = generate_synthetic_data(6000)
df.head()


Explanation: Creates a dataset for training. Targets are reach, likes, engagement_rate.

Cell 3 — Quick EDA (inspect distributions)
# Cell 3: EDA - quick look
print("Rows:", len(df))
print(df.describe().T[['min','50%','max']])
print("\nMedia distribution:")
print(df['media_type'].value_counts())


Explanation: Check basic distributions to understand scale and skew.

Cell 4 — Feature engineering
# Cell 4: Feature engineering
# We'll create daypart to capture time-of-day behavior
df_feat = df.copy()
df_feat['post_text_len'] = df_feat['post_text_len']  # already present
df_feat['daypart'] = pd.cut(df_feat['hour'], bins=[-1,5,11,17,22,24], labels=['late_night','morning','afternoon','evening','late_night2'])
# convert categorical types to string for OneHotEncoder
df_feat['media_type'] = df_feat['media_type'].astype(str)
df_feat['daypart'] = df_feat['daypart'].astype(str)

FEATURE_COLS = [
    'user_followers','user_avg_engagement_rate','post_text_len','num_hashtags',
    'num_mentions','num_emojis','has_link','media_type','daypart','day_of_week'
]

X = df_feat[FEATURE_COLS]
y_reach = df_feat['reach']
y_likes = df_feat['likes']
y_eng = df_feat['engagement_rate']

X.head()


Explanation: Prepare X and target vectors; create daypart to capture hour effects.

Cell 5 — Build preprocessing pipeline & train reach model
# Cell 5: preprocessing + model for reach
numeric_features = ['user_followers','user_avg_engagement_rate','post_text_len','num_hashtags','num_mentions','num_emojis','has_link']
cat_features = ['media_type','daypart','day_of_week']

preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

model_reach = make_pipeline(preprocessor, GradientBoostingRegressor(random_state=RANDOM_SEED, n_estimators=150, max_depth=4))

X_train, X_test, y_train, y_test = train_test_split(X, y_reach, test_size=0.2, random_state=RANDOM_SEED)
model_reach.fit(X_train, y_train)

pred = model_reach.predict(X_test)
print('Reach RMSE:', math.sqrt(mean_squared_error(y_test, pred)))
print('Reach R2:', r2_score(y_test, pred))

# persist model
joblib.dump(model_reach, 'models/gbm_reach.joblib')
print("Saved models/gbm_reach.joblib")


Explanation: Train a Gradient Boosting model as a strong baseline. Save with joblib.

Cell 6 — Train likes model
# Cell 6: likes model
model_likes = make_pipeline(preprocessor, GradientBoostingRegressor(random_state=RANDOM_SEED, n_estimators=150, max_depth=4))

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_likes, test_size=0.2, random_state=RANDOM_SEED)
model_likes.fit(X_train2, y_train2)

pred_l = model_likes.predict(X_test2)
print('Likes RMSE:', math.sqrt(mean_squared_error(y_test2, pred_l)))
print('Likes R2:', r2_score(y_test2, pred_l))

joblib.dump(model_likes, 'models/gbm_likes.joblib')
print("Saved models/gbm_likes.joblib")


Explanation: Separate model for likes. Alternatively, one could derive likes from predicted reach and predicted engagement — both approaches are valid.

Cell 7 — Simple hashtag & description suggester (heuristic)
# Cell 7: simple hashtag & description generator (heuristic)
STOPWORDS = set(_stop_words.ENGLISH_STOP_WORDS)

def suggest_hashtags_from_text(text, top_k=5):
    """Simple heuristic: pick most frequent non-stopword tokens (length>2)"""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []
    tokens = re.findall(r"\w+", text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    if not tokens:
        return []
    freq = pd.Series(tokens).value_counts()
    top = list(freq.head(top_k).index)
    return ['#' + t for t in top]

def suggest_description_for_media(text, max_words=25):
    """Simple description: first N words + a hint phrase."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ''
    words = text.strip().split()
    preview = ' '.join(words[:max_words])
    suffix = '...' if len(words) > max_words else ''
    return (preview + suffix).strip()

# quick demo
example_text = "Launching our new summer collection — lightweight jackets, breathable fabric, and bold colors! Shop now: https://example.com"
print("Hashtags:", suggest_hashtags_from_text(example_text, top_k=6))
print("Description:", suggest_description_for_media(example_text, max_words=10))


Explanation: Heuristic fallback that works offline. In production, replace with an LLM prompt or a trained keyphrase extraction model.

Cell 8 — Inference wrapper (payload -> suggestions & predictions)
# Cell 8: inference wrapper
# Load models (ensure they are present)
model_reach = joblib.load('models/gbm_reach.joblib')
model_likes = joblib.load('models/gbm_likes.joblib')

def prepare_features_from_payload(payload):
    # payload expects: post_text, media_type, num_media_items, scheduled_time (ISO), user_followers, user_avg_engagement_rate, num_hashtags, num_mentions, num_emojis
    post_text = payload.get('post_text', '') or ''
    media_type = payload.get('media_type', 'none')
    num_media_items = int(payload.get('num_media_items', 0))
    scheduled_time = payload.get('scheduled_time', None)
    if scheduled_time:
        try:
            dt = pd.to_datetime(scheduled_time)
        except:
            dt = pd.to_datetime(datetime.utcnow())
    else:
        dt = pd.to_datetime(datetime.utcnow())
    hour = int(dt.hour)
    day_of_week = int(dt.dayofweek)
    user_followers = int(payload.get('user_followers', 100))
    user_avg_engagement_rate = float(payload.get('user_avg_engagement_rate', 0.01))

    features = {
        'user_followers': user_followers,
        'user_avg_engagement_rate': user_avg_engagement_rate,
        'post_text_len': len(post_text),
        'num_hashtags': int(payload.get('num_hashtags', 0)),
        'num_mentions': int(payload.get('num_mentions', 0)),
        'num_emojis': int(payload.get('num_emojis', 0)),
        'has_link': 1 if ('http' in post_text or 'www.' in post_text) else 0,
        'media_type': media_type,
        'daypart': pd.cut([hour], bins=[-1,5,11,17,22,24], labels=['late_night','morning','afternoon','evening','late_night2'])[0],
        'day_of_week': day_of_week
    }
    return pd.DataFrame([features])

def inference(payload):
    # 1) suggestions
    suggested_hashtags = suggest_hashtags_from_text(payload.get('post_text',''), top_k=6)
    suggested_description = None
    if payload.get('media_type') in ('image','video'):
        suggested_description = suggest_description_for_media(payload.get('post_text',''), max_words=25)
    # ensure the features know the number of hashtags we would include
    feat_payload = payload.copy()
    feat_payload['num_hashtags'] = feat_payload.get('num_hashtags', len(suggested_hashtags))
    feat_df = prepare_features_from_payload(feat_payload)

    # 2) predictions
    pred_reach = int(max(0, round(float(model_reach.predict(feat_df)[0]))))
    pred_likes = int(max(0, round(float(model_likes.predict(feat_df)[0]))))
    pred_eng_rate = round(pred_likes / pred_reach if pred_reach > 0 else 0.0, 4)

    # minimal explanation (feature importance approximation)
    # For a real explanation, use SHAP. Here we simply return an ordered list of candidate features.
    explanations = {'top_features': ['user_followers','media_type','daypart','num_hashtags']}

    return {
        'suggested_hashtags': suggested_hashtags,
        'suggested_description': suggested_description,
        'predicted_reach': pred_reach,
        'predicted_likes': pred_likes,
        'predicted_engagement_rate': pred_eng_rate,
        'explanations': explanations
    }

# Example payload and call
payload_example = {
    "post_text": "Announcing our biggest sale of the year! 50% off on selected items. Free shipping for orders over $50. Grab yours now!",
    "media_type": "image",
    "num_media_items": 1,
    "scheduled_time": "2025-07-15T19:30:00Z",
    "user_followers": 12500,
    "user_avg_engagement_rate": 0.012
}
print("Inference result:", inference(payload_example))


Explanation: inference() ties together suggestions and predictions and returns the JSON-like dict that the frontend can consume.

Cell 9 — Example FastAPI endpoint (save as api/main.py for production)
# Cell 9: FastAPI example - save this as api/main.py to run with uvicorn
# NOTE: This cell is for illustrative purposes - running FastAPI should be done outside notebook.

fastapi_example = '''
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

# load helper functions and models (adapt import path when moving to package)
# from your_package import inference

app = FastAPI()

class PostPayload(BaseModel):
    post_text: str
    media_type: str = 'none'
    num_media_items: int = 0
    scheduled_time: str = None
    user_followers: int = 100
    user_avg_engagement_rate: float = 0.01
    num_hashtags: int = 0
    num_mentions: int = 0
    num_emojis: int = 0

@app.post('/predict')
def predict(payload: PostPayload):
    # call inference function from your module
    # result = inference(payload.dict())
    # temporary stub response
    return {"detail": "Replace with your inference(payload.dict()) call"}

# To run: uvicorn api.main:app --reload --port 8000
'''
print(fastapi_example)


Explanation: Shows how to wire the inference function into FastAPI. In production, put inference into a module and import it here. Run with uvicorn api.main:app --reload.

Cell 10 — Save sample synthetic dataset (optional)
# Cell 10: save synthetic dataset for inspection
df.to_csv('data/synthetic_social_posts.csv', index=False)
print("Saved synthetic dataset to data/synthetic_social_posts.csv")

API contract (detailed)

Endpoint: POST /predict
Content-Type: application/json

Request JSON schema:

{
  "post_text": "string",
  "media_type": "none|image|video",
  "num_media_items": 0,
  "scheduled_time": "2025-07-15T19:30:00Z",
  "user_followers": 10000,
  "user_avg_engagement_rate": 0.01,
  "num_hashtags": 0,
  "num_mentions": 0,
  "num_emojis": 0
}


Response JSON schema:

{
  "suggested_hashtags": ["#tag1","#tag2"],
  "suggested_description": "short description if media provided",
  "predicted_reach": 12345,
  "predicted_likes": 678,
  "predicted_engagement_rate": 0.054,
  "explanations": {
    "top_features": ["user_followers","media_type","daypart"]
  }
}


Notes:

predicted_reach and predicted_likes are integers.

predicted_engagement_rate is float (ratio).

Provide scheduled_time as ISO8601 in UTC to ensure consistent time-based features.

Example requests and expected outputs

Example Request

{
  "post_text": "New product launch: ultralight running shoes — performance meets comfort. Limited stock!",
  "media_type": "image",
  "num_media_items": 1,
  "scheduled_time": "2025-07-15T19:30:00Z",
  "user_followers": 20000,
  "user_avg_engagement_rate": 0.015
}


Example Response (approx)

{
  "suggested_hashtags": ["#product","#launch","#running","#shoes","#performance"],
  "suggested_description": "New product launch: ultralight running shoes — performance meets comfort. Limited stock!...",
  "predicted_reach": 13456,
  "predicted_likes": 876,
  "predicted_engagement_rate": 0.065,
  "explanations": {
    "top_features": ["user_followers","media_type","daypart","num_hashtags"]
  }
}


These numbers are synthetic — with your real historical data the model should be retrained.

Beginner-friendly explanation (short)

Data: Start with historical posts + features. If not available, use synthetic data to build and validate pipeline.

Feature engineering: Create simple, interpretable features (length, hour bucket, hashtags count).

Modeling: Start with scikit-learn (Gradient Boosting). Save the pipeline with preprocessing using joblib.

Inference: Expose a lightweight API that receives the post, extracts features, and returns suggestions + predictions.

Improve: Replace heuristics with LLM or trained NLP models for hashtags/description; add uncertainty estimates using quantile models or bootstrap.

Production checklist (actionable)

Add auth & rate limits to API.

Add logging + metrics (latency, errors, prediction distributions).

Save input + prediction + ground-truth after post completes for re-training.

Use SHAP for explainability, and add feature store for user historical metrics.

Containerize (Dockerfile) and add CI tests for model inference.
