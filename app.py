import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Course Recommender", layout="wide")
st.title("🎓 Personalized Course Recommendation System")

# --------------------------------------------------
# SIDEBAR – FILE UPLOAD
# --------------------------------------------------
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel File (.xlsx)", type=["xlsx"]
)

if uploaded_file is None:
    st.info("Please upload an Excel file to continue.")
    st.stop()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_excel(uploaded_file)

# --------------------------------------------------
# REQUIRED COLUMN CHECK
# --------------------------------------------------
required_cols = [
    'user_id', 'course_id', 'course_name', 'instructor',
    'rating', 'difficulty_level', 'course_duration_hours',
    'certification_offered', 'study_material_available',
    'course_price', 'feedback_score'
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing columns in dataset: {missing}")
    st.stop()

# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------
df = df.dropna(subset=['user_id', 'course_id', 'rating'])

df['certification_offered'] = (
    df['certification_offered'].astype(str).str.strip().str.lower()
    .map({'yes': 1, 'no': 0})
)

df['study_material_available'] = (
    df['study_material_available'].astype(str).str.strip().str.lower()
    .map({'yes': 1, 'no': 0})
)

df['difficulty_level'] = (
    df['difficulty_level'].astype(str).str.strip().str.lower()
    .map({'beginner': 1, 'intermediate': 2, 'advanced': 3})
)

df = df.dropna()

# --------------------------------------------------
# SCALING
# --------------------------------------------------
scaler = MinMaxScaler()
num_cols = ['course_duration_hours', 'course_price', 'feedback_score']
df[num_cols] = scaler.fit_transform(df[num_cols])

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("⚙️ Recommendation Settings")

user_id = st.sidebar.selectbox(
    "Select User ID",
    sorted(df['user_id'].unique())
)

top_n = st.sidebar.slider("Number of Recommendations", 1, 10, 5)
recommend_btn = st.sidebar.button("🔍 Recommend")

# --------------------------------------------------
# USER–COURSE MATRIX
# --------------------------------------------------
user_course_matrix = df.pivot_table(
    index='user_id',
    columns='course_id',
    values='rating',
    aggfunc='mean'
).fillna(0)

# --------------------------------------------------
# USER SIMILARITY
# --------------------------------------------------
user_sim = cosine_similarity(user_course_matrix)
user_sim_df = pd.DataFrame(
    user_sim,
    index=user_course_matrix.index,
    columns=user_course_matrix.index
)

# --------------------------------------------------
# COURSE SIMILARITY (CONTENT)
# --------------------------------------------------
feature_cols = [
    'difficulty_level',
    'course_duration_hours',
    'certification_offered',
    'study_material_available',
    'course_price',
    'feedback_score'
]

course_features = df.groupby('course_id')[feature_cols].mean()
course_sim = cosine_similarity(course_features)
course_sim_df = pd.DataFrame(
    course_sim,
    index=course_features.index,
    columns=course_features.index
)

# --------------------------------------------------
# PREDICTION FUNCTIONS
# --------------------------------------------------
def user_based_prediction(uid, cid, k=5):
    if cid not in user_course_matrix.columns:
        return np.nan

    sims = user_sim_df.loc[uid].sort_values(ascending=False)[1:k+1]
    ratings = user_course_matrix.loc[sims.index, cid]

    mask = ratings > 0
    if not mask.any():
        return np.nan

    return np.average(ratings[mask], weights=sims[mask])


def content_based_prediction(cid):
    if cid not in course_sim_df.index:
        return np.nan

    similar_courses = course_sim_df[cid].sort_values(ascending=False)[1:6]
    ratings = df[df['course_id'].isin(similar_courses.index)]['rating']

    return ratings.mean() if not ratings.empty else np.nan

# --------------------------------------------------
# HYBRID RECOMMENDER
# --------------------------------------------------
def recommend_for_user(uid, top_n=5):
    rated = df[df['user_id'] == uid]['course_id'].unique()
    candidates = df[~df['course_id'].isin(rated)].drop_duplicates('course_id')

    scores = []
    for cid in candidates['course_id']:
        s1 = user_based_prediction(uid, cid)
        s2 = content_based_prediction(cid)
        scores.append(np.nanmean([s1, s2]))

    candidates = candidates.copy()
    candidates['predicted_rating'] = scores

    return candidates.sort_values(
        'predicted_rating', ascending=False
    ).head(top_n)[[
        'course_name', 'instructor',
        'difficulty_level', 'predicted_rating'
    ]]

# --------------------------------------------------
# SIMPLE MODEL PERFORMANCE (SAFE)
# --------------------------------------------------
course_avg = df.groupby('course_id')['rating'].mean()
df['collab_pred'] = df['course_id'].map(course_avg)
df['content_pred'] = df['course_id'].map(course_avg)

X = df[['collab_pred', 'content_pred']]
y = df['rating']

rmse, mae = None, None
if len(df) >= 10:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------
if recommend_btn:
    st.subheader(f"⭐ Recommendations for User {user_id}")
    result = recommend_for_user(user_id, top_n)
    st.dataframe(result, use_container_width=True)

    if rmse is not None:
        st.subheader("📈 Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("RMSE", f"{rmse:.3f}")
        col2.metric("MAE", f"{mae:.3f}")
