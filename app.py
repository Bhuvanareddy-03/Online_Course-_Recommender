import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# --------------------------------------------------
st.set_page_config(page_title="Course Recommender", layout="wide")
st.title("Course Recommendation System")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Excel Dataset", type=["xlsx"]
)

if uploaded_file is None:
    st.info("Upload an Excel file to start.")
    st.stop()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_excel(uploaded_file)

# --------------------------------------------------
# REQUIRED COLUMNS CHECK
# --------------------------------------------------
required_cols = [
    'user_id', 'course_id', 'course_name', 'instructor',
    'rating', 'difficulty_level', 'course_duration_hours',
    'certification_offered', 'study_material_available',
    'course_price', 'feedback_score'
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# --------------------------------------------------
# CLEAN DATA
# --------------------------------------------------
df = df.dropna(subset=['user_id', 'course_id', 'rating'])

df['certification_offered'] = (
    df['certification_offered'].astype(str).str.lower().str.strip()
    .map({'yes': 1, 'no': 0})
)

df['study_material_available'] = (
    df['study_material_available'].astype(str).str.lower().str.strip()
    .map({'yes': 1, 'no': 0})
)

df['difficulty_level'] = (
    df['difficulty_level'].astype(str).str.lower().str.strip()
    .map({'beginner': 1, 'intermediate': 2, 'advanced': 3})
)

df = df.dropna()

if df.empty:
    st.error("Dataset is empty after cleaning.")
    st.stop()

# --------------------------------------------------
# SCALE NUMERIC FEATURES
# --------------------------------------------------
scaler = MinMaxScaler()
num_cols = ['course_duration_hours', 'course_price', 'feedback_score']
df[num_cols] = scaler.fit_transform(df[num_cols])

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
user_id = st.sidebar.selectbox(
    "Select User ID", sorted(df['user_id'].unique())
)
top_n = st.sidebar.slider("Top N Recommendations", 1, 10, 5)
run_btn = st.sidebar.button("🔍 Recommend")

# --------------------------------------------------
# RUN RECOMMENDER (LAZY EXECUTION)
# --------------------------------------------------
if run_btn:

    # ---- USER-COURSE MATRIX
    user_course = df.pivot_table(
        index='user_id',
        columns='course_id',
        values='rating',
        aggfunc='mean'
    ).fillna(0)

    if user_course.shape[0] < 2 or user_course.shape[1] < 2:
        st.error("Need at least 2 users and 2 courses for recommendations.")
        st.stop()

    user_sim = cosine_similarity(user_course)
    user_sim_df = pd.DataFrame(
        user_sim,
        index=user_course.index,
        columns=user_course.index
    )

    # ---- COURSE SIMILARITY
    feature_cols = [
        'difficulty_level',
        'course_duration_hours',
        'certification_offered',
        'study_material_available',
        'course_price',
        'feedback_score'
    ]

    course_features = df.groupby('course_id')[feature_cols].mean()

    if course_features.shape[0] < 2:
        st.error("Not enough courses for similarity calculation.")
        st.stop()

    course_sim = cosine_similarity(course_features)
    course_sim_df = pd.DataFrame(
        course_sim,
        index=course_features.index,
        columns=course_features.index
    )

    # ---- PREDICTION FUNCTIONS
    def user_based(uid, cid, k=5):
        sims = user_sim_df.loc[uid].sort_values(ascending=False)[1:k+1]
        ratings = user_course.loc[sims.index, cid]
        mask = ratings > 0
        if not mask.any():
            return np.nan
        return np.average(ratings[mask], weights=sims[mask])

    def content_based(cid):
        sims = course_sim_df[cid].sort_values(ascending=False)[1:6]
        ratings = df[df['course_id'].isin(sims.index)]['rating']
        return ratings.mean() if not ratings.empty else np.nan

    # ---- RECOMMENDATION
    rated = df[df['user_id'] == user_id]['course_id'].unique()
    candidates = df[~df['course_id'].isin(rated)].drop_duplicates('course_id')

    scores = []
    for cid in candidates['course_id']:
        scores.append(np.nanmean([user_based(user_id, cid), content_based(cid)]))

    candidates['predicted_rating'] = scores

    result = candidates.sort_values(
        'predicted_rating', ascending=False
    ).head(top_n)[[
        'course_name', 'instructor', 'difficulty_level', 'predicted_rating'
    ]]

    st.subheader(f"⭐ Recommendations for User {user_id}")
    st.dataframe(result, use_container_width=True)
