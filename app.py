import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# ==================================================
# PAGE CONFIG (FIRST STREAMLIT COMMAND)
# ==================================================
st.set_page_config(
    page_title="Course Recommender",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Personalized Course Recommendation System")

# ==================================================
# FILE UPLOAD
# ==================================================
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel File (.xlsx)", type=["xlsx"]
)

if uploaded_file is None:
    st.info("Please upload an Excel file to continue.")
    st.stop()

# ==================================================
# LOAD DATA
# ==================================================
df = pd.read_excel(uploaded_file)

# ==================================================
# REQUIRED COLUMN VALIDATION
# ==================================================
required_cols = [
    'user_id', 'course_id', 'course_name', 'instructor',
    'rating', 'difficulty_level', 'course_duration_hours',
    'certification_offered', 'study_material_available',
    'course_price', 'feedback_score'
]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"❌ Missing required columns: {missing_cols}")
    st.stop()

# ==================================================
# DATA CLEANING
# ==================================================
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
    st.error("❌ Dataset is empty after cleaning.")
    st.stop()

# ==================================================
# SCALE NUMERIC FEATURES
# ==================================================
scaler = MinMaxScaler()
num_cols = ['course_duration_hours', 'course_price', 'feedback_score']
df[num_cols] = scaler.fit_transform(df[num_cols])

# ==================================================
# SIDEBAR CONTROLS
# ==================================================
st.sidebar.header("⚙️ Recommendation Settings")

user_id = st.sidebar.selectbox(
    "Select User ID",
    sorted(df['user_id'].unique())
)

top_n = st.sidebar.slider(
    "Number of Recommendations", 1, 10, 5
)

recommend_btn = st.sidebar.button("🔍 Recommend")

# ==================================================
# RUN RECOMMENDER (SAFE EXECUTION)
# ==================================================
if recommend_btn:

    # ---------- USER-COURSE MATRIX ----------
    user_course = df.pivot_table(
        index='user_id',
        columns='course_id',
        values='rating',
        aggfunc='mean'
    ).fillna(0)

    if user_course.shape[0] < 2 or user_course.shape[1] < 2:
        st.warning("⚠️ Not enough users or courses for recommendations.")
        st.stop()

    user_sim = cosine_similarity(user_course)
    user_sim_df = pd.DataFrame(
        user_sim,
        index=user_course.index,
        columns=user_course.index
    )

    # ---------- COURSE SIMILARITY ----------
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
        st.warning("⚠️ Not enough courses for content-based similarity.")
        st.stop()

    course_sim = cosine_similarity(course_features)
    course_sim_df = pd.DataFrame(
        course_sim,
        index=course_features.index,
        columns=course_features.index
    )

    # ---------- PREDICTION FUNCTIONS ----------
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

    # ---------- CANDIDATE COURSES ----------
    rated_courses = df[df['user_id'] == user_id]['course_id'].unique()
    candidates = df[~df['course_id'].isin(rated_courses)].drop_duplicates('course_id')

    if candidates.empty:
        st.warning("ℹ️ User has already rated all courses.")
        st.stop()

    # ---------- SCORE PREDICTIONS ----------
    scores = []
    for cid in candidates['course_id']:
        score = np.nanmean([user_based(user_id, cid), content_based(cid)])
        scores.append(score)

    candidates = candidates.copy()
    candidates['predicted_rating'] = scores

    candidates = candidates.dropna(subset=['predicted_rating'])

    if candidates.empty:
        st.warning("⚠️ Not enough data to generate recommendations.")
        st.stop()

    # ---------- FINAL OUTPUT ----------
    result = candidates.sort_values(
        'predicted_rating', ascending=False
    ).head(top_n)[[
        'course_name',
        'instructor',
        'difficulty_level',
        'predicted_rating'
    ]]

    st.subheader(f"⭐ Recommendations for User {user_id}")
    st.dataframe(result, use_container_width=True)
