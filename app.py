import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Course Recommender",
    page_icon="🎓",
    layout="wide"
)

st.title("Course Recommendation System")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Excel Dataset", type=["xlsx"]
)

if uploaded_file is None:
    st.info("Please upload an Excel file to continue.")
    st.stop()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_excel(uploaded_file)

# --------------------------------------------------
# REQUIRED COLUMNS
# --------------------------------------------------
required_cols = [
    'user_id', 'course_id', 'course_name', 'instructor',
    'rating', 'difficulty_level', 'course_duration_hours',
    'certification_offered', 'study_material_available',
    'course_price', 'feedback_score'
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"❌ Missing columns: {missing}")
    st.stop()

# --------------------------------------------------
# CLEAN DATA
# --------------------------------------------------
df = df.dropna(subset=['user_id', 'course_id', 'rating'])

df['difficulty_level'] = (
    df['difficulty_level'].astype(str).str.lower().str.strip()
    .map({'beginner': 1, 'intermediate': 2, 'advanced': 3})
)

df['certification_offered'] = (
    df['certification_offered'].astype(str).str.lower().str.strip()
    .map({'yes': 1, 'no': 0})
)

df['study_material_available'] = (
    df['study_material_available'].astype(str).str.lower().str.strip()
    .map({'yes': 1, 'no': 0})
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
st.sidebar.header("⚙️ Recommendation Settings")

user_id = st.sidebar.selectbox(
    "Select User ID", sorted(df['user_id'].unique())
)

top_n = st.sidebar.slider("Top N Recommendations", 1, 10, 5)
run_btn = st.sidebar.button("🔍 Recommend")

# --------------------------------------------------
# RECOMMENDATION LOGIC
# --------------------------------------------------
if run_btn:

    user_course = df.pivot_table(
        index='user_id',
        columns='course_id',
        values='rating',
        aggfunc='mean'
    ).fillna(0)

    if user_course.shape[0] < 2 or user_course.shape[1] < 2:
        st.warning("Not enough users or courses for recommendations.")
        st.stop()

    user_sim = cosine_similarity(user_course)
    user_sim_df = pd.DataFrame(
        user_sim,
        index=user_course.index,
        columns=user_course.index
    )

    rated = df[df['user_id'] == user_id]['course_id'].unique()
    candidates = df[~df['course_id'].isin(rated)].drop_duplicates('course_id')

    if candidates.empty:
        st.warning("User has already rated all courses.")
        st.stop()

    scores = []
    for cid in candidates['course_id']:
        similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:6]
        ratings = user_course.loc[similar_users.index, cid]
        mask = ratings > 0

        if mask.any():
            score = np.average(ratings[mask], weights=similar_users[mask])
        else:
            score = df[df['course_id'] == cid]['rating'].mean()

        scores.append(score)

    candidates = candidates.copy()
    candidates['predicted_rating'] = scores
    candidates = candidates.dropna(subset=['predicted_rating'])

    if candidates.empty:
        st.warning("Not enough data to generate recommendations.")
        st.stop()

    result = candidates.sort_values(
        'predicted_rating', ascending=False
    ).head(top_n)[[
        'course_name', 'instructor', 'difficulty_level', 'predicted_rating'
    ]]

    st.subheader(f"⭐ Recommendations for User {user_id}")
    st.dataframe(result, use_container_width=True)
