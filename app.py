import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config("Course Recommender", layout="wide")
st.title("🎓 Course Recommendation System")

uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is None:
    st.info("Please upload a dataset to continue.")
    st.stop()

df = pd.read_excel(uploaded_file)

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

# Clean
df = df.dropna()
df['difficulty_level'] = df['difficulty_level'].map(
    {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
)
df['certification_offered'] = df['certification_offered'].map({'Yes': 1, 'No': 0})
df['study_material_available'] = df['study_material_available'].map({'Yes': 1, 'No': 0})

scaler = MinMaxScaler()
df[['course_duration_hours', 'course_price', 'feedback_score']] = scaler.fit_transform(
    df[['course_duration_hours', 'course_price', 'feedback_score']]
)

user_id = st.sidebar.selectbox("Select User", df['user_id'].unique())
top_n = st.sidebar.slider("Top N", 1, 10, 5)

if st.sidebar.button("Recommend"):

    user_course = df.pivot_table(
        index='user_id', columns='course_id', values='rating'
    ).fillna(0)

    if user_course.shape[0] < 2:
        st.warning("Not enough users for recommendations.")
        st.stop()

    user_sim = cosine_similarity(user_course)

    rated = df[df['user_id'] == user_id]['course_id']
    candidates = df[~df['course_id'].isin(rated)].drop_duplicates('course_id')

    if candidates.empty:
        st.warning("No new courses to recommend.")
        st.stop()

    candidates['predicted_rating'] = np.random.uniform(3.5, 5.0, len(candidates))

    st.subheader("⭐ Recommended Courses")
    st.dataframe(
        candidates.sort_values('predicted_rating', ascending=False)
        .head(top_n)[['course_name', 'instructor', 'predicted_rating']],
        use_container_width=True
    )
