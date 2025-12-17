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
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Course Recommender", layout="wide")
st.title("🎓 Personalized Course Recommendation System")

# --------------------------------------------------
# Sidebar – File Upload
# --------------------------------------------------
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel File",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info("Please upload an Excel file to continue.")
    st.stop()

# --------------------------------------------------
# Load Data
# --------------------------------------------------
df = pd.read_excel(uploaded_file)

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
df['certification_offered'] = df['certification_offered'].map({'Yes': 1, 'No': 0})
df['difficulty_level'] = df['difficulty_level'].map({
    'Beginner': 1,
    'Intermediate': 2,
    'Advanced': 3
})
df['study_material_available'] = df['study_material_available'].map({'Yes': 1, 'No': 0})

scaler = MinMaxScaler()
df[['course_duration_hours', 'course_price', 'feedback_score']] = scaler.fit_transform(
    df[['course_duration_hours', 'course_price', 'feedback_score']]
)

# --------------------------------------------------
# Sidebar – User Controls
# --------------------------------------------------
st.sidebar.header("⚙️ Recommendation Settings")

user_id = st.sidebar.selectbox(
    "Select User ID",
    sorted(df['user_id'].unique())
)

top_n = st.sidebar.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)

recommend_btn = st.sidebar.button("🔍 Recommend")

# --------------------------------------------------
# USER–COURSE MATRIX
# --------------------------------------------------
user_course_matrix = df.pivot_table(
    index='user_id',
    columns='course_id',
    values='rating'
)

# --------------------------------------------------
# USER SIMILARITY
# --------------------------------------------------
user_sim = cosine_similarity(user_course_matrix.fillna(0))
user_sim_df = pd.DataFrame(
    user_sim,
    index=user_course_matrix.index,
    columns=user_course_matrix.index
)

# --------------------------------------------------
# COURSE SIMILARITY (CONTENT)
# --------------------------------------------------
numeric_cols = [
    'difficulty_level',
    'course_duration_hours',
    'certification_offered',
    'study_material_available',
    'course_price',
    'feedback_score'
]

course_features = df.groupby('course_id')[numeric_cols].mean()
course_sim = cosine_similarity(course_features)
course_sim_df = pd.DataFrame(
    course_sim,
    index=course_features.index,
    columns=course_features.index
)

# --------------------------------------------------
# PREDICTION FUNCTIONS
# --------------------------------------------------
def user_based_prediction(user_id, course_id, k=5):
    if course_id not in user_course_matrix.columns:
        return np.nan

    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:k+1]
    ratings = user_course_matrix.loc[similar_users.index, course_id].dropna()

    if ratings.empty:
        return np.nan

    return np.average(
        ratings,
        weights=similar_users.loc[ratings.index]
    )


def content_based_prediction(course_id):
    if course_id not in course_sim_df.index:
        return np.nan

    similar_courses = course_sim_df[course_id].sort_values(ascending=False)[1:6]
    ratings = df[df['course_id'].isin(similar_courses.index)]['rating']

    return ratings.mean() if not ratings.empty else np.nan

# --------------------------------------------------
# HYBRID RECOMMENDER
# --------------------------------------------------
def recommend_for_user(user_id, top_n=5):
    rated_courses = df[df['user_id'] == user_id]['course_id'].tolist()

    candidates = df[~df['course_id'].isin(rated_courses)].drop_duplicates('course_id')

    predictions = []

    for _, row in candidates.iterrows():
        cid = row['course_id']

        collab_pred = user_based_prediction(user_id, cid)
        content_pred = content_based_prediction(cid)

        final_score = np.nanmean([collab_pred, content_pred])
        predictions.append(final_score)

    candidates = candidates.copy()
    candidates['predicted_rating'] = predictions

    return candidates.sort_values(
        by='predicted_rating',
        ascending=False
    ).head(top_n)[[
        'course_name',
        'instructor',
        'difficulty_level',
        'predicted_rating'
    ]]

# --------------------------------------------------
# PERFORMANCE (OPTIONAL SIMPLE REGRESSION)
# --------------------------------------------------
course_avg = df.groupby('course_id')['rating'].mean()
df['collab_pred'] = df['course_id'].map(course_avg)
df['content_pred'] = df['course_id'].map(course_avg)

X = df[['collab_pred', 'content_pred']]
y = df['rating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------
if recommend_btn:
    st.subheader(f"⭐ Recommendations for User {user_id}")
    recommendations = recommend_for_user(user_id, top_n)
    st.dataframe(recommendations, use_container_width=True)

    st.subheader("📈 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"{rmse:.3f}")
    col2.metric("MAE", f"{mae:.3f}")
