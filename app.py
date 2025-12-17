# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Course Recommendation System", layout="wide")
st.title("📚 Online Course Recommendation System")

# --- Load data ---
uploaded = st.file_uploader("Upload dataset (.xlsx)", type=["xlsx"])
if uploaded is None:
    st.info("Please upload the dataset to continue.")
    st.stop()

df = pd.read_excel(uploaded)

# --- Preprocessing ---
df['certification_offered'] = df['certification_offered'].map({'Yes': 1, 'No': 0})
df['difficulty_level'] = df['difficulty_level'].map({'Beginner': 1, 'Intermediate': 2, 'Advanced': 3})
df['study_material_available'] = df['study_material_available'].map({'Yes': 1, 'No': 0})

scaler = MinMaxScaler()
df[['course_duration_hours', 'course_price', 'feedback_score']] = scaler.fit_transform(
    df[['course_duration_hours', 'course_price', 'feedback_score']]
)

# --- Content-based similarity ---
numeric_cols = ['difficulty_level','course_duration_hours','certification_offered',
                'study_material_available','course_price','feedback_score']
course_features = df.groupby('course_id')[numeric_cols].mean()
course_sim = cosine_similarity(course_features)
course_sim_df = pd.DataFrame(course_sim, index=course_features.index, columns=course_features.index)

# --- Collaborative filtering baseline ---
course_avg = df.groupby('course_id')['rating'].mean()

content_preds = []
for _, row in df.iterrows():
    cid = row['course_id']
    if cid in course_sim_df.index:
        similar = course_sim_df[cid].sort_values(ascending=False)[1:6]
        sim_ratings = df[df['course_id'].isin(similar.index)]['rating']
        pred = sim_ratings.mean() if not sim_ratings.empty else np.nan
    else:
        pred = np.nan
    content_preds.append(pred)

df['content_pred'] = content_preds
df['collab_pred'] = df['course_id'].map(course_avg)

# --- Regression model ---
X = df[['content_pred','collab_pred']]
y = df['rating']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
reg_model = LinearRegression().fit(X_train,y_train)
y_pred = reg_model.predict(X_test)
reg_rmse = np.sqrt(mean_squared_error(y_test,y_pred))
reg_mae = mean_absolute_error(y_test,y_pred)

st.subheader("Regression Model Performance")
st.write(f"**RMSE:** {reg_rmse:.3f}, **MAE:** {reg_mae:.3f}")

# --- Recommendation function ---
def recommend_with_regression(user_id, top_n=5):
    rated = df[df['user_id']==user_id]['course_id'].tolist()
    unrated = df[~df['course_id'].isin(rated)].drop_duplicates('course_id')
    X_unrated = unrated[['content_pred','collab_pred']].fillna(course_avg.mean())
    unrated['predicted_rating'] = reg_model.predict(X_unrated)
    return unrated.sort_values('predicted_rating',ascending=False).head(top_n)[
        ['course_name','instructor','difficulty_level','predicted_rating']
    ]

# --- Sidebar controls ---
st.sidebar.header("Recommendation Settings")
user_id = st.sidebar.number_input("User ID", min_value=int(df['user_id'].min()), 
                                  max_value=int(df['user_id'].max()), value=int(df['user_id'].min()))
top_n = st.sidebar.slider("Top N Recommendations",3,10,5)

# --- Show recommendations ---
recs = recommend_with_regression(user_id, top_n)
st.subheader(f"Top {top_n} Recommendations for User {user_id}")
st.dataframe(recs)
