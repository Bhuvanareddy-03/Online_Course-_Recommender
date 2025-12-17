import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# Load Data
@st.cache_data
def load_data():
    df = pd.read_excel("online_course_recommendation_v2 (1) (1).xlsx")
    return df

df = load_data()


# Prepare Combined Features
df['combined_features'] = (
    df['course_name'].astype(str) + ' ' +
    df['instructor'].astype(str) + ' ' +
    df['difficulty_level'].astype(str)
)

# Build TF-IDF & Nearest Neighbors Model
vectorizer = TfidfVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(df['combined_features'])

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
model_knn.fit(feature_matrix)


# Recommendation Function
def recommend_courses(keyword, difficulty, engagement, n=5):
    # Filter by difficulty
    subset = df.copy()
    if difficulty != "Any":
        subset = subset[subset['difficulty_level'] == difficulty]

    # Filter by engagement (based on time spent)
    if engagement == "High":
        subset = subset[subset['time_spent_hours'] > 60]
    elif engagement == "Medium":
        subset = subset[(subset['time_spent_hours'] >= 30) & (subset['time_spent_hours'] <= 60)]
    elif engagement == "Low":
        subset = subset[subset['time_spent_hours'] < 30]

    # Find matching course names
    matches = df[df['course_name'].str.contains(keyword, case=False, na=False)]
    if matches.empty:
        return pd.DataFrame({'Message': [f"No matching courses found for '{keyword}'. Try another."]})
    
    # Get first matching course
    idx = matches.index[0]
    distances, indices = model_knn.kneighbors(feature_matrix[idx], n_neighbors=n+10)
    recs = df.iloc[indices.flatten()[1:]][[
        'course_name', 'instructor', 'difficulty_level', 'rating', 'feedback_score'
    ]]
    recs['similarity_score'] = 1 - distances.flatten()[1:]
    
    # Remove duplicates for variety
    recs = recs.drop_duplicates(subset=['course_name']).head(n)

    # Combine content similarity with user filters
    recs = recs[recs['difficulty_level'].isin(subset['difficulty_level'])]
    return recs.reset_index(drop=True)

# Streamlit App UI
st.title(" Online Course Recommendation System")

st.write("Find online courses based on your interests, difficulty level, and engagement preference.")

# Inputs
keyword = st.text_input("Enter a topic or keyword (e.g., Python, Data Science, Java):")
difficulty = st.selectbox("Preferred Difficulty Level:", ["Any", "Beginner", "Intermediate", "Advanced"])
engagement = st.selectbox("Your Engagement Level:", ["Any", "Low", "Medium", "High"])

# Button
if st.button(" Recommend Courses"):
    if not keyword.strip():
        st.warning("Please enter a keyword to get recommendations.")
    else:
        results = recommend_courses(keyword, difficulty, engagement)
        st.subheader("Recommended Courses:")
        st.dataframe(results)

st.markdown("---")
st.markdown("Built with using Streamlit and Scikit-Learn")
