import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------
# SIDEBAR THEME SWITCHER
# ----------------------------------------------------------
st.sidebar.header("🎨 Theme Customization")

theme_choice = st.sidebar.selectbox(
    "Choose Background Theme",
    ["Light Blue", "Dark Mode", "Gradient Purple", "Mint Green"]
)

# ----------------------------------------------------------
# DYNAMIC BACKGROUND CSS
# ----------------------------------------------------------
if theme_choice == "Light Blue":
    page_bg = """
    <style>
        .stApp { background-color: #e7f3ff !important; }
        body { background-color: #e7f3ff !important; }
        .css-1d391kg { background-color: #d9ecff !important; }
    </style>
    """

elif theme_choice == "Dark Mode":
    page_bg = """
    <style>
        .stApp { background-color: #1e1e1e !important; color: white !important; }
        body { background-color: #1e1e1e !important; color: white !important; }
        .css-1d391kg { background-color: #111 !important; }
        h1, h2, h3, h4, h5, h6, p, label, .stMetric { color: white !important; }
    </style>
    """

elif theme_choice == "Gradient Purple":
    page_bg = """
    <style>
        .stApp {
            background: linear-gradient(to bottom right, #8e2de2, #4a00e0) !important;
            color: white !important;
        }
        body {
            background: linear-gradient(to bottom right, #8e2de2, #4a00e0) !important;
        }
        .css-1d391kg { background-color: #3d009b !important; }
        h1, h2, h3, h4, h5, h6, p, label { color: white !important; }
    </style>
    """

elif theme_choice == "Mint Green":
    page_bg = """
    <style>
        .stApp { background-color: #d7fff1 !important; }
        body { background-color: #d7fff1 !important; }
        .css-1d391kg { background-color: #b8f7e6 !important; }
    </style>
    """

st.markdown(page_bg, unsafe_allow_html=True)

# ----------------------------------------------------------
# RECOMMENDATION FUNCTION (WITH SCORES)
# ----------------------------------------------------------
def content_based_recommendations(df2, course_name, top_n=10):
    if course_name not in df2['course_name'].values:
        return pd.DataFrame(), []

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df2['course_name'])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    item_index = df2[df2['course_name'] == course_name].index[0]

    similar_items = sorted(
        list(enumerate(cosine_sim[item_index])),
        key=lambda x: x[1],
        reverse=True
    )

    top_similar = similar_items[1:top_n + 1]
    indices = [x[0] for x in top_similar]
    scores = [x[1] for x in top_similar]

    result = df2.iloc[indices][[
        'course_name',
        'instructor',
        'certification_offered',
        'difficulty_level',
        'study_material_available'
    ]]

    return result, scores

# ----------------------------------------------------------
# MAIN UI
# ----------------------------------------------------------
st.title("📘 Course Recommendation System")
st.markdown("### Find similar courses using TF-IDF + Cosine Similarity + Evaluation Metrics")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls"])

if uploaded_file:

    ext = uploaded_file.name.split(".")[-1]
    df2 = pd.read_csv(uploaded_file) if ext == "csv" else pd.read_excel(uploaded_file)

    st.success("File loaded successfully!")

    if st.checkbox("Show Dataset Preview"):
        st.dataframe(df2.head())

    course_name = st.selectbox("Select a Course", df2['course_name'].unique().tolist())

    top_n = st.slider("Number of Recommendations", 3, 20, 10)

    if st.button("Get Recommendations"):
        results, scores = content_based_recommendations(df2, course_name, top_n)

        if results.empty:
            st.warning("Course not found.")
        else:
            st.subheader("Recommended Courses")
            st.table(results)

            # ----------------------------------------------------------
            # EVALUATION METRICS SECTION
            # ----------------------------------------------------------
            st.subheader("📊 Evaluation Metrics")

            scores_np = np.array(scores)

            col1, col2, col3 = st.columns(3)

            col1.metric("Average Similarity", round(scores_np.mean(), 4))
            col2.metric("Highest Similarity", round(scores_np.max(), 4))
            col3.metric("Lowest Similarity", round(scores_np.min(), 4))

            st.write("### Similarity Score Table")
            score_df = pd.DataFrame({
                "Course": results['course_name'],
                "Similarity Score": scores
            })
            st.table(score_df)

            st.write("### 📈 Similarity Score Trend")
            st.line_chart(scores_np)

else:
    st.info("Please upload a dataset.")
