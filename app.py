import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

st.title("📚 Course Recommendation System (User–User & Item–Item)")

# ===========================
# File Upload
# ===========================
uploaded_file = st.sidebar.file_uploader("Upload dataset (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.info("⬆️ Please upload your dataset to continue.")
    st.stop()

df = pd.read_excel(uploaded_file)

# ===========================
# Preprocessing
# ===========================
df['certification_offered'] = df['certification_offered'].map({'Yes': 1, 'No': 0})
df['difficulty_level'] = df['difficulty_level'].map({'Beginner': 1, 'Intermediate': 2, 'Advanced': 3})
df['study_material_available'] = df['study_material_available'].map({'Yes': 1, 'No': 0})

scaler = MinMaxScaler()
df[['course_duration_hours', 'course_price', 'feedback_score']] = scaler.fit_transform(
    df[['course_duration_hours', 'course_price', 'feedback_score']]
)

ratings = df.pivot_table(index='user_id', columns='course_id', values='rating')

# ===========================
# Similarities
# ===========================
user_sim = pd.DataFrame(cosine_similarity(ratings.fillna(0)),
                        index=ratings.index, columns=ratings.index)
item_sim = pd.DataFrame(cosine_similarity(ratings.fillna(0).T),
                        index=ratings.columns, columns=ratings.columns)

# ===========================
# Visualizations
# ===========================
st.subheader("📊 Dataset Overview")

fig1, ax1 = plt.subplots(figsize=(8,5))
sns.histplot(df['rating'], bins=10, kde=True, color="purple", ax=ax1)
ax1.set_title("Distribution of Course Ratings")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(6,4))
sns.countplot(x='difficulty_level', data=df, palette="viridis", ax=ax2)
ax2.set_title("Course Counts by Difficulty Level")
st.pyplot(fig2)

cert_counts = df['certification_offered'].value_counts()
fig3, ax3 = plt.subplots(figsize=(6,6))
ax3.pie(cert_counts, labels=['Yes','No'], autopct='%1.1f%%', colors=['skyblue','lightcoral'])
ax3.set_title("Certification Offered Distribution")
st.pyplot(fig3)

# Sample heatmaps to avoid crashes
st.subheader("🔎 Sampled Similarity Heatmaps")

fig4, ax4 = plt.subplots(figsize=(8,6))
sns.heatmap(user_sim.iloc[:20,:20], cmap="coolwarm", ax=ax4)
ax4.set_title("User–User Similarity (Sampled)")
st.pyplot(fig4)

fig5, ax5 = plt.subplots(figsize=(8,6))
sns.heatmap(item_sim.iloc[:20,:20], cmap="YlGnBu", ax=ax5)
ax5.set_title("Item–Item Similarity (Sampled)")
st.pyplot(fig5)

# ===========================
# Predictors
# ===========================
def user_cf(user, course):
    sims = user_sim[user].drop(user).sort_values(ascending=False)
    raters = ratings.loc[sims.index, course].dropna()
    return raters.mean() if not raters.empty else np.nan

def item_cf(user, course):
    rated = ratings.loc[user].dropna()
    if rated.empty: return np.nan
    sims = item_sim.loc[course, rated.index]
    return (rated.values * sims.values).sum() / sims.sum()

# ===========================
# Hybrid Regression
# ===========================
df['user_cf'] = [user_cf(u,c) for u,c in zip(df['user_id'], df['course_id'])]
df['item_cf'] = [item_cf(u,c) for u,c in zip(df['user_id'], df['course_id'])]

X = df[['user_cf','item_cf']].fillna(df['rating'].mean())
y = df['rating']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LinearRegression().fit(X_train,y_train)
preds = model.predict(X_test)

reg_rmse = np.sqrt(mean_squared_error(y_test,preds))
reg_mae = mean_absolute_error(y_test,preds)

# ===========================
# Sidebar Controls
# ===========================
st.sidebar.header("Recommendation Settings")
user_id = st.sidebar.selectbox("Select User ID", df['user_id'].unique())
top_n = st.sidebar.slider("Number of Recommendations", 3, 10, 5)

# ===========================
# Recommendations
# ===========================
def recommend(user, top_n=5):
    courses = ratings.columns
    scores = []
    for c in courses:
        if pd.notna(ratings.loc[user,c]): continue
        f = pd.DataFrame([[user_cf(user,c), item_cf(user,c)]],
                         columns=['user_cf','item_cf']).fillna(0)
        score = model.predict(f)[0]
        scores.append((c,score))
    recs = pd.DataFrame(scores, columns=['course_id','pred']).sort_values('pred',ascending=False).head(top_n)
    return df.drop_duplicates('course_id').set_index('course_id').loc[recs['course_id'],['course_name','instructor']].assign(pred=recs['pred'].values)

recs = recommend(user_id, top_n)

st.subheader(f"🎯 Top {top_n} Recommendations for User {user_id}")
st.dataframe(recs)

fig6, ax6 = plt.subplots(figsize=(8,5))
sns.barplot(x=recs['course_name'], y=recs['pred'], palette="viridis", ax=ax6)
ax6.set_title(f"Top {top_n} Recommended Courses for User {user_id}")
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
ax6.set_ylabel("Predicted Rating")
st.pyplot(fig6)

# ===========================
# Performance Comparison
# ===========================
st.subheader("📈 Model Performance Comparison")
results = []
for col,label in [('user_cf','User CF'),('item_cf','Item CF')]:
    valid = df[~df[col].isna()]
    rmse = np.sqrt(mean_squared_error(valid['rating'], valid[col]))
    mae = mean_absolute_error(valid['rating'], valid[col])
    results.append((label, rmse, mae))
results.append(('Hybrid', reg_rmse, reg_mae))
res_df = pd.DataFrame(results, columns=['Model','RMSE','MAE'])
st.dataframe(res_df)

fig7, ax7 = plt.subplots(figsize=(10,6))
x = np.arange(len(res_df))
width = 0.35
ax7.bar(x - width/2, res_df['RMSE'], width, label='RMSE', color='skyblue')
ax7.bar(x + width/2, res_df['MAE'], width, label='MAE', color='salmon')
ax7.set_ylabel('Error')
ax7.set_title('Model Performance Comparison')
ax7.set_xticks(x)
ax7.set_xticklabels(res_df['Model'])
ax7.legend()
st.pyplot(fig7)
