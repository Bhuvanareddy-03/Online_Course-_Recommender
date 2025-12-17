# streamlit_app_min.py
import streamlit as st, pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Course Recommender", layout="wide")
st.title("Course Recommendation System")

# --- Load & preprocess ---
df = pd.read_excel("online_course_recommendation_v2.xlsx")
df['certification_offered'] = df['certification_offered'].map({'Yes':1,'No':0})
df['difficulty_level'] = df['difficulty_level'].map({'Beginner':1,'Intermediate':2,'Advanced':3})
df['study_material_available'] = df['study_material_available'].map({'Yes':1,'No':0})
scaler = MinMaxScaler()
df[['course_duration_hours','course_price','feedback_score']] = scaler.fit_transform(
    df[['course_duration_hours','course_price','feedback_score']]
)

# --- Feature matrices ---
user_course = df.pivot_table(index='user_id', columns='course_id', values='rating')
item_sim = pd.DataFrame(cosine_similarity(user_course.T.fillna(0)), 
                        index=user_course.columns, columns=user_course.columns)
user_sim = pd.DataFrame(cosine_similarity(user_course.fillna(0)), 
                        index=user_course.index, columns=user_course.index)

# --- Baseline preds ---
course_avg = df.groupby('course_id')['rating'].mean()
df['collab_pred'] = df['course_id'].map(course_avg)
course_features = df.groupby('course_id')[['difficulty_level','course_duration_hours',
                                           'certification_offered','study_material_available',
                                           'course_price','feedback_score']].mean()
course_sim = pd.DataFrame(cosine_similarity(course_features.fillna(0)), 
                          index=course_features.index, columns=course_features.index)
df['content_pred'] = df['course_id'].map(course_sim.mean())

# --- Hybrid regression ---
X = df[['content_pred','collab_pred']].fillna(df['rating'].mean())
y = df['rating']
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
reg = LinearRegression().fit(Xtr,ytr)

# --- Recommend functions ---
def item_item(user, n=5):
    rated = user_course.loc[user].dropna()
    scores={}
    for c,r in rated.items():
        for sc,sim in item_sim[c].drop(c).nlargest(10).items():
            if pd.isna(user_course.loc[user,sc]):
                scores[sc]=scores.get(sc,0)+sim*r
    recs = pd.DataFrame(scores.items(),columns=['course_id','score']).nlargest(n,'score')
    return df[df['course_id'].isin(recs.course_id)].drop_duplicates('course_id')[['course_name','instructor']]

def user_user(user,n=5):
    sims=user_sim[user].drop(user).nlargest(5)
    scores={}
    for u,s in sims.items():
        for c,r in user_course.loc[u].dropna().items():
            if pd.isna(user_course.loc[user,c]):
                scores[c]=scores.get(c,0)+s*r
    recs=pd.DataFrame(scores.items(),columns=['course_id','score']).nlargest(n,'score')
    return df[df['course_id'].isin(recs.course_id)].drop_duplicates('course_id')[['course_name','instructor']]

def hybrid(user,n=5):
    rated=df[df.user_id==user].course_id
    unrated=df[~df.course_id.isin(rated)].drop_duplicates('course_id')
    Xun=unrated[['content_pred','collab_pred']].fillna(y.mean())
    unrated['pred']=reg.predict(Xun)
    return unrated.nlargest(n,'pred')[['course_name','instructor','pred']]

# --- UI ---
user=st.sidebar.selectbox("User",df.user_id.unique())
algo=st.sidebar.radio("Algorithm",["Item–Item","User–User","Hybrid"])
if algo=="Item–Item": recs=item_item(user)
elif algo=="User–User": recs=user_user(user)
else: recs=hybrid(user)
st.write(recs)
