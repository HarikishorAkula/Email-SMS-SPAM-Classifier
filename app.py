
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Text preprocessing
def transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load Model
tfidf = pickle.load(open('vectorizer (1).pkl','rb'))
model = pickle.load(open('model (1).pkl','rb'))

# Page Config
st.set_page_config(page_title="SMS/Email Spam Detector", layout="centered")

# UI CSS
st.markdown("""
<style>

.stApp {
background: linear-gradient(135deg,#000000,#3a0ca3,#4cc9f0);
}

.main {
background: transparent;
}

h1 {
color:white;
text-align:center;
font-size:42px;
font-weight:800;
letter-spacing:1px;
}

.app-card {
background: rgba(255,255,255,0.18);
backdrop-filter: blur(20px);
border-radius:25px;
padding:35px;
box-shadow:0px 0px 30px rgba(0,0,0,0.35);
border:1px solid rgba(255,255,255,0.25);
}
.stTextArea textarea {
background: rgba(255,255,255,0.95) !important;
color: black !important;
border-radius:18px;
font-size:18px;
padding:18px;
}

/* Placeholder */
.stTextArea textarea::placeholder {
color: #555 !important;
}


.stButton>button {
background: linear-gradient(90deg,#00f260,#0575e6);
color:white;
border-radius:35px;
font-size:21px;
padding:14px 45px;
border:none;
transition:0.4s;
}

.stButton>button:hover {
transform: scale(1.08);
box-shadow:0px 0px 20px #00f260;
}

.spam {
background: linear-gradient(90deg,#ff416c,#ff4b2b);
padding:22px;
border-radius:18px;
color:white;
font-size:32px;
text-align:center;
animation:pulse 1s infinite;
}

.ham {
background: linear-gradient(90deg,#11998e,#38ef7d);
padding:22px;
border-radius:18px;
color:white;
font-size:32px;
text-align:center;
}

.confidence {
text-align:center;
color:white;
font-size:22px;
margin-top:12px;
}

@keyframes pulse {
0% {transform: scale(1);}
50% {transform: scale(1.05);}
100% {transform: scale(1);}
}

.footer {
text-align:center;
color:white;
opacity:0.7;
margin-top:25px;
font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# UI
st.markdown("<div class='app-card'>", unsafe_allow_html=True)

st.title("📧 Real-Time SMS / Email Spam Detector")

input_sms = st.text_area("✍️ Paste your message")

if st.button("🚀 Analyze Message"):

    with st.spinner("Analyzing..."):

        transform_sms = transform(input_sms)
        vector_input = tfidf.transform([transform_sms])

        prediction = model.predict(vector_input)[0]
        prob = model.predict_proba(vector_input)[0]

        confidence = round(max(prob)*100,2)

        if prediction == 1:
            st.markdown("<div class='spam'>🚨 SPAM MESSAGE</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='ham'>✅ SAFE MESSAGE</div>", unsafe_allow_html=True)

        st.markdown(f"<div class='confidence'>Confidence: {confidence}%</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>Built by Hero • Machine Learning Security App</div>", unsafe_allow_html=True)

