import streamlit as st
import joblib
import re
import contractions
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

st.set_page_config(page_title="Sentiment Analysis", page_icon="📝")

st.title("📝 Sentiment Analysis: TF-IDF vs DistilBERT")
st.write("Compare a classical ML baseline against a fine-tuned transformer model.")

# ---------- Preprocessing functions (same as training) ----------
stop_words = set(stopwords.words('english'))
negation_words = {'not', 'no', 'nor', 'never', "n't"}
stop_words = stop_words - negation_words
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text)
    text = BeautifulSoup(text, "html.parser").get_text(separator=' ')
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_for_tfidf(text):
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# ---------- Load models ----------
@st.cache_resource
def load_baseline():
    model = joblib.load("models/baseline_logreg.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer

@st.cache_resource
def load_distilbert():
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/distilbert_sentiment")
    model = DistilBertForSequenceClassification.from_pretrained("models/distilbert_sentiment")
    model.eval()
    return model, tokenizer

baseline_model, tfidf_vectorizer = load_baseline()
bert_model, bert_tokenizer = load_distilbert()

id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}

st.success("Both models loaded successfully!")

# ---------- UI ----------
user_input = st.text_area("Enter a product review:", height=150)

if st.button("Analyze"):
    if user_input.strip():
        word_count = len(user_input.split())
        if word_count < 5:
            st.caption("⚠️ Short inputs may be less reliable — the model was trained mostly on longer reviews.")

        col1, col2 = st.columns(2)

        # Baseline prediction
        clean = clean_text(user_input)
        tfidf_ready = clean_for_tfidf(clean)
        vec = tfidf_vectorizer.transform([tfidf_ready])
        baseline_pred = baseline_model.predict(vec)[0]
        baseline_proba = baseline_model.predict_proba(vec)[0]

        with col1:
            st.subheader("Baseline (TF-IDF + LogReg)")
            st.metric("Prediction", baseline_pred.capitalize())
            st.bar_chart(dict(zip(baseline_model.classes_, baseline_proba)))

        # DistilBERT prediction
        light_clean = clean_text(user_input)
        inputs = bert_tokenizer(light_clean, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        with torch.no_grad():
            outputs = bert_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].numpy()
        bert_pred_id = probs.argmax()
        bert_pred = id_to_label[bert_pred_id]

        with col2:
            st.subheader("DistilBERT")
            st.metric("Prediction", bert_pred.capitalize())
            st.bar_chart({id_to_label[i]: probs[i] for i in range(3)})

    else:
        st.warning("Please enter some text.")