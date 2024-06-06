import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from nltk.tokenize import word_tokenize
import json
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from streamlit_option_menu import option_menu

# Ensure required NLTK tokenizers are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Function to clean text
def clean_review(text):
    text = re.sub("\n", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

# Load normalization dictionary
with open('slang_words.txt', 'r', encoding='utf-8') as file:
    normalization_dict = json.load(file)

# Function to normalize text
def normalize_text(text):
    words = text.split()
    normalized_words = [normalization_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)

# Load stop words
stopwords_set = set()
with open('stop_words.txt', 'r', encoding='utf-8') as file:
    for line in file:
        stopwords_set.add(line.strip())

# Function to remove stop words
def remove_stopwords(text):
    return [word for word in text if word.lower() not in stopwords_set]

# Create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Function to apply stemming
def stemming(text):
    return [stemmer.stem(word) for word in text]

# Function to preprocess text
def preprocess_text(text):
    text = clean_review(text)
    text = normalize_text(text)
    text = word_tokenize(text)
    text = remove_stopwords(text)
    text = stemming(text)
    return ' '.join(text)

# Load pre-selected feature dataframes
resampled_df_ig = pd.read_csv('resampled_df_ig.csv')
resampled_df_chi = pd.read_csv('resampled_df_chi.csv')
resampled_df_selected = pd.read_csv('resampled_df_selected.csv')

# Function to train and evaluate model
def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm_model = SVC(C=10, gamma=1, kernel='rbf')
    
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return svm_model, accuracy, conf_matrix

# Streamlit App
st.set_page_config(page_title="Analisis Sentimen", page_icon="📊", layout="wide")

st.title("📊 Analisis Sentimen dengan SVM")

# Sidebar menu with option_menu
with st.sidebar:
    selected = option_menu(
        "Menu Prediksi",
        ["Prediksi Teks Tunggal", "Prediksi Batch dari CSV"],
        icons=["typewriter", "file-earmark-spreadsheet"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f0f2f6"},
            "icon": {"color": "blue", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#2C6DD5"},
        }
    )

# Prediksi teks tunggal
if selected == "Prediksi Teks Tunggal":
    st.subheader("Prediksi Teks Tunggal")
    user_input = st.text_area("Masukkan kalimat untuk prediksi sentimen:")

    feature_selection_method = st.selectbox("Pilih Metode Seleksi Fitur", ["Information Gain", "Chi-Square", "Combined"])

    if feature_selection_method == "Information Gain":
        resampled_df = resampled_df_ig
    elif feature_selection_method == "Chi-Square":
        resampled_df = resampled_df_chi
    else:
        resampled_df = resampled_df_selected

    if st.button("Prediksi Sentimen") and user_input.strip() != "":
        preprocessed_text = preprocess_text(user_input)

        selected_features = resampled_df.columns.drop('sentimen')

        X_selected = resampled_df[selected_features]
        svm_model, accuracy, conf_matrix = train_and_evaluate_model(X_selected, resampled_df['sentimen'])

        st.write(f"Model Accuracy: {accuracy}")
        st.write("Confusion Matrix:")
        st.write(conf_matrix)

# Prediksi batch dari CSV
elif selected == "Prediksi Batch dari CSV":
    st.subheader("Prediksi Batch dari CSV")
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    feature_selection_method = st.selectbox("Pilih Metode Seleksi Fitur untuk Batch", ["Information Gain", "Chi-Square", "Combined"])

    if feature_selection_method == "Information Gain":
        resampled_df = resampled_df_ig
    elif feature_selection_method == "Chi-Square":
        resampled_df = resampled_df_chi
    else:
        resampled_df = resampled_df_selected

    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        batch_data['cleaned_text'] = batch_data['text'].apply(preprocess_text)

        selected_features = resampled_df.columns.drop('sentimen')
        X_selected = resampled_df[selected_features]
        svm_model, accuracy, conf_matrix = train_and_evaluate_model(X_selected, resampled_df['sentimen'])

        # Predict batch
        batch_data['predicted_sentiment'] = batch_data['cleaned_text'].apply(lambda x: svm_model.predict([x])[0])
        st.write(batch_data)
        st.write(f"Model Accuracy on training data: {accuracy}")
        st.write("Confusion Matrix on training data:")
        st.write(conf_matrix)
