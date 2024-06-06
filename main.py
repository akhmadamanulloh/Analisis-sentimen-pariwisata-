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

# Pastikan model tokenizer NLTK yang diperlukan sudah terunduh
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Fungsi untuk membersihkan teks
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

# Muat kamus normalisasi
with open('slang_words.txt', 'r', encoding='utf-8') as file:
    normalization_dict = json.load(file)

# Fungsi untuk menormalkan teks
def normalize_text(text):
    words = text.split()
    normalized_words = [normalization_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)

# Muat stop words
stopwords_set = set()
with open('stop_words.txt', 'r', encoding='utf-8') as file:
    for line in file:
        stopwords_set.add(line.strip())

# Fungsi untuk menghapus stop words
def remove_stopwords(text):
    return [word for word in text if word.lower() not in stopwords_set]

# Buat stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi untuk menerapkan stemming
def stemming(text):
    return [stemmer.stem(word) for word in text]

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    text = clean_review(text)
    text = normalize_text(text)
    text = word_tokenize(text)
    text = remove_stopwords(text)
    text = stemming(text)
    return ' '.join(text)

# Muat DataFrame yang telah diseleksi fiturnya
resampled_df_ig = pd.read_csv('resampled_df_ig.csv')
resampled_df_chi = pd.read_csv('resampled_df_chi.csv')
resampled_df_selected = pd.read_csv('resampled_df_selected.csv')

# Fungsi untuk melatih dan mengevaluasi model
def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm_model = SVC(C=10, gamma=1, kernel='rbf')
    
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return svm_model, accuracy, conf_matrix

# Aplikasi Streamlit
st.title("Analisis Sentimen dengan SVM")

with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Prediksi Teks Tunggal", "Prediksi Batch dari CSV"],
        icons=["file-text", "file-upload"],
        menu_icon="cast",
        default_index=0,
    )

# Prediksi Teks Tunggal
if selected == "Prediksi Teks Tunggal":
    st.subheader("Prediksi Teks Tunggal")
    user_input = st.text_area("Masukkan teks untuk prediksi sentimen:")
    feature_selection_method = st.selectbox("Pilih Metode Seleksi Fitur", ["Information Gain", "Chi-Square", "Combined"])

    if st.button("Prediksi Sentimen"):
        if not user_input.strip():
            st.error("Silakan masukkan kalimat.")
        else:
            preprocessed_text = preprocess_text(user_input)
            
            if feature_selection_method == "Information Gain":
                resampled_df = resampled_df_ig
            elif feature_selection_method == "Chi-Square":
                resampled_df = resampled_df_chi
            else:
                resampled_df = resampled_df_selected
            
            selected_features = resampled_df.columns.drop('sentimen')
            
            X_selected = resampled_df[selected_features]
            svm_model, accuracy, conf_matrix = train_and_evaluate_model(X_selected, resampled_df['sentimen'])
            
            vectorized_text = pd.DataFrame([dict.fromkeys(selected_features, 0)])
            for word in preprocessed_text.split():
                if word in vectorized_text.columns:
                    vectorized_text[word] += 1
            
            prediction = svm_model.predict(vectorized_text)
            sentiment = "Positif" if prediction[0] == 1 else "Negatif"
            st.write(f"Prediksi Sentimen: {sentiment}")

# Prediksi Batch dari CSV
elif selected == "Prediksi Batch dari CSV":
    st.subheader("Prediksi Batch dari CSV")
    uploaded_file = st.file_uploader("Unggah file CSV", type="csv")

    if uploaded_file is not None:
        feature_selection_method = st.selectbox("Pilih Metode Seleksi Fitur", ["Information Gain", "Chi-Square", "Combined"])
            
            if feature_selection_method == "Information Gain":
                resampled_df = resampled_df_ig
            elif feature_selection_method == "Chi-Square":
                resampled_df = resampled_df_chi
            else:
                resampled_df = resampled_df_selected
        df_uploaded = pd.read_csv(uploaded_file)
        if 'text' not in df_uploaded.columns:
            st.error("File CSV yang diunggah harus berisi kolom 'text'.")
        else:
            df_uploaded['cleaned_text'] = df_uploaded['text'].apply(preprocess_text)
            
            selected_features = resampled_df.columns.drop('sentimen')
            
            X_selected = resampled_df[selected_features]
            svm_model, accuracy, conf_matrix = train_and_evaluate_model(X_selected, resampled_df['sentimen'])
            
            vectorized_texts = pd.DataFrame(columns=selected_features)
            for text in df_uploaded['cleaned_text']:
                vectorized_text = dict.fromkeys(selected_features, 0)
                for word in text.split():
                    if word in vectorized_text:
                        vectorized_text[word] += 1
                vectorized_texts = vectorized_texts.append(vectorized_text, ignore_index=True)
            
            predictions = svm_model.predict(vectorized_texts)
            df_uploaded['predicted_sentiment'] = ["Positif" if pred == 1 else "Negatif" for pred in predictions]
            
            st.write(df_uploaded)

            # Unduh hasil prediksi
            csv = df_uploaded.to_csv(index=False)
            st.download_button(label="Unduh Prediksi", data=csv, file_name='predictions.csv', mime='text/csv')
