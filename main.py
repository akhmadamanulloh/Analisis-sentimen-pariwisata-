import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif, chi2
import time
import pickle
import json
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

# Pastikan model tokenizer NLTK yang diperlukan sudah terunduh
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Load data yang telah dibersihkan
with open('df_cleaned (1).pickle', 'rb') as handle:
    df_cleaned = pickle.load(handle)

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
        word = line.strip()
        stopwords_set.add(word)

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

# Inisialisasi TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
tfidf_matrix = tfidf_vectorizer.fit_transform(df_cleaned['text'])

# Buat DataFrame untuk TF-IDF
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Muat kolom sentimen
y = df_cleaned['sentimen']

# Inisialisasi SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(tfidf_matrix, y)

# Buat DataFrame resampled
resampled_df = pd.DataFrame(X_resampled.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
resampled_df['sentimen'] = y_resampled

# Seleksi fitur - Information Gain
ig_scores = mutual_info_classif(X_resampled, y_resampled, random_state=42)
ig_scores_df = pd.DataFrame({'Feature': tfidf_vectorizer.get_feature_names_out(), 'Information_Gain': ig_scores})
ig_scores_df_sorted = ig_scores_df.sort_values(by='Information_Gain', ascending=False)
top_features_count_ig = int(len(ig_scores_df_sorted) * 0.50)
top_features_df_ig = ig_scores_df_sorted.head(top_features_count_ig)
ig_selected_features = set(top_features_df_ig['Feature'])

# Seleksi fitur - Chi-Square
chi2_scores, _ = chi2(X_resampled, y_resampled)
chi2_scores_df = pd.DataFrame({'Feature': tfidf_vectorizer.get_feature_names_out(), 'Chi_Square_Score': chi2_scores})
chi2_scores_df_sorted = chi2_scores_df.sort_values(by='Chi_Square_Score', ascending=False)
top_features_count_chi = int(len(chi2_scores_df_sorted) * 0.50)
top_features_df_chi = chi2_scores_df_sorted.head(top_features_count_chi)
chi_selected_features = set(top_features_df_chi['Feature'])

# Fitur terpilih gabungan
combined_selected_features = ig_selected_features.intersection(chi_selected_features)

# Fungsi untuk melatih dan mengevaluasi model
def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm_model = SVC(C=10, gamma=1, kernel='rbf')
    
    start_training_time = time.time()
    svm_model.fit(X_train, y_train)
    end_training_time = time.time()
    training_time = end_training_time - start_training_time

    start_testing_time = time.time()
    y_pred = svm_model.predict(X_test)
    end_testing_time = time.time()
    testing_time = end_testing_time - start_testing_time

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return svm_model, training_time, testing_time, accuracy, conf_matrix

# Aplikasi Streamlit
st.title("Analisis Sentimen dengan SVM")

# Input teks untuk prediksi tunggal
st.subheader("Prediksi Teks Tunggal")
user_input = st.text_area("Masukkan teks untuk prediksi sentimen:")
feature_selection_method = st.selectbox("Pilih Metode Seleksi Fitur", ["Information Gain", "Chi-Square", "Combined"])

if st.button("Prediksi Sentimen"):
    preprocessed_text = preprocess_text(user_input)
    vectorized_text = tfidf_vectorizer.transform([preprocessed_text])
    
    if feature_selection_method == "Information Gain":
        selected_features = ig_selected_features
    elif feature_selection_method == "Chi-Square":
        selected_features = chi_selected_features
    else:
        selected_features = combined_selected_features
    
    selected_features_list = list(selected_features)
    selected_vectorized_text = vectorized_text[:, [tfidf_vectorizer.vocabulary_[word] for word in selected_features_list if word in tfidf_vectorizer.vocabulary_]]
    
    X_selected = resampled_df[selected_features_list]
    svm_model, training_time, testing_time, accuracy, conf_matrix = train_and_evaluate_model(X_selected, resampled_df['sentimen'])
    
    prediction = svm_model.predict(selected_vectorized_text)
    sentiment = "Positif" if prediction[0] == 1 else "Negatif"
    st.write(f"Prediksi Sentimen: {sentiment}")

# Unggah file untuk prediksi batch
st.subheader("Prediksi Batch dari CSV")
uploaded_file = st.file_uploader("Unggah file CSV", type="csv")

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)
    if 'text' not in df_uploaded.columns:
        st.error("File CSV yang diunggah harus berisi kolom 'text'.")
    else:
        df_uploaded['cleaned_text'] = df_uploaded['text'].apply(preprocess_text)
        vectorized_texts = tfidf_vectorizer.transform(df_uploaded['cleaned_text'])
        
        if feature_selection_method == "Information Gain":
            selected_features = ig_selected_features
        elif feature_selection_method == "Chi-Square":
            selected_features = chi_selected_features
        else:
            selected_features = combined_selected_features
        
        selected_features_list = list(selected_features)
        selected_vectorized_texts = vectorized_texts[:, [tfidf_vectorizer.vocabulary_[word] for word in selected_features_list if word in tfidf_vectorizer.vocabulary_]]
        
        X_selected = resampled_df[selected_features_list]
        svm_model, training_time, testing_time, accuracy, conf_matrix = train_and_evaluate_model(X_selected, resampled_df['sentimen'])
        
        predictions = svm_model.predict(selected_vectorized_texts)
        df_uploaded['predicted_sentiment'] = ["Positif" if pred == 1 else "Negatif" for pred in predictions]
        
        st.write(df_uploaded)

        # Unduh hasil prediksi
        csv = df_uploaded.to_csv(index=False)
        st.download_button(label="Unduh Prediksi", data=csv, file_name='predictions.csv', mime='text/csv')
