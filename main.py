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
    text = text.lower()  # Case folding
    text = clean_review(text)
    text = normalize_text(text)
    text = word_tokenize(text)  # Tokenizing
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
st.title("Analisis Sentimen Destinasi Pariwisata Melalui Ulasan Google Maps Menggunakan Support Vector Machine Dan Kombinasi Seleksi Fitur")

with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Home", "Dataset", "Grafik", "Prediksi Teks Tunggal", "Prediksi Batch dari CSV"],
        icons=["house", "table", "bar-chart", "file-text", "file-text"],
        menu_icon="cast",
        default_index=0,
    )

# Home
if selected == "Home":
    st.image("https://assets.promediateknologi.id/crop/0x0:0x0/750x500/webp/photo/2023/02/09/3724430694.jpg", caption="Destinasi Pariwisata Jombang")
    st.subheader("Penjelasan Singkat")
    st.write("""
        <div style="text-align:justify;">
        Pariwisata merupakan industri penting dalam pembangunan perekonomian suatu daerah. Salah satu daerah yang mengembangkan industri pariwisata adalah Kabupaten Jombang. Kabupaten Jombang yang 
        terletak di provinsi Jawa Timur. Kabupaten Jombang memiliki berbagai keindahan alam dan potensi pariwisata yang menarik, 
        karena posisi Kabupaten Jombang yang bersebelahan dengan daerah tujuan wisata alam Malang di tenggara serta wisata historis 
        (situs Majapahit) Trowulan. Jombang memiliki beberapa tempat pariwisata dan budaya yang menarik, 
        terdiri dari wisata buatan, wisata alam, dan wisata religi. Banyaknya ulasan pengunjung terhadap pariwisata di Kabupaten Jombang di google maps, secara manual 
        menganalisis dan memahami banyak memakan banyak waktu. Oleh karena itu, diperlukan metode yang mampu mengolah data dengan cepat dan akurat 
        untuk menganalisis sentimen ulasan pariwisata di Kabupaten Jombang. Metode yang akan digunakan adalah Analisis Sentimen menggunakan Metode Support Vector Machine dengan kombinasi seleksi fitur information gain dan Chi-Square. 
        <br><br>
        <b>Analisis Sentimen</b> merupakan salah satu cabang dari text mining yang bertugas mengklasifikasikan dokumen teks. Dalam prosesnya, Analisis Sentimen mampu mengekstraksi komentar, emosi, dan penilaian tertulis seseorang mengenai suatu topik tertentu dengan memanfaatkan teknik pemrosesan Bahasa alami, seperti menilai apakah teks tersebut bersifat positif atau negatif.
        <br><br>
        <b>Support Vector Machine (SVM)</b> adalah sebuah metode klasifikasi yang digunakan dalam pembelajaran mesin (supervised learning) untuk memprediksi kategori berdasarkan model atau pola yang diperoleh dari proses pelatihan. 
        <br><br>
        <b>Seleksi Fitur</b> adalah proses memilih fitur yang paling relevan dari data untuk digunakan dalam model pembelajaran mesin. 
        Metode seleksi fitur yang umum digunakan meliputi Information Gain dan Chi-Square. 
        </div>
    """, unsafe_allow_html=True)

# Dataset
elif selected == "Dataset":
    st.subheader("Dataset")
    st.write("""
        <div style="text-align:justify;">
        Dataset yang digunakan merupakan data ulasan dari 15 tempat pariwisata di Kabupaten Jombang yang ada pada google maps dengan menggunakan teknik scraping. Setelah pengumpulan data, langkah selanjutnya adalah melakukan penyaringan data. Selanjutnya tahap pelabelan dilakukan secara manual dengan berdasarkan sentimen yang terkandung dalam ulasan yaitu positif dan negatif. Data dilabeli oleh 2 orang volunteer dan akan divalidasi oleh seorang guru bahasa Indonesia.
        <br><br>
        <b>Tahapan Preprocessing</b><br>
        1. Case Folding : Mengubah huruf besar ke huruf kecil (lowercase).<br>2. Cleansing : Menghilangkan karakter atau elemen yang tidak relevan atau seperti mencakup penghapusan tautan, tanda baca yang tidak diperlukan, dan lainnya. 
        <br>3. Normalisasi : Mengubah bahasa yang tidak baku menjadi bahasa yang baku sesuai Kamus Besar Bahasa Indoneisa (KBBI)<br>4. Tokenizing : Memisahan atau membagi teks berupa kalimat pada dokumen menjadi token atau term.
        <br>5. Stopword Removal : Menghapus kata-kata sesuai dengan kata-kata yang terdapat dalam stopword. <br>6. Stemming : mengubah kata menjadi bentuk kata dasar.
        </div>
    """, unsafe_allow_html=True)
    # Load datasets
    df_raw = pd.read_csv('dataset.csv')
    df_preprocessed = pd.read_csv('preproessing.csv')
    
    st.write("### Dataset Sebelum Preprocessing")
    st.write(df_raw)
    
    st.write("### Dataset Setelah Preprocessing")
    st.write(df_preprocessed)

# Prediksi Teks Tunggal
elif selected == "Prediksi Teks Tunggal":
    st.subheader("Prediksi Teks Tunggal")
    user_input = st.text_area("Masukkan teks untuk prediksi sentimen:")
    feature_selection_method = st.selectbox("Pilih Metode Seleksi Fitur", ["Information Gain", "Chi-Square", "Kombinasi Seleksi Fitur"])

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
    feature_selection_method = st.selectbox("Pilih Metode Seleksi Fitur", ["Information Gain", "Chi-Square", "Kombinasi Seleksi Fitur"])
    
    if st.button("Prediksi Sentimen") and uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        if 'text' not in df_uploaded.columns:
            st.error("File CSV yang diunggah harus berisi kolom 'text'.")
        else:
            df_uploaded['cleaned_text'] = df_uploaded['text'].apply(preprocess_text)
            
            if feature_selection_method == "Information Gain":
                resampled_df = resampled_df_ig
            elif feature_selection_method == "Chi-Square":
                resampled_df = resampled_df_chi
            else:
                resampled_df = resampled_df_selected
            
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

# Grafik
elif selected == "Grafik":
    
    st.write("### Grafik Information Gain")
    st.image("information_gain.jpg")
    st.write("""
        <div style="text-align:justify;">
        Pada skenario ini dilakukan pengujian untuk mencari akurasi dan waktu. pengujian dilakukan menggunakan SMOTE Selanjutnya dilakukan seleksi fitur information gain. Selanjutnya dilakukan pemodelan Support Vector menggunakan best hyperparameter. Pada pengujian ini menggunakan variasi jumlah fitur sebesar 95%, 90%, 85% 80%, 75%, 70%, 65%, 60%, 55%, 50%, 45%, 40 %, 35%, 30%, 25%, 20%, 15%, 10%, 5% dari keseluruhan fitur yang ada.
        <br>Akurasi dan waktu terbaik terdapat di variasi jumlah fitur sebesar 50% dengan akurasi 95% dan total waktu 57,59 detik.
        </div>
    """, unsafe_allow_html=True)
    
    st.write("### Grafik Chi-Square")
    st.image("chi_square.jpg")
    st.write("""
        <div style="text-align:justify;">
        Pada skenario ini dilakukan pengujian untuk mencari akurasi dan waktu. pengujian dilakukan menggunakan SMOTE Selanjutnya dilakukan seleksi fitur Chi Square. Selanjutnya dilakukan pemodelan Support Vector menggunakan best hyperparameter. Pada pengujian ini menggunakan variasi jumlah fitur sebesar 95%, 90%, 85% 80%, 75%, 70%, 65%, 60%, 55%, 50%, 45%, 40 %, 35%, 30%, 25%, 20%, 15%, 10%, 5% dari keseluruhan fitur yang ada.
        <br>Pada gambar diatas akurasi dan waktu terbaik terdapat di variasi jumlah fitur sebesar 20% dengan akurasi 95,43% dan total waktu 21,15 detik.
        </div>
    """, unsafe_allow_html=True)
    
    st.write("### Grafik Kombinasi Seleksi Fitur")
    st.image("kombinasi_seleksi_fitur.jpg")
    st.write("""
        <div style="text-align:justify;">
        Pada skenario ini dilakukan pengujian untuk mencari akurasi dan waktu. pengujian dilakukan menggunakan SMOTE Selanjutnya dilakukan pemodelan Support Vector dengan kombinasi seleksi fitur information gain dan Chi Square. Pada pengujian ini akan dilakukan kombinasi dengan menggunakan variasi jumlah fitur sebesar 95%, 90%, 85% 80%, 75%, 70%, 65%, 60%, 55%, 50%, 45%, 40 %, 35%, 30%, 25%, 20%, 15%, 10%, 5% dari keseluruhan fitur yang ada
        <br>Pada gambar diatas akurasi dan waktu terbaik terdapat di variasi jumlah fitur sebesar 60% dengan akurasi 95,02% dan total waktu 47,66 detik.
        </div>
    """, unsafe_allow_html=True)
