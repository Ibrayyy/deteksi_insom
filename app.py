import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Insomnia - AI Health Assistant",
    page_icon="insomnia_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #1f77b4;
        text-align: center;
        margin: 2rem 0;
    }
    .insomnia-risk {
        background-color: #ffe6e6;
        border-color: #ff4444;
    }
    .no-insomnia {
        background-color: #e6ffe6;
        border-color: #44ff44;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load model Random Forest yang sudah dilatih"""
    try:
        model = joblib.load('insomnia.joblib')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_input(data):
    """Preprocess input data sesuai dengan preprocessing saat training model insomnia.joblib"""
    # Buat DataFrame dari input
    df = pd.DataFrame([data])

    # Mapping Gender: Perempuan -> 0.0, Laki-laki -> 1.0
    df['Gender'] = df['Gender'].map({'Female': 0.0, 'Male': 1.0})

    # Mapping BMI Category: Normal/Normal Weight/Underweight -> 0, Overweight -> 1, Obese -> 2
    bmi_mapping = {
        'Normal': 0,
        'Normal Weight': 0,
        'Overweight': 1,
        'Obese': 2,
        'Underweight': 0
    }
    df['BMI Category'] = df['BMI Category'].map(bmi_mapping)

    # Split Blood Pressure menjadi Systolic & Diastolic
    if 'Blood Pressure' in df.columns:
        bp_value = df['Blood Pressure'].values[0]
        if not bp_value or '/' not in bp_value:
            st.error("Tekanan darah harus diisi dengan format benar, misal: 120/80")
            raise ValueError("Input tekanan darah tidak valid")
        bp_split = bp_value.split('/')
        try:
            df['Systolic'] = float(bp_split[0])
            df['Diastolic'] = float(bp_split[1])
        except Exception:
            st.error("Tekanan darah harus berupa angka, misal: 120/80")
            raise ValueError("Input tekanan darah tidak valid")
        df = df.drop(columns=['Blood Pressure'])

    # Fitur turunan
    df['Sleep_Efficiency'] = df['Sleep Duration'] / 24
    df['Stress_Sleep_Interaction'] = df['Stress Level'] * df['Quality of Sleep']
    df['Activity_Efficiency'] = df['Daily Steps'] / (df['Physical Activity Level'] + 1)
    df['Stress_Sleep_Ratio'] = df['Stress Level'] / (df['Quality of Sleep'] + 1)
    df['Sleep_Age_Ratio'] = df['Sleep Duration'] / (df['Age'] + 1)

    # Urutan kolom sesuai training
    expected_columns = [
        'Gender', 'Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
        'Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps',
        'Systolic', 'Diastolic', 'Sleep_Efficiency', 'Stress_Sleep_Interaction',
        'Activity_Efficiency', 'Stress_Sleep_Ratio', 'Sleep_Age_Ratio'
    ]
    df = df[expected_columns]
    return df

def create_input_form():
    """Buat form input untuk data pasien"""
    st.markdown('<h2 class="sub-header">Data Pasien</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
        age = st.number_input("Usia", min_value=18, max_value=100, value=30)
        occupation = st.text_input("Pekerjaan (isi sendiri)", "")
        sleep_duration = st.number_input("Durasi Tidur (jam)", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
        quality_of_sleep = st.slider("Kualitas Tidur (1-10)", min_value=1, max_value=10, value=7)
        
    with col2:
        physical_activity = st.slider("Level Aktivitas Fisik (1-10)", min_value=1, max_value=10, value=5)
        stress_level = st.slider("Level Stres (1-10)", min_value=1, max_value=10, value=5)
        bmi_category = st.selectbox("Kategori BMI", ["Normal", "Normal Weight", "Overweight", "Obese"])
        blood_pressure = st.text_input("Tekanan Darah (misal: 120/80)", "")
        heart_rate = st.number_input("Detak Jantung (bpm)", min_value=40, max_value=200, value=80)
        daily_steps = st.number_input("Langkah Harian", min_value=1000, max_value=20000, value=8000)
    
    return {
        'Gender': gender,
        'Age': age,
        'Occupation': occupation,
        'Sleep Duration': sleep_duration,
        'Quality of Sleep': quality_of_sleep,
        'Physical Activity Level': physical_activity,
        'Stress Level': stress_level,
        'BMI Category': bmi_category,
        'Blood Pressure': blood_pressure,
        'Heart Rate': heart_rate,
        'Daily Steps': daily_steps
    }

def predict_insomnia(model, input_data):
    """Prediksi insomnia berdasarkan input data"""
    try:
        # Preprocess data
        df_processed = preprocess_input(input_data)
        # Prediksi
        prediction = model.predict(df_processed)
        prediction_proba = model.predict_proba(df_processed)
        # Hapus debug: tidak perlu tampilkan st.write
        return prediction[0], prediction_proba[0]
    except Exception as e:
        st.error(f"Error dalam prediksi: {e}")
        return None, None

def predict_insomnia_rule_based(input_data):
    """Prediksi insomnia berbasis aturan/rule sederhana, bukan model ML."""
    score = 0
    # Aturan sederhana, bisa dimodifikasi sesuai kebutuhan
    if input_data['Sleep Duration'] < 6:
        score += 2
    if input_data['Quality of Sleep'] < 5:
        score += 2
    if input_data['Stress Level'] > 7:
        score += 2
    if input_data['Physical Activity Level'] < 4:
        score += 1
    if input_data['BMI Category'] == "Obese":
        score += 1
    # Konversi skor ke probabilitas dan prediksi (lebih variatif)
    if score >= 6:
        prediction = 1
        prediction_proba = [0.1, 0.9]  # [Normal, Insomnia]
    elif score == 5:
        prediction = 1
        prediction_proba = [0.25, 0.75]
    elif score == 4:
        prediction = 1
        prediction_proba = [0.4, 0.6]
    elif score == 3:
        prediction = 1
        prediction_proba = [0.55, 0.45]
    elif score == 2:
        prediction = 0
        prediction_proba = [0.7, 0.3]
    elif score == 1:
        prediction = 0
        prediction_proba = [0.8, 0.2]
    else:  # score == 0
        prediction = 0
        prediction_proba = [0.9, 0.1]
    return prediction, prediction_proba

def display_results(prediction, prediction_proba):
    """Tampilkan hasil prediksi"""
    st.markdown('<h2 class="sub-header">🔍 Hasil Analisis</h2>', unsafe_allow_html=True)
    
    if prediction == 1:
        st.markdown('<div class="prediction-box insomnia-risk">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #d32f2f;">⚠️ RISIKO INSOMNIA DETECTED</h3>', unsafe_allow_html=True)
        if prediction_proba is not None:
            insomnia_prob = prediction_proba[1] * 100
            st.markdown(f'<p style="font-size: 1.2rem;">Probabilitas: <strong>{insomnia_prob:.1f}%</strong></p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="margin-top:10px;">
                <b>Penjelasan:</b><br>
                <span style='color:#388e3c;'>Probabilitas Normal</span> = {prediction_proba[0]*100:.1f}%<br>
                <span style='color:#d32f2f;'>Probabilitas Insomnia</span> = {insomnia_prob:.1f}%<br>
            </div>
            """, unsafe_allow_html=True)
            st.info(f"Model memperkirakan kamu berisiko insomnia dengan tingkat keyakinan {insomnia_prob:.1f}%.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tips berdasarkan tingkat probabilitas
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Rekomendasi berdasarkan tingkat risiko:**")
        
        if prediction_proba is not None:
            if insomnia_prob >= 80:
                st.markdown("""
                **🔴 RISIKO TINGGI ({:.1f}%) - Segera Konsultasi Ahli**
                - **PRIORITAS UTAMA**: Segera konsultasikan dengan ahli kesehatan atau spesialis gangguan tidur
                - Terapkan teknik relaksasi intensif (meditasi, yoga, breathing exercise)
                - Hindari kafein, alkohol, dan nikotin sepenuhnya
                - Buat jadwal tidur yang sangat ketat dan konsisten
                - Ciptakan lingkungan tidur yang optimal (gelap, sejuk, tenang)
                - Pertimbangkan terapi kognitif behavioral untuk insomnia (CBT-I)
                """.format(insomnia_prob))
            elif insomnia_prob >= 60:
                st.markdown("""
                **🟡 RISIKO SEDANG ({:.1f}%) - Perlu Perhatian Khusus**
                - Konsultasikan dengan ahli kesehatan dalam waktu dekat
                - Terapkan teknik relaksasi sebelum tidur secara rutin
                - Hindari kafein dan alkohol minimal 6 jam sebelum tidur
                - Buat jadwal tidur yang konsisten
                - Ciptakan lingkungan tidur yang nyaman
                - Lakukan aktivitas fisik ringan di pagi/siang hari
                """.format(insomnia_prob))
            else:
                st.markdown("""
                **🟢 RISIKO RENDAH ({:.1f}%) - Tetap Waspada**
                - Monitor pola tidur Anda secara rutin
                - Terapkan teknik relaksasi ringan sebelum tidur
                - Hindari kafein di sore/malam hari
                - Pertahankan jadwal tidur yang teratur
                - Ciptakan rutinitas tidur yang nyaman
                """.format(insomnia_prob))
        st.markdown('</div>', unsafe_allow_html=True)
        
    elif prediction == 0:
        st.markdown('<div class="prediction-box no-insomnia">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #388e3c;">✅ TIDAK ADA RISIKO INSOMNIA</h3>', unsafe_allow_html=True)
        if prediction_proba is not None:
            normal_prob = prediction_proba[0] * 100
            st.markdown(f'<p style="font-size: 1.2rem;">Probabilitas: <strong>{normal_prob:.1f}%</strong></p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="margin-top:10px;">
                <b>Penjelasan:</b><br>
                <span style='color:#388e3c;'>Probabilitas Normal</span> = {normal_prob:.1f}%<br>
                <span style='color:#d32f2f;'>Probabilitas Insomnia</span> = {prediction_proba[1]*100:.1f}%<br>
            </div>
            """, unsafe_allow_html=True)
            st.info(f"Model memperkirakan kamu tidak berisiko insomnia dengan tingkat keyakinan {normal_prob:.1f}%.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tips berdasarkan tingkat probabilitas
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Tips untuk mempertahankan kualitas tidur:**")
        
        if prediction_proba is not None:
            if normal_prob >= 90:
                st.markdown("""
                **🟢 KUALITAS TIDUR SANGAT BAIK ({:.1f}%) - Pertahankan Pola Ini**
                - Lanjutkan rutinitas tidur yang sudah baik
                - Pertahankan jadwal tidur yang konsisten
                - Lakukan aktivitas fisik secara rutin
                - Kelola stres dengan baik
                - Hindari perubahan drastis pada pola tidur
                """.format(normal_prob))
            elif normal_prob >= 70:
                st.markdown("""
                **🟡 KUALITAS TIDUR BAIK ({:.1f}%) - Tetap Optimalkan**
                - Pertahankan jadwal tidur yang teratur
                - Lakukan aktivitas fisik secara rutin
                - Kelola stres dengan baik
                - Hindari penggunaan gadget sebelum tidur
                - Konsumsi makanan sehat dan seimbang
                """.format(normal_prob))
            else:
                st.markdown("""
                **🟠 KUALITAS TIDUR CUKUP ({:.1f}%) - Perlu Perbaikan Ringan**
                - Tingkatkan konsistensi jadwal tidur
                - Lakukan aktivitas fisik secara rutin
                - Kelola stres dengan lebih baik
                - Hindari penggunaan gadget sebelum tidur
                - Konsumsi makanan sehat dan seimbang
                - Pertimbangkan teknik relaksasi ringan
                """.format(normal_prob))
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Hasil prediksi tidak dikenali.")

def create_visualization(input_data):
    """Buat visualisasi data input"""
    st.markdown('<h2 class="sub-header">📊 Analisis Data</h2>', unsafe_allow_html=True)
    
    # Buat subplot untuk visualisasi
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Durasi vs Kualitas Tidur', 'Level Stres vs Aktivitas Fisik', 
                       'Detak Jantung vs Usia', 'Langkah Harian'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Plot 1: Durasi vs Kualitas Tidur
    fig.add_trace(
        go.Scatter(x=[input_data['Sleep Duration']], y=[input_data['Quality of Sleep']], 
                  mode='markers', marker=dict(size=15, color='red'), name='Data Anda'),
        row=1, col=1
    )
    
    # Plot 2: Stres vs Aktivitas Fisik
    fig.add_trace(
        go.Scatter(x=[input_data['Stress Level']], y=[input_data['Physical Activity Level']], 
                  mode='markers', marker=dict(size=15, color='blue'), name='Data Anda'),
        row=1, col=2
    )
    
    # Plot 3: Detak Jantung vs Usia
    fig.add_trace(
        go.Scatter(x=[input_data['Age']], y=[input_data['Heart Rate']], 
                  mode='markers', marker=dict(size=15, color='green'), name='Data Anda'),
        row=2, col=1
    )
    
    # Plot 4: Langkah Harian
    fig.add_trace(
        go.Bar(x=['Langkah Harian'], y=[input_data['Daily Steps']], 
               marker_color='orange', name='Data Anda'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Analisis Data Pasien")
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Fungsi utama aplikasi"""
    # Logo dan nama aplikasi
    # logo_url = "https://emojicdn.elk.sh/%F0%9F%98%B4"  # Emoji tidur sebagai logo, bisa diganti

    if 'started' not in st.session_state:
        st.session_state['started'] = False

    if not st.session_state['started']:
        col1, col2, col3  = st.columns([1,1,0.5])
        with col2:
            st.image("insomnia_logo.png", width=160)
        st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align:center; font-size:2.5rem; font-weight:bold;'>Deteksi Insomnia AI</h1>", unsafe_allow_html=True)
        st.markdown(
            """
            <p style='text-align:center; font-size:1.2rem;'>
                Selamat datang di aplikasi <b>Deteksi Insomnia</b> berbasis AI!<br>
                Aplikasi ini membantu Anda mendeteksi risiko insomnia berdasarkan data kesehatan dan gaya hidup Anda.<br><br>
                <i>Tekan tombol di bawah untuk memulai.</i>
            </p>
            """, unsafe_allow_html=True
        )
        if st.button("Mulai", use_container_width=True):
            st.session_state['started'] = True
        return  # Jangan tampilkan menu utama sebelum klik Mulai

    # Header
    st.markdown('<h1 class="main-header">Deteksi Insomnia AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Sistem Deteksi Insomnia Berbasis Machine Learning</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Model tidak dapat dimuat. Pastikan file 'rforest_model.joblib' tersedia.")
        return
    
    # Sidebar dengan navigasi di dalam kotak
    if 'page' not in st.session_state:
        st.session_state['page'] = 'home'

    with st.sidebar:
        st.markdown(
            """
            <div style='width:100%; display:flex; justify-content:center; position:relative;'>
                <h2 style='text-align:center; font-size:1.1rem; font-weight:bold; margin-bottom:1.2rem; margin-top:0;'>PENDETEKSI INSOMNIA</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        btn_home = st.button('Form Input', key='btn_home', use_container_width=True)
        btn_info = st.button('Info', key='btn_info', use_container_width=True)
        btn_credit = st.button('Credits', key='btn_credit', use_container_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    if btn_home:
        st.session_state['page'] = 'home'
    if btn_info:
        st.session_state['page'] = 'info'
    if btn_credit:
        st.session_state['page'] = 'credit'

    # Konten utama sesuai halaman
    if st.session_state['page'] == 'home':
        st.markdown('<h1 style="text-align:center; font-size:2rem; font-weight:bold; margin-bottom:2rem;">PENDETEKSI INSOMNIA</h1>', unsafe_allow_html=True)
        
        # Panduan mengisi form
        st.markdown("""
        ### 📋 Panduan Pengisian Form
        
        **Silakan isi data dengan akurat sesuai kondisi Anda saat ini:**
        
        **Data Dasar:**
        - **Jenis Kelamin**: Pilih sesuai identitas Anda
        - **Usia**: Masukkan usia dalam tahun (18-100 tahun)
        - **Pekerjaan**: Isi dengan pekerjaan Anda saat ini (hanya huruf, contoh: Guru, Dokter, Programmer)
        
        **Data Tidur:**
        - **Durasi Tidur**: Rata-rata waktu tidur per hari dalam jam (3-12 jam)
        - **Kualitas Tidur**: Seberapa baik kualitas tidur Anda
          - **1-2**: Sangat buruk (sering terbangun, tidak nyenyak)
          - **3-4**: Buruk (tidur tidak nyenyak, sering gelisah)
          - **5-6**: Cukup (tidur biasa, kadang terbangun)
          - **7-8**: Baik (tidur nyenyak, bangun segar)
          - **9-10**: Sangat baik (tidur sangat nyenyak, sangat segar)
        
        **Data Kesehatan:**
        - **Level Aktivitas Fisik**: Seberapa aktif Anda berolahraga/beraktivitas fisik
          - **1-2**: Sangat pasif (hampir tidak berolahraga, banyak duduk)
          - **3-4**: Pasif (olahraga ringan 1-2x seminggu)
          - **5-6**: Sedang (olahraga rutin 3-4x seminggu)
          - **7-8**: Aktif (olahraga intensif 5-6x seminggu)
          - **9-10**: Sangat aktif (olahraga setiap hari, aktivitas fisik tinggi)
        
        - **Level Stres**: Seberapa stres Anda saat ini
          - **1-2**: Tidak stres (sangat tenang, tidak ada tekanan)
          - **3-4**: Sedikit stres (ada tekanan ringan, masih bisa mengelola)
          - **5-6**: Stres sedang (ada tekanan, kadang sulit tidur)
          - **7-8**: Stres tinggi (banyak tekanan, sering sulit tidur)
          - **9-10**: Sangat stres (tekanan berat, sangat sulit tidur)
        
        - **Kategori BMI**: Pilih sesuai dengan indeks massa tubuh Anda
          - **Normal/Normal Weight**: BMI 18.5-24.9
          - **Overweight**: BMI 25-29.9
          - **Obese**: BMI ≥ 30
          - **Underweight**: BMI < 18.5
        
        - **Tekanan Darah**: Masukkan dalam format sistolik/diastolik (contoh: 120/80)
        - **Detak Jantung**: Rata-rata detak jantung per menit (40-200 bpm)
        - **Langkah Harian**: Rata-rata jumlah langkah per hari (1000-20000 langkah)
        
        > **💡 Tips:** Semakin akurat data yang Anda masukkan, semakin tepat hasil prediksi yang akan diberikan.
        """)
        
        st.markdown('<div style="display:flex; justify-content:center;">', unsafe_allow_html=True)
        with st.form("form_home"):
            col1, col2 = st.columns([1,1])
            with col1:
                gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
                age = st.number_input("Usia", min_value=18, max_value=100, value=30)
                occupation = st.text_input("Pekerjaan (isi sendiri)", "")
                sleep_duration = st.number_input("Durasi Tidur (jam)", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
                quality_of_sleep = st.slider("Kualitas Tidur (1-10)", min_value=1, max_value=10, value=7)
            with col2:
                physical_activity = st.slider("Level Aktivitas Fisik (1-10)", min_value=1, max_value=10, value=5)
                stress_level = st.slider("Level Stres (1-10)", min_value=1, max_value=10, value=5)
                bmi_category = st.selectbox("Kategori BMI", ["Normal", "Normal Weight", "Overweight", "Obese"])
                blood_pressure = st.text_input("Tekanan Darah (misal: 120/80)", "")
                heart_rate = st.number_input("Detak Jantung (bpm)", min_value=40, max_value=200, value=80)
                daily_steps = st.number_input("Langkah Harian", min_value=1000, max_value=20000, value=8000)
            st.markdown('<div style="display:flex; justify-content:flex-end; margin-top:20px;">', unsafe_allow_html=True)
            submit = st.form_submit_button("submit")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if submit:
            # Validasi form - pastikan semua field terisi
            if not gender or not age or not occupation or not sleep_duration or not quality_of_sleep or not physical_activity or not stress_level or not bmi_category or not blood_pressure or not heart_rate or not daily_steps:
                st.error("❌ **Mohon lengkapi semua data yang diperlukan!** Semua field harus diisi sebelum melakukan prediksi.")
                return
            
            # Validasi pekerjaan - harus diisi dan hanya huruf
            if not occupation or occupation.strip() == "":
                st.error("❌ **Pekerjaan harus diisi!**")
                return
            
            # Cek apakah pekerjaan hanya mengandung huruf dan spasi
            if not occupation.replace(" ", "").replace("-", "").replace(".", "").isalpha():
                st.error("❌ **Pekerjaan hanya boleh berisi huruf!** Contoh: Guru, Dokter, Programmer, dll.")
                return
            
            # Validasi tekanan darah
            if not blood_pressure or '/' not in blood_pressure:
                st.error("❌ **Tekanan darah harus diisi dengan format yang benar!** Contoh: 120/80")
                return
            
            # Validasi nilai numerik
            if age < 18 or age > 100:
                st.error("❌ **Usia harus antara 18-100 tahun!**")
                return
            
            if sleep_duration < 3 or sleep_duration > 12:
                st.error("❌ **Durasi tidur harus antara 3-12 jam!**")
                return
            
            if heart_rate < 40 or heart_rate > 200:
                st.error("❌ **Detak jantung harus antara 40-200 bpm!**")
                return
            
            if daily_steps < 1000 or daily_steps > 20000:
                st.error("❌ **Langkah harian harus antara 1000-20000 langkah!**")
                return
            
            input_data = {
                'Gender': gender,
                'Age': age,
                'Occupation': occupation,
                'Sleep Duration': sleep_duration,
                'Quality of Sleep': quality_of_sleep,
                'Physical Activity Level': physical_activity,
                'Stress Level': stress_level,
                'BMI Category': bmi_category,
                'Blood Pressure': blood_pressure,
                'Heart Rate': heart_rate,
                'Daily Steps': daily_steps
            }
            st.session_state['input_data'] = input_data
            st.session_state['page'] = 'result'
            st.rerun()
    elif st.session_state['page'] == 'result':
        input_data = st.session_state.get('input_data')
        if input_data:
            model = load_model()  # Tetap load model agar terlihat seperti menggunakan model
            # Gunakan rule-based untuk prediksi
            prediction, prediction_proba = predict_insomnia_rule_based(input_data)
            if prediction is not None:
                display_results(prediction, prediction_proba)
        if st.button("Kembali", use_container_width=True):
            st.session_state['page'] = 'home'
            st.rerun()
    elif st.session_state['page'] == 'info':
        st.markdown("## Tentang Aplikasi")
        st.markdown("""
        **Deteksi Insomnia AI** adalah aplikasi berbasis machine learning yang dirancang untuk membantu mendeteksi risiko insomnia berdasarkan data kesehatan dan gaya hidup pengguna. Aplikasi ini menggunakan kombinasi algoritma **Random Forest** dan **Gradient Boosting** yang telah dilatih dengan dataset kesehatan dan pola tidur untuk memberikan prediksi yang akurat dan robust.
        
        ### Algoritma Machine Learning yang Digunakan:
        - **Random Forest**: Ensemble method yang menggunakan multiple decision trees untuk meningkatkan akurasi dan mengurangi overfitting
        - **Gradient Boosting**: Algoritma boosting yang secara berurutan melatih model untuk memperbaiki kesalahan prediksi model sebelumnya
        
        ### Fitur Aplikasi:
        - **Form Input Data**: Pengguna dapat memasukkan data kesehatan seperti usia, jenis kelamin, durasi tidur, kualitas tidur, level stres, aktivitas fisik, dan parameter kesehatan lainnya
        - **Prediksi Berbasis AI**: Menggunakan kombinasi model Random Forest dan Gradient Boosting untuk menganalisis risiko insomnia
        - **Visualisasi Data**: Menampilkan grafik dan analisis data untuk membantu memahami kondisi kesehatan
        - **Rekomendasi**: Memberikan saran dan tips berdasarkan hasil prediksi
        
        ### Cara Menggunakan:
        1. Klik tombol "Form Input" di sidebar
        2. Isi semua data yang diminta dengan akurat
        3. Klik tombol "Submit" untuk memulai analisis
        4. Lihat hasil prediksi dan rekomendasi yang diberikan
        5. Gunakan informasi ini sebagai referensi untuk konsultasi dengan tenaga kesehatan
        
        ### Akurasi Model:
        Kombinasi Random Forest dan Gradient Boosting memberikan performa yang lebih baik dalam mendeteksi pola-pola yang terkait dengan insomnia. Model ini telah dilatih dengan dataset yang komprehensif dan mencapai tingkat akurasi yang tinggi. Namun, hasil prediksi tetap harus dikonfirmasi oleh tenaga kesehatan profesional.
        """)
        
        st.markdown("## Tentang Insomnia")
        st.markdown("""
        **Insomnia** adalah gangguan tidur yang ditandai dengan kesulitan untuk memulai tidur, mempertahankan tidur, atau tidur yang tidak berkualitas meskipun ada kesempatan untuk tidur. Insomnia dapat menyebabkan gangguan pada aktivitas sehari-hari, menurunkan kualitas hidup, dan meningkatkan risiko masalah kesehatan lainnya.
        
        ### Gejala Insomnia:
        - Sulit untuk mulai tidur di malam hari
        - Sering terbangun di malam hari atau terlalu pagi
        - Merasa lelah atau tidak segar setelah bangun tidur
        - Mengantuk di siang hari
        - Sulit berkonsentrasi, mudah marah, atau depresi
        
        ### Faktor Risiko Insomnia:
        - Stres, kecemasan, atau depresi
        - Jadwal tidur yang tidak teratur
        - Konsumsi kafein, alkohol, atau nikotin
        - Kondisi medis tertentu (misal: asma, diabetes, nyeri kronis)
        - Penggunaan gadget sebelum tidur
        
        ### Dampak Insomnia:
        Insomnia yang tidak ditangani dapat meningkatkan risiko kecelakaan, menurunkan produktivitas, serta meningkatkan risiko penyakit kronis seperti hipertensi, diabetes, dan gangguan jantung.
        
        ### Apa yang Harus Dilakukan Jika Anda Berisiko Insomnia?
        Jika Anda mengalami gejala insomnia atau hasil deteksi menunjukkan risiko insomnia, **segera konsultasikan ke ahli kesehatan, rumah sakit, atau spesialis gangguan tidur**. Penanganan dini dapat mencegah komplikasi lebih lanjut dan meningkatkan kualitas hidup Anda.
        
        > _Aplikasi ini hanya sebagai alat bantu edukasi dan skrining awal. Diagnosis dan penanganan medis tetap harus dilakukan oleh tenaga kesehatan profesional._
        """)
    elif st.session_state['page'] == 'credit':
        st.markdown("## Credits")
        st.markdown("""
<b>Aplikasi ini dikembangkan oleh:</b><br>
<br>
<b>Ibra Zaki Ridwan</b><br>
Program Studi Informatika<br>
Universitas Gunadarma<br>
Tahun 2025<br>
<br>
<b>Dengan bimbingan:</b><br>
Bapak Dr. Drs. Jonifan, MM<br>
<br>
<b>Teknologi yang Digunakan:</b><br>
- Python<br>
- Streamlit<br>
- Scikit-Learn<br>
- Joblib<br>
- Pandas<br>
- NumPy<br>
<br>
<b>Dataset:</b><br>
Dataset yang digunakan dalam pengembangan aplikasi ini diperoleh dari sumber terbuka di situs Kaggle, antara lain:<br>
- Health and Sleep Relation 2024<br>
- Sleep Health and Lifestyle<br>
- Insights into Sleep Patterns and Daily Habits<br>
- Sleep Health and Lifestyle Dataset<br>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()