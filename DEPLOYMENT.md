# 🚀 Panduan Deployment Aplikasi Deteksi Insomnia

## 📋 Platform Deployment yang Didukung

### 1. Streamlit Cloud (Direkomendasikan)

**Langkah-langkah:**
1. Upload repository ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Login dengan GitHub
4. Pilih repository dan file `app.py`
5. Klik "Deploy"

**Keuntungan:**
- Gratis untuk aplikasi publik
- Deploy otomatis
- Mudah digunakan
- Terintegrasi dengan GitHub

### 2. Heroku

**Langkah-langkah:**
1. Install Heroku CLI
2. Login ke Heroku: `heroku login`
3. Buat aplikasi: `heroku create nama-aplikasi`
4. Deploy: `git push heroku main`

**File yang diperlukan:**
- `Procfile` ✅
- `requirements.txt` ✅
- `runtime.txt` ✅

### 3. Railway

**Langkah-langkah:**
1. Buka [railway.app](https://railway.app)
2. Connect dengan GitHub
3. Pilih repository
4. Deploy otomatis

### 4. Local Server

**Langkah-langkah:**
```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

## 🔧 Konfigurasi

### Environment Variables (Opsional)
```bash
# Untuk production
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### File Konfigurasi
- `.streamlit/config.toml` - Konfigurasi tema dan server
- `requirements.txt` - Dependencies Python
- `runtime.txt` - Versi Python

## 📊 Monitoring

### Logs
- Streamlit Cloud: Dashboard built-in
- Heroku: `heroku logs --tail`
- Railway: Dashboard built-in

### Performance
- Monitor penggunaan memory
- Cek response time
- Optimasi jika diperlukan

## 🛠️ Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Pastikan file `rforest_model.joblib` ada
   - Cek path file

2. **Dependencies Error**
   - Update `requirements.txt`
   - Cek versi Python

3. **Memory Issues**
   - Optimasi model size
   - Gunakan caching

### Support
- Streamlit Documentation: [docs.streamlit.io](https://docs.streamlit.io)
- Heroku Documentation: [devcenter.heroku.com](https://devcenter.heroku.com)

---

**Aplikasi siap untuk deployment! 🎉** 