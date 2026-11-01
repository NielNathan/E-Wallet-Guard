import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import os

# cek dulu file datasetnya ada apa engga
if not os.path.exists('paysim.csv'):
    print("ERROR: File 'paysim.csv' tidak ditemukan!")
    exit(1)

# Membaca dataset 
df = pd.read_csv('paysim.csv')
print(f"Dataset dimuat: {len(df):,} baris")

# ambil yg payment sama transfer aja, soalnya yg relevan cuma itu
df_fraud = df[(df['type'] == 'PAYMENT') | (df['type'] == 'TRANSFER')].copy()

# ubah kolom type jadi angka pake get_dummies
df_fraud = pd.get_dummies(df_fraud, columns=['type'], drop_first=True)

# bikin fitur-fitur baru (FEATURE ENGINEERING)
df_fraud['Frekuensi_Transaksi'] = df_fraud.groupby(['nameOrig', 'step'])['amount'].transform('count')
df_fraud['Jam_Aktual'] = df_fraud['step'] % 24
df_fraud['Is_Jam_Malam'] = df_fraud['Jam_Aktual'].apply(lambda x: 1 if x <= 5 else 0)

# hitung berapa uang yg beneran masuk ke rekening tujuan
df_fraud['Uang_Diterima'] = df_fraud['newbalanceDest'] - df_fraud['oldbalanceDest']
df_fraud['Anomali_Uang_Tujuan'] = df_fraud['amount'] - df_fraud['Uang_Diterima']

# kasih label fraud berdasarkan kondisi anomali
df_fraud['isFraud'] = 0
df_fraud.loc[df_fraud['Anomali_Uang_Tujuan'] > 0, 'isFraud'] = 1
df_fraud.loc[(df_fraud['Frekuensi_Transaksi'] >= 5) & (df_fraud['Is_Jam_Malam'] == 1), 'isFraud'] = 1

# susun urutan kolom secara eksplisit agar sama persis dengan input di Streamlit
features_list = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
    'oldbalanceDest', 'newbalanceDest', 'type_TRANSFER', 
    'Frekuensi_Transaksi', 'Jam_Aktual', 'Is_Jam_Malam', 'Anomali_Uang_Tujuan'
]

X = df_fraud[features_list]
y = df_fraud['isFraud']

print(f"Features: {len(features_list)} kolom")
print(f"Samples: {len(X):,} baris")

# split data 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {len(X_train):,} baris")
print(f"Testing set: {len(X_test):,} baris")

# pake SMOTE biar datanya seimbang antara fraud dan aman
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"Training set setelah SMOTE: {len(X_train_smote):,} baris")

# training pake random forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_smote, y_train_smote)

# cek hasilnya gimana
print("HASIL EVALUASI MODEL")
y_pred = rf_model.predict(X_test)
y_pred_train = rf_model.predict(X_train)

accuracy_test = accuracy_score(y_test, y_pred)
accuracy_train = accuracy_score(y_train, y_pred_train)

print(f"\nAkurasi Training Set: {accuracy_train*100:.2f}%")
print(f"Akurasi Testing Set: {accuracy_test*100:.2f}%")

# confusion matrix buat liat detail prediksinya
print("\nConfusion Matrix (Tabel Kebenaran):")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nInterpretasi:")
print(f"  - True Negative (Benar Aman): {cm[0][0]:,}")
print(f"  - False Positive (Salah Deteksi Fraud): {cm[0][1]:,}")
print(f"  - False Negative (Fraud Tidak Terdeteksi): {cm[1][0]:,}")
print(f"  - True Positive (Benar Fraud): {cm[1][1]:,}")

print("\nClassification Report (Metrik Validasi):")
print(classification_report(y_test, y_pred, target_names=['Aman', 'Fraud']))

# liat fitur mana yg paling berpengaruh
print("\nSignifikansi Fitur Menurut Model AI:")
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Fitur': features_list,
    'Importance': importances
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance_df.iterrows():
    print(f"   {row['Fitur']:.<30} {row['Importance']*100:>6.2f}%")

# simpen modelnya ke file pkl
model_filename = 'qris_fraud_model.pkl'
joblib.dump(rf_model, model_filename)
model_size = os.path.getsize(model_filename) / (1024*1024)  # ubah ke MB
print(f"Model berhasil disimpan: {model_filename}")
print(f"Ukuran file: {model_size:.2f} MB")
print(f"Selesai! Model siap digunakan di Streamlit.")