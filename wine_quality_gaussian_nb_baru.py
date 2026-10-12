# ============================================================
# MACHINE LEARNING - WINE QUALITY
# Klasifikasi dengan Gaussian Naive Bayes
# Dataset: Wine Quality (Red + White)
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. MEMBACA DATASET LOKAL
red = pd.read_csv("winequality-red.csv", sep=";")
white = pd.read_csv("winequality-white.csv", sep=";")

red["wine_type"] = "red"
white["wine_type"] = "white"
df = pd.concat([red, white], ignore_index=True)

print("Ukuran dataset awal:", df.shape)
print("\n5 data pertama:")
print(df.head())

# 2. INFORMASI DATA
print("\n================ INFO DATA ================")
df.info()
print("\n================ TIPE DATA ================")
print(df.dtypes)
print("\n================ MISSING VALUE ================")
print(df.isnull().sum())
print("\nJumlah baris duplikat:", df.duplicated().sum())

# 3. PEMERIKSAAN DAN PEMBERSIHAN KATEGORI
print("\n================ KATEGORI WINE TYPE ================")
print(df["wine_type"].value_counts())

df["wine_type"] = df["wine_type"].astype(str).str.strip().str.lower()
print("\nKategori setelah dibersihkan:")
print(df["wine_type"].unique())

# 4. HAPUS DUPLIKAT
jumlah_duplikat = df.duplicated().sum()
if jumlah_duplikat > 0:
    df = df.drop_duplicates().reset_index(drop=True)
print("\nJumlah duplikat yang dihapus:", jumlah_duplikat)
print("Ukuran dataset setelah menghapus duplikat:", df.shape)

# 5. MEMBUAT TARGET KLASIFIKASI
# Quality 3-5 = Kurang Baik
# Quality 6-9 = Baik
# Kolom quality asli tidak digunakan sebagai fitur.
df["quality_class"] = np.where(df["quality"] >= 6, "Baik", "Kurang Baik")

# 6. ANALISIS KETIDAKSEIMBANGAN KELAS
print("\n========== ANALISIS KETIDAKSEIMBANGAN KELAS ==========")

class_distribution = df["quality_class"].value_counts()
class_percentage = df["quality_class"].value_counts(normalize=True) * 100

print("Jumlah setiap kelas:")
print(class_distribution)

print("\nPersentase setiap kelas:")
print(class_percentage.round(2))

print("\n================ DISTRIBUSI TARGET ================")
print(df["quality_class"].value_counts())

# 7. MENENTUKAN FITUR DAN TARGET
feature_columns = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH", "sulphates",
    "alcohol", "wine_type"
]

X = df[feature_columns]
y = df["quality_class"]

print("\nJumlah fitur yang digunakan:", len(feature_columns))
print("Daftar fitur:")
for i, feature in enumerate(feature_columns, start=1):
    print(f"{i}. {feature}")

# 8. 5 DATA TERATAS DAN TERBAWAH SETELAH PEMBERSIHAN
print("\n================ 5 BARIS TERATAS ================")
print(df[feature_columns + ["quality_class"]].head())
print("\n================ 5 BARIS TERBAWAH ================")
print(df[feature_columns + ["quality_class"]].tail())

# 9. TRAINING DAN TESTING
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\n================ PEMBAGIAN DATA ================")
print("Jumlah data training:", X_train.shape[0])
print("Jumlah data testing :", X_test.shape[0])

# 10. FITUR NUMERIK DAN KATEGORIKAL
numeric_features = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
]
categorical_features = ["wine_type"]

# 11. PREPROCESSING
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ============================================================
# HASIL STANDARDISASI DATA TRAINING DAN TESTING
# ============================================================

# Fit preprocessing hanya menggunakan data training
preprocessor.fit(X_train)

# Ambil preprocessing untuk fitur numerik
numeric_preprocessor = preprocessor.named_transformers_["num"]

# Transform data numerik training dan testing
X_train_numeric_scaled = numeric_preprocessor.transform(
    X_train[numeric_features]
)

X_test_numeric_scaled = numeric_preprocessor.transform(
    X_test[numeric_features]
)

# Ubah hasil standardisasi menjadi DataFrame
X_train_scaled_df = pd.DataFrame(
    X_train_numeric_scaled,
    columns=numeric_features,
    index=X_train.index
)

X_test_scaled_df = pd.DataFrame(
    X_test_numeric_scaled,
    columns=numeric_features,
    index=X_test.index
)

# ------------------------------------------------------------
# TAMPILKAN DATA SEBELUM STANDARDISASI
# ------------------------------------------------------------

print("\n========== 5 DATA TRAINING SEBELUM STANDARDISASI ==========")
print(X_train[numeric_features].head())

print("\n========== 5 DATA TESTING SEBELUM STANDARDISASI ==========")
print(X_test[numeric_features].head())

# ------------------------------------------------------------
# TAMPILKAN DATA SETELAH STANDARDISASI
# ------------------------------------------------------------

print("\n========== 5 DATA TRAINING SETELAH STANDARDISASI ==========")
print(X_train_scaled_df.head())

print("\n========== 5 DATA TESTING SETELAH STANDARDISASI ==========")
print(X_test_scaled_df.head())

# 12. MODEL GAUSSIAN NAIVE BAYES
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GaussianNB())
])

# 13. TRAINING
model.fit(X_train, y_train)
print("\nModel Gaussian Naive Bayes berhasil dilatih.")

# 14. PREDIKSI
# Preprocessing otomatis diterapkan dengan aturan yang dipelajari dari data training.
y_pred = model.predict(X_test)

# 15. EVALUASI
accuracy = accuracy_score(y_test, y_pred)
print("\n================ HASIL EVALUASI ================")
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("""
\n============================================================
CATATAN PREPROCESSING
============================================================
Model utama menggunakan STANDARDISASI dengan StandardScaler.

Minmadisasi:
- Rata-rata data menjadi sekitar 0
- Standar deviasi menjadi sekitar 1
============================================================
""")
