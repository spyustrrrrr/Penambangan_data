import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# BAGIAN 1: MEMBUAT DATASET DUMMY
# ==============================================================================

print("--- [1] Membuat Dataset Dummy ---")

# Tentukan jumlah baris data yang ingin dibuat
JUMLAH_BARIS = 500  # Kita perbanyak datanya agar grafiknya lebih bagus

# --- 1.1. Membuat Data ---
list_kategori = ['Elektronik', 'Fashion', 'Rumah Tangga', 'Kesehatan', 'Olahraga']
list_wilayah = ['Jakarta', 'Bandung', 'Surabaya', 'Yogyakarta', 'Lainnya']
list_pembayaran = ['Kartu Kredit', 'Transfer Bank', 'E-Wallet', 'COD']

# Buat data untuk setiap kolom
data = {
    'ID_Pesanan': range(1001, 1001 + JUMLAH_BARIS),
    'Tanggal_Transaksi': [datetime(2024, 1, 1) + timedelta(days=random.randint(0, 364), hours=random.randint(0, 23)) for _ in range(JUMLAH_BARIS)],
    'Kategori_Produk': np.random.choice(list_kategori, JUMLAH_BARIS, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
    'Wilayah': np.random.choice(list_wilayah, JUMLAH_BARIS, p=[0.4, 0.2, 0.2, 0.1, 0.1]),
    'Metode_Pembayaran': np.random.choice(list_pembayaran, JUMLAH_BARIS),
    'Jumlah_Barang': np.random.randint(1, 6, size=JUMLAH_BARIS),
    'Status_Member': np.random.choice([True, False], JUMLAH_BARIS, p=[0.6, 0.4]),
    'Rating_Produk': np.random.randint(1, 6, size=JUMLAH_BARIS) # Rating dari 1-5
}

# --- 1.2. Membuat DataFrame ---
df = pd.DataFrame(data)

# --- 1.3. Menyesuaikan Data agar Lebih Realistis ---
def hitung_total(row):
    basis_harga = {
        'Elektronik': 1500000,
        'Fashion': 350000,
        'Rumah Tangga': 500000,
        'Kesehatan': 150000,
        'Olahraga': 700000
    }
    variasi = random.uniform(0.8, 1.2)
    total = (basis_harga[row['Kategori_Produk']] * row['Jumlah_Barang'] * variasi)
    if row['Status_Member']:
        total *= 0.9 # Diskon 10%
    return int(total)

df['Total_Pembelian'] = df.apply(hitung_total, axis=1)

# Mengatur ulang urutan kolom
df = df[['ID_Pesanan', 'Tanggal_Transaksi', 'Kategori_Produk', 'Wilayah', 'Metode_Pembayaran', 
         'Status_Member', 'Jumlah_Barang', 'Rating_Produk', 'Total_Pembelian']]

print("Dataset berhasil dibuat. 5 baris pertama:")
print(df.head())
print("-" * 40)


# ==============================================================================
# BAGIAN 2: MEMBUAT VISUALISASI GAMBAR
# ==============================================================================
print("\n--- [2] Membuat Visualisasi (Plot) ---")

# Mengatur style plot agar lebih bagus
sns.set_theme(style="whitegrid")

# --- 2.1. EDA: Distribusi Total_Pembelian (Histogram) ---
plt.figure(figsize=(10, 6))
sns.histplot(df['Total_Pembelian'], kde=True, bins=30)
plt.title('Distribusi Total Pembelian', fontsize=16)
plt.xlabel('Total Pembelian (Rp)')
plt.ylabel('Frekuensi')
plt.show()

# --- 2.2. EDA: Kategori Produk Terpopuler (Bar Chart) ---
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Kategori_Produk', order=df['Kategori_Produk'].value_counts().index, palette='viridis')
plt.title('Jumlah Penjualan per Kategori Produk', fontsize=16)
plt.xlabel('Kategori Produk')
plt.ylabel('Jumlah Pesanan')
plt.show()

# --- 2.3. Wawasan: Hubungan Rating vs Total Pembelian (Scatter Plot) ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Rating_Produk', y='Total_Pembelian', alpha=0.6)
plt.title('Hubungan Rating Produk vs Total Pembelian', fontsize=16)
plt.xlabel('Rating Produk (1-5)')
plt.ylabel('Total Pembelian (Rp)')
plt.show()

# --- 2.4. Wawasan: Tren Penjualan Sepanjang Waktu (Line Chart) ---
# Kita perlu mengubah format data (resample) berdasarkan bulan
df_time = df.set_index('Tanggal_Transaksi')
df_bulanan = df_time['Total_Pembelian'].resample('M').sum() # 'M' = Month end

plt.figure(figsize=(12, 6))
df_bulanan.plot(kind='line', marker='o')
plt.title('Tren Total Penjualan Bulanan (2024)', fontsize=16)
plt.xlabel('Bulan')
plt.ylabel('Total Penjualan (Rp)')
plt.grid(True)
plt.show()

# --- 2.5. Hipotesis (T-Test): Perbandingan Member vs Non-Member (Box Plot) ---
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Status_Member', y='Total_Pembelian', palette='muted')
plt.title('Perbandingan Total Pembelian: Member vs Non-Member', fontsize=16)
plt.xlabel('Status Member')
plt.ylabel('Total Pembelian (Rp)')
plt.xticks([0, 1], ['Non-Member', 'Member'])
plt.show()

# --- 2.6. Hipotesis (Chi-Square): Hubungan Kategori vs Wilayah (Heatmap) ---
# Buat tabel kontingensi (crosstab) terlebih dahulu
contingency_table = pd.crosstab(df['Wilayah'], df['Kategori_Produk'])

plt.figure(figsize=(12, 7))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap Hubungan Wilayah vs Kategori Produk', fontsize=16)
plt.xlabel('Kategori Produk')
plt.ylabel('Wilayah')
plt.show()

# --- 2.7. Hipotesis (ANOVA): Perbandingan Rating di Tiap Kategori (Box Plot) ---
plt.figure(figsize=(12, 7))
sns.boxplot(data=df, x='Kategori_Produk', y='Rating_Produk', palette='pastel')
plt.title('Distribusi Rating Produk per Kategori', fontsize=16)
plt.xlabel('Kategori Produk')
plt.ylabel('Rating Produk (1-5)')
plt.show()

from scipy import stats

# 1. Pisahkan data member dan non-member
member_sales = df[df['Status_Member'] == True]['Total_Pembelian']
non_member_sales = df[df['Status_Member'] == False]['Total_Pembelian']

# 2. Lakukan T-Test
# Kita set equal_var=False karena variasi datanya mungkin beda
t_stat, p_value_ttest = stats.ttest_ind(member_sales, non_member_sales, equal_var=False)

print("\n--- [4] Hasil Uji Statistik T-Test ---")
print(f"P-Value: {p_value_ttest}")

# 3. Interpretasi hasil
alpha = 0.05  # Tingkat signifikansi 5%
if p_value_ttest < alpha:
    print("Hasil: Menolak H0. Ada perbedaan yang signifikan secara statistik.")
else:
    print("Hasil: Gagal Menolak H0. Tidak ada perbedaan yang signifikan.")

print("\n--- [3] Selesai. Semua plot telah ditampilkan. ---")