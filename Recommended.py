# %% [markdown]
# # Import Library

# %%
import shutil
import os
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import seaborn as sns
import matplotlib.pyplot as plt

from surprise import Reader
from surprise import Dataset
from surprise import SVD
from surprise import accuracy

from surprise.model_selection import train_test_split

# %% [markdown]
# # Data Load

# %%
file_path = 'buku.zip'
shutil.unpack_archive(file_path, 'buku')

# %%
books_df = pd.read_csv('buku/Books.csv')
users_df = pd.read_csv('buku/Users.csv')
ratings_df = pd.read_csv('buku/Ratings.csv')

# %% [markdown]
# # Data Understanding

# %% [markdown]
# Books_df dataframe

# %%
books_df.info()

# %%
books_df.shape

# %% [markdown]
# pada dataframe books_df terdiri dari 271360 baris dan 8 kolom didalamnya dengan kolomsebagai berikut:
# 1. ISBN = Nomor buku berstandar internasional
# 2. Book-Title = Judul buku
# 3. Book-Author = Penulis buku
# 4. Year-Of-Publication = Tahun terbit
# 5. Publisher = Penerbit
# 6. Image-URL-S = url gambar
# 7. Image-URL-M = url gambar
# 8. Image-URL-L = url gambar

# %%
books_df.head()

# %%
books_df['Publisher'].value_counts()

# %%
books_df['Book-Author'].value_counts()

# %%
books_df.isna().sum()

# %%
books_df.duplicated().sum()

# %% [markdown]
# pada data books_df terdapat data yang hilang pada beberapa kolom seperti Book-Author, Publisher dan Image-URL-L, selain itu juga terdapat ketidaksesuaian tipe data pada kolom year-of-publication.
# 
# pada data yang hilang tersebut tidak aakn digunakan pada tahapan selanjutnya maka akan diabaikan.

# %% [markdown]
# ## users_df dataframe

# %%
users_df.info()

# %%
users_df.shape

# %% [markdown]
# dataframe users_df sendiri adalah dataframe yang berisi data user yang terdiri dari 278858 baris dan 3 kolom didalamnya dengan kolom sebagai berikut:
# 1. User-ID = ID user
# 2. Location = Lokasi
# 3. Age = Umur

# %%
users_df.describe()

# %%
users_df.head()

# %%
users_df.isna().sum()

# %%
users_df.duplicated().sum()

# %%
users_df['Age'].fillna(users_df['Age'].mean(), inplace=True)

# %%
bins = [0, 10, 20, 30, 40, 50, 60, float('inf')]
labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '>60']
users_df['Age-Range'] = pd.cut(users_df['Age'], bins=bins, labels=labels, right=False)

# %%
age_counts = users_df['Age-Range'].value_counts().sort_index()

# %%
sns_barplot = sns.barplot(x=age_counts.index, y=age_counts.values, palette='rocket')
plt.title('Distribution of Age Ranges')
plt.xlabel('Age Range')
plt.ylabel('Count')
plt.show()

# %%
users_df['Nationality'] = users_df['Location'].str.split(',').str[-1].str.strip()

# %%
users_df['Province'] = users_df['Location'].str.split(',').str[1].str.strip()

# %%
users_df['City'] = users_df['Location'].str.split(',').str[0].str.strip()

# %% [markdown]
# pada data users_df terdapat data yang hilang pada beberapa kolom Age, data yang hilang pun tidak sedikit yang mencapai hampir setengah dari keseluruhan data yang ada, akan tetapi berhubung tidak akan digunakan pada tahapan selanjutnya maka akan diabaikan.

# %%
nationality_counts = users_df['Nationality'].value_counts()

top_n = 5
top_nationalities = nationality_counts[:top_n]
other_nationalities = nationality_counts[top_n:].sum()

if other_nationalities > 0:
    top_nationalities.loc['Other'] = other_nationalities

plt.figure(figsize=(8, 8))
plt.pie(top_nationalities.values, labels=top_nationalities.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20c(np.linspace(0, 1, len(top_nationalities))))

plt.title('Distribution of Nationalities')
plt.show()

# %% [markdown]
# ## ratings_df dataframe

# %%
ratings_df.info()

# %%
ratings_df.shape

# %% [markdown]
# dataframe ratings_df sendiri adalah dataframe yang berisi data rating dari setiap film dan useryang terdiri dari 1149780 baris dan 3 kolom didalamnya dengan kolom sebagai berikut:
# 1. User-ID = ID user
# 2. ISBN = Nomor buku berstandar internasional
# 3. Book-Rating = Rating

# %%
ratings_df.head()

# %%
ratings_df.describe()

# %%
ratings_df['Book-Rating'].describe().round(2)

# %%
sns.barplot(x=ratings_df['Book-Rating'].value_counts().index, y=ratings_df['Book-Rating'].value_counts().values, palette='rocket')
plt.show()

# %%
ratings_df.isna().sum()

# %%
ratings_df.duplicated().sum()

# %% [markdown]
# pada data ratings_df tidak ada data hilang, akan tetapi tipe data pada kolom user-id belum menjadi str yang nantinya akan bermasalah saat pemrosessan model, maka akan dilakukan perubahan tipe data pada kolom tersebut pada proses selanjutnya.

# %% [markdown]
# # Data Preparation
# pada bagain ini akan dilakukan beberapa tahapan yang diperlukan untuk mempersiapkan beberapa buku yang akan direkomendasikan kepada user. Tahapan tersebut diantaranya:
# 1. Mengisi data Age yang hilang pada dataframe users_df
# 2. Menggabungkan data buku dan rating melalui kolom ISBN
# 3. Membuat tabel yang berisi jumlah yang membaca sebuah buku oleh user
# 4. Membuat tabel rata-rata rating pada setiap buku yang ada melalui rata-rata rating yang diberikan oleh user
# 5. Menggabungkan data jumlah rating (count_df) dan rata-rata rating (mean_df) menjadi satu tabel
# 6. Mengambil data buku dengan jumlah pembaca di atas 20 dan rata-rata rating di atas 4.5
# 7. Menerapkan filter pada merge_df dengan data preparation yang dihasilkan dari tahapan ke-4
# 8. Drop kolom yang tidak diperlukan
# 9. Membuat matrix dengan menggunakan pivot table
# 10. Mempersiapkan data untuk model SVD

# %%
# mengisi data yang hilang
users_df['Age'].fillna(users_df['Age'].mean(), inplace=True)

# %%
# menggabungkan data buku dan rating
merge_df = pd.merge(ratings_df, books_df, on='ISBN')
merge_df.head()

# %%
merge_df.info()

# %%
# membuat tabel rating pada setiap buku yang ada 
mean_df = merge_df.groupby('Book-Title')['Book-Rating'].mean().sort_values(ascending=False)
mean_df = mean_df.reset_index(name = 'mean')
mean_df.head()

# %%
# membuat tabel yang berisi jumlah yang membaca setiap buku
count_df = merge_df.groupby('Book-Title')['Book-Rating'].count().sort_values(ascending=False)
count_df = count_df.reset_index(name='count')
count_df.head()


# %%
#menggabungkan data jumlah rating (count_df) dan rata-rata rating (mean_df) menjadi satu tabel
preparation = pd.merge(mean_df, count_df, on='Book-Title')
preparation.head()

# %%
# mengambil data dengan ketentuan jumlah pembaca di atas 20 dan rata-rata rating di atas 4.5
preparation = preparation[preparation['count'] > 20]
preparation = preparation[preparation['mean'] > 4.5]
preparation

# %%
# menerapkan filter pada merge_df dengan data preparation dengan melakukkan filter pada kolom Book-Title
fix_df = merge_df[merge_df['Book-Title'].isin(preparation['Book-Title'])]

# %%
fix_df.info()

# %%
# mengubah tipe data User-ID menjadi string
fix_df['User-ID'] = fix_df['User-ID'].astype(str)

# %%
fix_df.info()

# %%
fix_df.head()

# %%
fix_df = fix_df.drop(columns=['ISBN', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1)
fix_df.head()

# %%
# melakukan filter pada fix_df untuk memilih hanya yang memiliki rating lebih besar dari 0
fix_df = fix_df[fix_df['Book-Rating'] > 0]

# %%
# membuat matrix dengan menggunakan pivot table
user_item_matrix = fix_df.pivot_table(
    index='User-ID', columns='Book-Title', values='Book-Rating'
)

# %%
user_item_matrix.fillna(0, inplace=True)

# %% [markdown]
# mempersiapkan data untuk model SVD

# %%
# membaca dataframe ke format surprise
reader = Reader(rating_scale=(0, 10))
data = Dataset.load_from_df(merge_df[['User-ID', 'Book-Title', 'Book-Rating']], reader)

# %%
# split data menjadi train dan test
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# %% [markdown]
# # Model Development
# 1. Cosine Similarity
# 2. Model SVD

# %% [markdown]
# ## Cosine Similarity

# %%
# melakukan perhitungan cosine similarity
book_similarity = cosine_similarity(user_item_matrix.T)

# %%
# membuat hasil perhitungan cosine similarity menjadi bentuk dataframe
book_similarity_df = pd.DataFrame(
    book_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns
)

# %%
book_similarity_df.head()

# %%
def get_similar_books(book_title, top_n=5):
    # Mengecek apakah book_title ada di DataFrame
    if book_title not in book_similarity_df.columns:
        print(f"Buku '{book_title}' tidak ditemukan di database.")
        return
    
    # Mengambil daftar rekomendasi berdasarkan cosine similarity
    recommendation = book_similarity_df[book_title].sort_values(ascending=False)

    # Memilih buku dengan kesamaan cosine similarity di atas 0.0
    filtered_recommendation = recommendation[recommendation > 0.0]

    # Mengatur ulang index
    filtered_recommendation = filtered_recommendation.reset_index()
    filtered_recommendation.columns = ['Book-Title', 'Similarity']

    # Mengurutkan ulang berdasarkan nilai similarity (meskipun sudah diurutkan)
    filtered_recommendation.sort_values('Similarity', ascending=False, inplace=True)

    # Menampilkan hasil rekomendasi (mengabaikan buku itu sendiri di indeks 0)
    print(f"Rekomendasi untuk buku '{book_title}':")
    for i, (book, similarity) in enumerate(filtered_recommendation.iloc[1:top_n+1].values, start=1):
        print(f"{i}. Book: {book} (Similarity: {round(similarity, 2)})")

# %%
get_similar_books('Harry Potter and the Order of the Phoenix (Book 5)')

# %% [markdown]
# ## Model SVD

# %%
model = SVD()

# %%
model.fit(trainset)

# %%
def get_recommendations(user_id):
    # daftar judul buku
    all_books_title = fix_df['Book-Title'].unique()

    # buku yang telah dibaca oleh pengguna
    books_read = fix_df[fix_df['User-ID'] == user_id]['Book-Title'].unique()

    # buku yang belum dibaca oleh pengguna
    books_not_read = [book for book in all_books_title if book not in books_read]

    # prediksi rating untuk buku yang belum dibaca oleh pengguna
    recommendations = []
    for book_title in books_not_read:
        predictions = model.predict(user_id, book_title)
        recommendations.append((book_title, predictions.est))

    # mengurutkan rekomendasi berdasarkan prediksi rating
    recommendations.sort(key=lambda x: x[1], reverse=True)

    print (f"Rekomendasi buku untuk pengguna dengan ID {user_id}:")

    for book_title, rating in recommendations[:5]:
        print(f"Buku: {book_title} (Prediksi Rating: {rating.round(1)})")

# %%
get_recommendations('276729')

# %% [markdown]
# # Model Evaluation

# %% [markdown]
# ## Model Cosine Similarity

# %%
# Fungsi untuk memprediksi rating berdasarkan cosine similarity
def predict_rating(books_similarity, rating_matrix):
    predict_rating = np.dot(books_similarity, rating_matrix.T)/np.abs(books_similarity).sum(axis=1, keepdims=True)
    return predict_rating.T

# %%
predicted_rating = predict_rating(book_similarity, user_item_matrix)
predicted_rating_df = pd.DataFrame(predicted_rating, index=user_item_matrix.index, columns=user_item_matrix.columns)
predicted_rating_df.head()

# %%
# mengambil data rating yang sebenarnya
actual_rating = user_item_matrix.values 

# menghitung RMSE pada data yang tidak nol
mask = actual_rating > 0
rmse = np.sqrt(np.mean((predicted_rating_df.values[mask] - actual_rating[mask])**2))
print (f"RMSE: {rmse:.4f}")

# %% [markdown]
# ## Model SVD

# %%
# Evaluasi model SVD berdasarkan RMSE
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)

# %% [markdown]
# dari hasil diatas dapat disimpulkan bahwa model SVD memiliki RMSE yang lebih kecil yang menunjukkan bahwa model SVD lebih akurat dalam memprediksi rating buku yang sesuai dengan rating yang sebenarnya dibandingkan dengan model cosine similarity.


