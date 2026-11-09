#!/usr/bin/env python
# coding: utf-8

# In[7]:


# KarinaAuliaWulandari - 202431047
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Naive Bayes

# ## 1. Load Data & EDA

# In[8]:


# KarinaAuliaWulandari - 202431047
data = pd.read_csv('04_bank_customers.csv', sep=';')
data


# In[9]:


# KarinaAuliaWulandari - 202431047
data.head()


# In[10]:


# KarinaAuliaWulandari - 202431047
data.info()


# In[11]:


# KarinaAuliaWulandari - 202431047
data.describe()


# In[12]:


# KarinaAuliaWulandari - 202431047
print("ukuran data (baris, kolom) =", data.shape)
print("\nmissing value per kolom:")
print(data.isnull().sum())
print("\njumlah data duplikat =", data.duplicated().sum())


# In[14]:


# KarinaAuliaWulandari - 202431047
for col in data.columns:
    if str(data[col].dtype) == 'string':
        data[col] = data[col].astype('object')


# In[15]:


# KarinaAuliaWulandari - 202431047
numerical_features = [feature for feature in data.columns if pd.api.types.is_numeric_dtype(data[feature])]
print("jumlah fitur numerik = ", len(numerical_features))
print("daftar fitur numerik =", (numerical_features))
display(data[numerical_features].head())


# In[17]:


# KarinaAuliaWulandari - 202431047
categorical_features = [feature for feature in data.columns if not pd.api.types.is_numeric_dtype(data[feature])]
print("jumlah fitur categorical = ", len(categorical_features))
print("daftar fitur categorical =", (categorical_features))
display(data[categorical_features].head())


# In[18]:


# KarinaAuliaWulandari - 202431047
for i in categorical_features:
    print(i, ':', data[i].unique())


# ## Distribusi target (y)

# In[19]:


# KarinaAuliaWulandari - 202431047
print(data['y'].value_counts())
print(data['y'].value_counts(normalize=True).round(4) * 100)

plt.figure(figsize=(5, 4))
sns.countplot(x='y', data=data, color='steelblue')
plt.title("Distribusi Target (y)")
plt.show()


# ## Distribusi fitur numerik

# In[20]:


# KarinaAuliaWulandari - 202431047
data[numerical_features].hist(bins=30, figsize=(14, 9), color='steelblue', edgecolor='black')
plt.suptitle("Histogram Fitur Numerik")
plt.tight_layout()
plt.show()


# In[21]:


# KarinaAuliaWulandari - 202431047
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, col in zip(axes.flatten(), numerical_features):
    sns.boxplot(y=data[col], ax=ax, color='lightcoral')
    ax.set_title(col)
axes.flatten()[-1].axis('off')
plt.tight_layout()
plt.show()


# ## Distribusi fitur kategorikal terhadap target

# In[22]:


# KarinaAuliaWulandari - 202431047
cat_features_x = [c for c in categorical_features if c != 'y']

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
for ax, col in zip(axes.flatten(), cat_features_x):
    sns.countplot(x=col, hue='y', data=data, ax=ax, palette='Set2')
    ax.set_title(f"{col} vs y")
    ax.tick_params(axis='x', rotation=45)
for ax in axes.flatten()[len(cat_features_x):]:
    ax.axis('off')
plt.tight_layout()
plt.show()


# ## Korelasi antar fitur numerik

# In[23]:


# KarinaAuliaWulandari - 202431047
plt.figure(figsize=(8, 6))
sns.heatmap(data[numerical_features].corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title("Heatmap Korelasi Fitur Numerik")
plt.show()


# ## 2. Preprocessing Data (Encoding)

# In[24]:


# KarinaAuliaWulandari - 202431047
from sklearn.preprocessing import LabelEncoder

data_nb = data.copy()   # salinan agar data asli tetap utuh untuk bagian KNN

label_encoder = LabelEncoder()
for feature in categorical_features:
    print(feature, ':', data_nb[feature].unique())
    data_nb[feature] = label_encoder.fit_transform(data_nb[feature])
    print(feature, ':', data_nb[feature].unique())
    print('---')


# In[25]:


# KarinaAuliaWulandari - 202431047
data_nb.head()


# In[26]:


# KarinaAuliaWulandari - 202431047
x = data_nb.drop(['y'], axis = 1)
y = data_nb['y']


# In[27]:


# KarinaAuliaWulandari - 202431047
x


# In[28]:


# KarinaAuliaWulandari - 202431047
y


# ## 3. Split Data & Training Naive Bayes

# In[29]:


# KarinaAuliaWulandari - 202431047
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
print(f'x_train shape : {x_train.shape}')
print(f'x_test shape  : {x_test.shape}')


# In[30]:


# KarinaAuliaWulandari - 202431047
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(x_train, y_train)


# ## 4. Evaluasi Naive Bayes

# In[31]:


# KarinaAuliaWulandari - 202431047
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

acc_nb_train = accuracy_score(y_train, y_pred_train)
acc_nb_test = accuracy_score(y_test, y_pred_test)

print(f"Akurasi Naive Bayes (data train) : {acc_nb_train:.4f}")
print(f"Akurasi Naive Bayes (data test)  : {acc_nb_test:.4f}")


# In[34]:


# KarinaAuliaWulandari - 202431047
labels = [0, 1]   # 0 = no, 1 = yes (hasil LabelEncoder)

cm_train = confusion_matrix(y_train, y_pred_train, labels=labels)
print("Evaluasi Training: ")
plt.figure(figsize=(6, 5))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Reds', linewidths=0.5, linecolor='black',
            xticklabels=['no', 'yes'], yticklabels=['no', 'yes'])
plt.title("Confusion Matrix - Training (Naive Bayes)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

cm_test = confusion_matrix(y_test, y_pred_test, labels=labels)
print("\nEvaluasi Test: ")
plt.figure(figsize=(6, 5))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', linewidths=0.5, linecolor='black',
            xticklabels=['no', 'yes'], yticklabels=['no', 'yes'])
plt.title("Confusion Matrix - Test (Naive Bayes)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(y_test, y_pred_test, target_names=['no', 'yes'], zero_division=0))


# ## 5. Prediksi Data Baru 

# In[36]:


# KarinaAuliaWulandari - 202431047
print(list(x.columns))

# age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome
# (nilai kategori berupa hasil LabelEncoder, bukan teks)
new_customer = pd.DataFrame([[45, 4, 1, 2, 0, 1500, 1, 0, 0, 15, 8, 300, 2, -1, 0, 3]], columns=x.columns)
prediction = model.predict(new_customer)
print("prediksi nasabah baru (0 = no, 1 = yes):", prediction[0])


# # KNN

# ## 1. Load Data & Preprocessing

# In[37]:


# KarinaAuliaWulandari - 202431047
data = pd.read_csv('04_bank_customers.csv', sep=';')
data


# In[38]:


# KarinaAuliaWulandari - 202431047
x = data.drop(['y'], axis = 1)
y = data['y']


# In[39]:


# KarinaAuliaWulandari - 202431047
x


# In[40]:


# KarinaAuliaWulandari - 202431047
y


# In[41]:


# KarinaAuliaWulandari - 202431047
y = y.map({'no': 0, 'yes': 1})

x = pd.get_dummies(x, drop_first=True)
x = x.astype(float)

print(x.dtypes.value_counts())
print("jumlah fitur setelah encoding =", x.shape[1])
x.head()


# ## 2. Split Data

# In[42]:


# KarinaAuliaWulandari - 202431047
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


# In[43]:


# KarinaAuliaWulandari - 202431047
print(f'x_train shape : {x_train.shape}')
print(f'x_test shape : {x_test.shape}')


# ## 3. KNN tanpa Scaling (K = 7)

# In[44]:


# KarinaAuliaWulandari - 202431047
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)


# In[45]:


# KarinaAuliaWulandari - 202431047
y_pred = knn.predict(x_test)


# In[46]:


# KarinaAuliaWulandari - 202431047
acc_knn_raw = accuracy_score(y_test, y_pred)
print(f"Akurasi KNN (K=7, tanpa scaling): {acc_knn_raw:.4f}")


# ## 4. KNN dengan Scaling (StandardScaler)

# In[47]:


# KarinaAuliaWulandari - 202431047
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)

# Transform x_train dan x_test
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[48]:


# KarinaAuliaWulandari - 202431047
K_TARGET = 15
model_k15 = KNeighborsClassifier(n_neighbors=K_TARGET)
model_k15.fit(x_train_scaled, y_train)
y_pred_k15 = model_k15.predict(x_test_scaled)

# Hitung Akurasi Model
accuracy_k15 = accuracy_score(y_test, y_pred_k15)

print(f"Nilai K yang digunakan: {K_TARGET}")
print(f"Akurasi Model KNN pada data test: {accuracy_k15:.4f}")


# ## 5. Pemilihan Nilai K Terbaik

# In[52]:


# KarinaAuliaWulandari - 202431047
k_values = range(1, 20, 2)   # K ganjil 1 - 19
acc_raw = []
acc_scaled = []

for k in k_values:
    m = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

    m.fit(x_train, y_train)
    acc_raw.append(accuracy_score(y_test, m.predict(x_test)))

    m.fit(x_train_scaled, y_train)
    acc_scaled.append(accuracy_score(y_test, m.predict(x_test_scaled)))

plt.figure(figsize=(8, 5))
plt.plot(k_values, acc_raw, marker='o', label='Tanpa scaling')
plt.plot(k_values, acc_scaled, marker='o', label='Dengan scaling')
plt.xlabel("Nilai K")
plt.ylabel("Akurasi")
plt.title("Pemilihan Nilai K pada KNN")
plt.xticks(list(k_values))
plt.legend()
plt.grid(True)
plt.show()

best_k = list(k_values)[int(np.argmax(acc_scaled))]
print(f"K terbaik (data scaling) = {best_k} dengan akurasi {max(acc_scaled):.4f}")


# ## 6. Evaluasi Model KNN Terbaik

# In[53]:


# KarinaAuliaWulandari - 202431047
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(x_train_scaled, y_train)
y_pred_best = knn_best.predict(x_test_scaled)

acc_knn_best = accuracy_score(y_test, y_pred_best)
print(f"Akurasi KNN terbaik (K={best_k}) : {acc_knn_best:.4f}\n")

cm_knn = confusion_matrix(y_test, y_pred_best, labels=[0, 1])
plt.figure(figsize=(6, 5))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', linewidths=0.5, linecolor='black',
            xticklabels=['no', 'yes'], yticklabels=['no', 'yes'])
plt.title(f"Confusion Matrix - Test (KNN, K={best_k})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(y_test, y_pred_best, target_names=['no', 'yes'], zero_division=0))


# ## 7. Prediksi Data Baru (KNN)

# In[55]:


# KarinaAuliaWulandari - 202431047
x_new = x_test.iloc[[0]]
x_new_scaled = scaler.transform(x_new)

print("prediksi KNN  :", knn_best.predict(x_new_scaled)[0], "(0 = no, 1 = yes)")
print("label aslinya :", y_test.iloc[0])


# # Perbandingan Akurasi Naive Bayes vs KNN

# In[56]:


# KarinaAuliaWulandari - 202431047
baseline = max(y_test.mean(), 1 - y_test.mean())   # akurasi jika selalu menebak kelas mayoritas

hasil = pd.DataFrame({
    'Model': ['Naive Bayes (GaussianNB)', 'KNN (K=7, tanpa scaling)', f'KNN (K={best_k}, scaling)', 'Baseline (tebak kelas mayoritas)'],
    'Akurasi Test': [acc_nb_test, acc_knn_raw, acc_knn_best, baseline]
})
display(hasil)

plt.figure(figsize=(8, 4))
sns.barplot(x='Akurasi Test', y='Model', data=hasil, color='teal')
plt.xlim(0, 1)
plt.title("Perbandingan Akurasi Model")
plt.show()

