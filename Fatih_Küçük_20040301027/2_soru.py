import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

# Veri setini yükleme
file_path = "veri-seti.txt"
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(file_path, names=names, sep='\t')

# Bağımsız değişkenler ve hedef değişkeni ayırma
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# PCA modelini oluşturma ve uygulama
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# LDA modelini oluşturma ve uygulama
lda = LinearDiscriminantAnalysis(n_components=1) # LDA'da n_components parametresi 1 olarak ayarlandı
X_lda = lda.fit_transform(X, y)

# PCA sonuçlarını ekrana yazdırma
print("PCA Sonuçları:")
print(pd.DataFrame(X_pca, columns=['PCA1', 'PCA2']).head())

# LDA sonuçlarını ekrana yazdırma
print("\nLDA Sonuçları:")
print(pd.DataFrame(X_lda, columns=['LD1']).head())
# Görselleştirme
plt.figure(figsize=(12, 5))

# PCA görselleştirmesi
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# LDA görselleştirmesi
plt.subplot(1, 2, 2)
plt.scatter(X_lda[:, 0], [0] * len(X_lda), c=y, cmap='viridis')
plt.title('LDA')
plt.xlabel('LD 1')

plt.tight_layout()
plt.show()
