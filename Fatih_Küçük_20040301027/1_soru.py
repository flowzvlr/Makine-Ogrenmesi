from sklearn import preprocessing
import numpy as np

# Veri setini dosyadan okuma
with open("veri-seti.txt", "r") as file:
    lines = file.readlines()

# Veri setini numpy dizisine dönüştürme
data = []
for line in lines:
    row = [float(x) for x in line.strip().split('\t')]
    data.append(row)
data = np.array(data)

# Normalizasyon işlemi için Min-Max ölçeklendirme
min_max_scaler = preprocessing.MinMaxScaler()
normalized_data = min_max_scaler.fit_transform(data)

print("Normalize Edilmiş Veri:")
print(normalized_data)