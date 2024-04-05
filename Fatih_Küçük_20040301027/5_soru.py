import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Veri setini yükleme
file_path = "veri-seti.txt"
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(file_path, names=names, sep='\t')

# Bağımsız değişkenler ve hedef değişkeni ayırma
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Veri setini eğitim ve test alt kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Naive Bayes sınıflandırıcısını uygulama
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

# Eğitim veri seti için performans metriklerini hesaplama
train_predictions = naive_bayes.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print("Eğitim Veri Seti Doğruluğu:", train_accuracy)

# Test veri seti için performans metriklerini hesaplama
test_predictions = naive_bayes.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Test Veri Seti Doğruluğu:", test_accuracy)

# Test veri seti için sınıflandırma raporunu yazdırma
print("\nTest Veri Seti Sınıflandırma Raporu:")
print(classification_report(y_test, test_predictions))
