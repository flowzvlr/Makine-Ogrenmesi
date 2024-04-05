import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Veri setini yükleme
file_path = "veri-seti.txt"
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(file_path, names=names, sep='\t')

# Bağımsız değişkenler ve hedef değişkeni ayırma
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Veri setini eğitim ve test alt kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Karar ağacı sınıflandırma modelini eğitme
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Ağaç yapısını görselleştirme
plt.figure(figsize=(12, 8))
plot_tree(decision_tree, feature_names=names[:-1], class_names=['0', '1'], filled=True)
plt.show()

# Modeli test veri seti üzerinde değerlendirme
y_pred = decision_tree.predict(X_test)

# Doğruluk değerini hesaplama
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk:", accuracy)

# Kestirim sonuçlarını ekrana yazdırma
print("Kestirim Sonuçları:")
for i, (real, pred) in enumerate(zip(y_test, y_pred)):
    print(f"Test {i+1}: Gerçek={real}, Tahmin={pred}")


# Sınıflandırma raporunu oluşturma
classification_report_result = classification_report(y_test, y_pred)
print("Sınıflandırma Raporu:")
print(classification_report_result)