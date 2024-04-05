import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score

# Veri setini yükleme
file_path = "veri-seti.txt"
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(file_path, names=names, sep='\t')

# Bağımsız değişkenler ve hedef değişkeni ayırma
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Veri setini eğitim ve test alt kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Çoklu Doğrusal Regresyon analizi uygulama
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Katsayıları raporlama
print("Çoklu Doğrusal Regresyon Katsayıları:")
coefficients_linear = pd.DataFrame(linear_reg.coef_, X.columns, columns=['Katsayı'])
print(coefficients_linear)

# Multinominal Lojistik Regresyon analizi uygulama
logistic_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
logistic_reg.fit(X_train, y_train)

# Katsayıları raporlama
print("\nMultinominal Lojistik Regresyon Katsayıları:")
coefficients_logistic = pd.DataFrame(logistic_reg.coef_.transpose(), X.columns, columns=['Katsayı'])
print(coefficients_logistic)

# Test veri seti için performans metriklerini hesaplama
y_pred_linear = linear_reg.predict(X_test)
y_pred_logistic = logistic_reg.predict(X_test)

# Performans metriklerini hesaplama
accuracy_linear = accuracy_score(y_test, y_pred_linear.round())
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)

print("\nTest Veri Seti Performans Metrikleri:")
print("Çoklu Doğrusal Regresyon Doğruluk:", accuracy_linear)
print("Multinominal Lojistik Regresyon Doğruluk:", accuracy_logistic)
