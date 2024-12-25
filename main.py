import numpy as np
from sklearn.preprocessing import Binarizer, StandardScaler, MinMaxScaler, normalize

# Вхідні дані (варіант №1)
data = np.array([[-7.2, 2.4, 6.8],
                 [5.8, 1.8, 3.7],
                 [9.6, -0.3, -1.7],
                 [4.1, 7.8, 4.2]])
print("Оригінальні дані:\n", data)

# Бінаризація
binarizer = Binarizer(threshold=2.1)
binary_data = binarizer.transform(data)
print("Бінаризовані дані:\n", binary_data)

# Виключення середнього
scaler = StandardScaler(with_mean=True, with_std=False)  # Тільки центрування
mean_removed_data = scaler.fit_transform(data)
print("Дані після виключення середнього:\n", mean_removed_data)

# Масштабування
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = min_max_scaler.fit_transform(data)
print("Масштабовані дані:\n", scaled_data)

# Нормалізація (L1-норма)
normalized_data_l1 = normalize(data, norm='l1')
print("L1-нормалізовані дані:\n", normalized_data_l1)

# Нормалізація (L2-норма)
normalized_data_l2 = normalize(data, norm='l2')
print("L2-нормалізовані дані:\n", normalized_data_l2)

