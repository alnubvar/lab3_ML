import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsRegressor

# === Настройки ===
plt.style.use("seaborn-v0_8-whitegrid")
os.makedirs("figures", exist_ok=True)

# === Загрузка данных ===
df = pd.read_excel("Pumpkin_Seeds_Dataset.xlsx")
df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]

THRESH = 0.9903
df["target_bin"] = (df["Solidity"] <= THRESH).astype(int)

# === Задание 1. KNeighborsClassifier ===
print("\n--- Задание 1: KNeighborsClassifier ---")

X = df.drop(columns=["Class", "Solidity", "target_bin"])
y = df["target_bin"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_errors, test_errors = [], []
k_values = range(1, 61)

for k in k_values:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])
    model.fit(X_train, y_train)

    train_errors.append(1 - accuracy_score(y_train, model.predict(X_train)))
    test_errors.append(1 - accuracy_score(y_test, model.predict(X_test)))

best_k = k_values[np.argmin(test_errors)]
print(f"Оптимальное количество соседей: {best_k}")

plt.figure(figsize=(8, 5))
plt.plot(k_values, train_errors, label="Train error", marker='o')
plt.plot(k_values, test_errors, label="Test error", marker='o')
plt.xlabel("Количество соседей (k)")
plt.ylabel("Ошибка")
plt.title("Задание 1 — KNN: Ошибка на обучающей и тестовой выборках")
plt.legend()
plt.tight_layout()
plt.savefig("figures/lab3_task1_knn_errors.png")
plt.close()

# === Задание 2. KNeighborsRegressor ===
print("\n--- Задание 2: KNeighborsRegressor ---")

X = df.drop(columns=["Class", "Solidity", "target_bin"])
y = df["Solidity"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
A, B = np.quantile(X_scaled, 0.10, axis=0), np.quantile(X_scaled, 0.90, axis=0)
t = np.linspace(0, 1, 1000).reshape(-1, 1)
X_line = A + t * (B - A)

# (а) влияние n_neighbors
plt.figure(figsize=(8, 5))
for k in [1, 5, 20, 60]:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_scaled, y)
    plt.plot(t, model.predict(X_line), label=f"k={k}")
plt.title("Задание 2 — KNeighborsRegressor: влияние n_neighbors")
plt.xlabel("Положение точки на отрезке AB")
plt.ylabel("Прогноз Solidity")
plt.legend()
plt.tight_layout()
plt.savefig("figures/lab3_task2_n_neighbors.png")
plt.close()

# (б) влияние weights
plt.figure(figsize=(8, 5))
for w in ["uniform", "distance"]:
    model = KNeighborsRegressor(n_neighbors=10, weights=w)
    model.fit(X_scaled, y)
    plt.plot(t, model.predict(X_line), label=f"weights='{w}'")
plt.title("Задание 2 — KNeighborsRegressor: влияние параметра weights")
plt.xlabel("Положение точки на отрезке AB")
plt.ylabel("Прогноз Solidity")
plt.legend()
plt.tight_layout()
plt.savefig("figures/lab3_task2_weights.png")
plt.close()

# (в) сравнение KNN и RadiusNeighborsRegressor
plt.figure(figsize=(8, 5))
model_knn = KNeighborsRegressor(n_neighbors=10, weights='distance')
model_rad = RadiusNeighborsRegressor(radius=2.0, weights='distance')
model_knn.fit(X_scaled, y)
model_rad.fit(X_scaled, y)
plt.plot(t, model_knn.predict(X_line), label="KNeighborsRegressor")
plt.plot(t, model_rad.predict(X_line), label="RadiusNeighborsRegressor")
plt.title("Задание 2 — Сравнение KNN и RadiusNeighborsRegressor")
plt.xlabel("Положение точки на отрезке AB")
plt.ylabel("Прогноз Solidity")
plt.legend()
plt.tight_layout()
plt.savefig("figures/lab3_task2_comparison.png")
plt.close()

# === Задание 3. Проклятие размерности ===
print("\n--- Задание 3: Проклятие размерности ---")

X = df.drop(columns=["Class", "Solidity", "target_bin"])
X_scaled = StandardScaler().fit_transform(X)

dist_small = pairwise_distances(X_scaled[:, :2])
dist_high = pairwise_distances(X_scaled)
d_small = dist_small[np.triu_indices_from(dist_small, k=1)]
d_high = dist_high[np.triu_indices_from(dist_high, k=1)]

plt.figure(figsize=(8, 5))
plt.hist(d_small, bins=40, alpha=0.6, label="2 признака")
plt.hist(d_high, bins=40, alpha=0.6, label=f"{X.shape[1]} признаков")
plt.title("Задание 3 — Проклятие размерности")
plt.xlabel("Расстояние между точками")
plt.ylabel("Частота")
plt.legend()
plt.tight_layout()
plt.savefig("figures/lab3_task3_dimensionality_curse.png")
plt.close()

# === Задание 4. Типичность объектов ===
print("\n--- Задание 4: Типичность объектов ---")

X = df.drop(columns=["Class", "Solidity", "target_bin"])
y = df["target_bin"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

best_k = 6

model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k))
])
model.fit(X_train, y_train)

proba = model.predict_proba(X_train)
p_true = proba[np.arange(len(y_train)), y_train.to_numpy()]
G = 2 * p_true - 1
G_sorted = np.sort(G)

plt.figure(figsize=(8, 5))
plt.plot(G_sorted, marker='.', color='blue', linewidth=1.5)

plt.axhspan(-1, 0, color='red', alpha=0.15, label='Выбросы (G < 0)')
plt.axhspan(0, 0.5, color='orange', alpha=0.15, label='Периферийные (0 ≤ G ≤ 0.5)')
plt.axhspan(0.5, 1, color='green', alpha=0.15, label='Эталоны (G > 0.5)')

plt.title("Задание 4 — Типичность объектов (значения выступа G)")
plt.xlabel("Номер объекта (отсортировано)")
plt.ylabel("G(x)")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()

# 5. Сохранение графика
plt.savefig("figures/lab3_task4_typicality_zones.png")
plt.close()

print("График типичности с зонами сохранён в figures/lab3_task4_typicality_zones.png")
print("\nВсе графики сохранены в папку ./figures/")
