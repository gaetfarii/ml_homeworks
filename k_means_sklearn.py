from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt

# Загрузка данных ирисов
dataset_iris = datasets.load_iris()
array = dataset_iris.data

# Массив для суммы квадратов расстояний от точек до центроид
sum_squared_distances = []

# Метод локтя
for clusters in range(1, 11):
    kmeans = KMeans(n_clusters=clusters, n_init=10, random_state=42)
    kmeans.fit(array)  # Обучение модели KMeans на загруженных данных
    sum_squared_distances.append(kmeans.inertia_)

# Построение графика
plt.plot(range(1, 11), sum_squared_distances, marker='o')
plt.xlabel('Количество кластеров')
plt.ylabel('Сумма квадратов расстояний')
plt.title('Метод локтя')
plt.grid(True)  # Добавляем сетку для удобства
plt.show()
