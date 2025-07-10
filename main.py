from arctic_data_loader import ArcticDataLoader
from clustering import perform_clustering, visualize_clusters
from feature_extraction import extract_features, compute_trends
from classification import train_classifier, evaluate_classifier
from evaluation import calculate_metrics
from visualization import plot_similar_years
import numpy as np

def main():
    # 1. Загрузка и подготовка данных
    data_loader = ArcticDataLoader(data_path="data/arctic_ice.npy")
    data = data_loader.load_data()
    data = data_loader.preprocess(data)

    # 2. Извлечение признаков и трендов
    features = extract_features(data)
    trends = compute_trends(features)

    # 3. Кластеризация
    cluster_labels = perform_clustering(features, n_clusters=4)
    visualize_clusters(data, cluster_labels)

    # 4. Обучение классификатора
    classifier = train_classifier(features, cluster_labels)

    # 5. Оценка модели
    predictions = classifier.predict(features)
    metrics = calculate_metrics(cluster_labels, predictions)
    print("Оценка модели:", metrics)

    # 6. Визуализация схожих годов
    plot_similar_years(data, cluster_labels)

if __name__ == "__main__":
    main()
