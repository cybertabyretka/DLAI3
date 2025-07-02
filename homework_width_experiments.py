import itertools
import logging
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch.cuda as cuda

from utils.datasets_utils import get_cifar_loaders, get_mnist_loaders
from utils.experiment_utils import *
from utils.model_utils import count_parameters

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f'CUDA доступна: {cuda.is_available()}')
logging.info(f'Количество GPU: {cuda.device_count()}')
logging.info(f'Текущее устройство: {cuda.current_device()}')
logging.info(f'Название GPU: {cuda.get_device_name(0)}')

# 2.1 Сравнение моделей разной ширины
print(f'{"-" * 70}\n2.1 Сравнение моделей разной ширины\n{"-" * 70}')
configs = {
    "narrow": {
        "layers": [64, 32, 16],
    },
    "medium": {
        "layers": [256, 128, 64],
    },
    "wide": {
        "layers": [1024, 512, 256],
    },
    "extra_wide": {
        "layers": [2048, 1024, 512],
    }
}

results = []

mnist_train, mnist_test = get_mnist_loaders(batch_size=1024)
cifar_train, cifar_test = get_cifar_loaders(batch_size=1024)

mnist_epochs = 10
cifar_epochs = 10

for name, cfg in configs.items():
    w1, w2, w3 = cfg["layers"]
    for dataset_type, (train_loader, test_loader, epochs) in {
        "mnist": (mnist_train, mnist_test, mnist_epochs),
        "cifar10": (cifar_train, cifar_test, cifar_epochs),
    }.items():
        logging.info(f'Ширина: ({w1}, {w2}, {w3}), Датасет: {dataset_type}')

        model_cfg = {
            "num_classes": 10,
            "layers": [
                {"type": "linear", "size": w1},
                {"type": "relu"},
                {"type": "linear", "size": w2},
                {"type": "relu"},
                {"type": "linear", "size": w3},
                {"type": "relu"},
            ]
        }
        model = FullyConnectedModel(**fix_config(model_cfg, dataset_type=dataset_type))
        total_params = count_parameters(model)

        cuda.empty_cache()
        start_time = time.time()
        metrics = make_experiment(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            device='cuda:0'
        )
        elapsed = time.time() - start_time

        max_acc = metrics.get('test_accuracy', None)

        results.append({
            'model_width': name,
            'dataset': dataset_type,
            'w1': w1,
            'w2': w2,
            'w3': w3,
            'train_time_s': elapsed,
            'max_test_accuracy': max_acc,
            'num_parameters': total_params
        })

df = pd.DataFrame(results)
csv_path = 'results/width_experiments/width_comparison_summary.csv'
df.to_csv(csv_path, index=False)
logging.info(f'Сводный CSV сохранён: {csv_path}')
print(f'Сводные данные о времени и максимальной точности сохранены в {csv_path}')

# 2.2 Оптимизация архитектуры
print(f'{"-" * 70}\n2.2 Оптимизация архитектуры\n{"-" * 70}')
logging.info('Grid search по архитектурам')

width_values = [64, 128, 256, 512, 1024]

patterns = {
    "expanding": lambda w: w[0] < w[1] < w[2],
    "contracting": lambda w: w[0] > w[1] > w[2],
    "constant": lambda w: w[0] == w[1] == w[2],
}

results = []

for dataset_type in ("mnist", "cifar10"):
    if dataset_type == "mnist":
        train_loader, test_loader = get_mnist_loaders(batch_size=512)
    else:
        train_loader, test_loader = get_cifar_loaders(batch_size=512)

    for pattern_name, pattern_fn in patterns.items():
        logging.info(f'Pattern: {pattern_name}, Dataset: {dataset_type}')

        for w1, w2, w3 in itertools.product(width_values, repeat=3):
            if not pattern_fn((w1, w2, w3)):
                continue
            logging.info(f'w1: {w1}, w2: {w2}, w3: {w3}')
            cfg = {
                "num_classes": 10,
                "layers": [
                    {"type": "linear", "size": w1},
                    {"type": "relu"},
                    {"type": "linear", "size": w2},
                    {"type": "relu"},
                    {"type": "linear", "size": w3},
                    {"type": "relu"},
                ]
            }

            model = FullyConnectedModel(**fix_config(cfg, dataset_type=dataset_type))
            torch.cuda.empty_cache()
            start_time = time.time()

            metrics = make_experiment(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=5,
                device='cuda:0',
            )
            elapsed = time.time() - start_time

            num_params = sum(p.numel() for p in model.parameters())

            results.append({
                "dataset": dataset_type,
                "pattern": pattern_name,
                "w1": w1,
                "w2": w2,
                "w3": w3,
                "accuracy": metrics["test_accuracy"],
                "train_time_s": elapsed,
                "num_params": num_params
            })

df = pd.DataFrame(results)
df.to_csv("results/width_experiments/grid_search_results.csv", index=False)
logging.info('Grid search завершён, результаты сохранены в results/grid_search_results.csv')

# Построение 3D heatmap
for dataset_type in df["dataset"].unique():
    for pattern_name in patterns:
        sub = df[(df["dataset"] == dataset_type) & (df["pattern"] == pattern_name)]
        if sub.empty:
            continue

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        xs = sub["w1"].values
        ys = sub["w2"].values
        zs = sub["w3"].values
        cs = sub["accuracy"].values

        sc = ax.scatter(xs, ys, zs, c=cs, marker="o")

        for x, y, z in zip(xs, ys, zs):
            ax.plot([x, x], [y, y], [0, z], color="gray", linewidth=0.5, linestyle="--")
            ax.plot([x, x], [0, y], [z, z], color="gray", linewidth=0.5, linestyle="--")
            ax.plot([0, x], [y, y], [z, z], color="gray", linewidth=0.5, linestyle="--")

        for x, y, z in zip(xs, ys, zs):
            ax.text(
                x, y, z,
                f"({int(x)},{int(y)},{int(z)})",
                size=8,
                zorder=1,
                color="black"
            )

        ax.set_title(f"{dataset_type.upper()} — {pattern_name}")
        ax.set_xlabel("w1")
        ax.set_ylabel("w2")
        ax.set_zlabel("w3")
        fig.colorbar(sc, label="Accuracy")
        plt.tight_layout()

        path = f"plots/grid_search/3dscatter_{dataset_type}_{pattern_name}_annotated.png"
        plt.savefig(path)
        plt.close()
