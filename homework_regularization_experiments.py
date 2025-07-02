import logging
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from utils.datasets_utils import get_mnist_loaders, get_cifar_loaders
from utils.experiment_utils import plot_training_history
from utils.model_utils import run_epoch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATASET = 'mnist'
BATCH_SIZE = 512
EPOCHS = 10
DEVICE = 'cuda:0'
BASE_LAYERS = [256, 128, 64]

RESULT_DIR = 'results/regularization_experiments'
PLOT_DIR = os.path.join('plots/reg/')
os.makedirs(PLOT_DIR, exist_ok=True)

if DATASET == 'mnist':
    train_loader, test_loader = get_mnist_loaders(batch_size=BATCH_SIZE)
else:
    train_loader, test_loader = get_cifar_loaders(batch_size=BATCH_SIZE)


def build_model(with_dropout: float = 0.0, with_batchnorm: bool = False) -> nn.Sequential:
    """
    Собирает многослойную полносвязную модель с опциональной регуляризацией.
    :param with_dropout: Уровень dropout (0.0 означает отсутствие dropout).
    :param with_batchnorm: Использовать ли BatchNorm после каждого слоя.
    :return: Модель (nn.Sequential)
    """
    layers = [nn.Flatten()]
    in_features = train_loader.dataset[0][0].numel()
    dropout_counter = 0
    for size in BASE_LAYERS:
        layers.append(nn.Linear(in_features, size))
        if with_batchnorm:
            bn = nn.BatchNorm1d(size)
            bn.bn_id = dropout_counter
            layers.append(bn)
        layers.append(nn.ReLU())
        if with_dropout > 0.0:
            layers.append(nn.Dropout(p=with_dropout))
            layers[-1].dropout_id = dropout_counter
            dropout_counter += 1
        in_features = size
    layers.append(nn.Linear(in_features, 10))
    return nn.Sequential(*layers)


def experiment_variants() -> None:
    """
    Запускает серию экспериментов для сравнения техник регуляризации
    Результаты сохраняются в csv и графики в директорию plots/reg/.
    :return: None
    """
    variants = [
        {'name': 'no_reg', 'dropout': 0.0, 'batchnorm': False, 'weight_decay': 0.0},
        {'name': 'dropout_0.1', 'dropout': 0.1, 'batchnorm': False, 'weight_decay': 0.0},
        {'name': 'dropout_0.3', 'dropout': 0.3, 'batchnorm': False, 'weight_decay': 0.0},
        {'name': 'dropout_0.5', 'dropout': 0.5, 'batchnorm': False, 'weight_decay': 0.0},
        {'name': 'batchnorm', 'dropout': 0.0, 'batchnorm': True,  'weight_decay': 0.0},
        {'name': 'drop_bn', 'dropout': 0.5, 'batchnorm': True,  'weight_decay': 0.0},
        {'name': 'l2_wd', 'dropout': 0.0, 'batchnorm': False, 'weight_decay': 1e-4},
    ]

    results = []
    for var in variants:
        logging.info(f"Variant: {var['name']}")
        model = build_model(with_dropout=var['dropout'], with_batchnorm=var['batchnorm'])
        model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=var['weight_decay'])
        criterion = nn.CrossEntropyLoss()

        history = {'train_losses': [], 'train_accs': [], 'test_losses': [], 'test_accs': []}
        start = time.time()
        for epoch in range(EPOCHS):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, DEVICE, is_test=False)
            test_loss, test_acc = run_epoch(model, test_loader, criterion, None, DEVICE, is_test=True)
            history['train_losses'].append(train_loss)
            history['train_accs'].append(train_acc)
            history['test_losses'].append(test_loss)
            history['test_accs'].append(test_acc)
            logging.info(f"Epoch {epoch+1}/{EPOCHS} - {var['name']} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        elapsed = time.time() - start

        final_acc = history['test_accs'][-1]
        stability = float(pd.Series(history['test_accs']).std())

        plot_training_history(history, save_path=os.path.join(PLOT_DIR, f"{var['name']}_history.png"))

        weights = torch.cat([p.view(-1).cpu() for p in model.parameters()])
        plt.figure()
        plt.hist(weights.detach().numpy(), bins=50)
        plt.title(f"Weights Distribution: {var['name']}")
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(PLOT_DIR, f"{var['name']}_weights.png"))
        plt.close()

        results.append({
            'variant': var['name'],
            'final_test_acc': final_acc,
            'stability_std': stability,
            'train_time_s': elapsed
        })

    df = pd.DataFrame(results)
    csv_path = os.path.join(RESULT_DIR, 'regularization_comparison.csv')
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved comparison CSV: {csv_path}")


def experiment_adaptive() -> None:
    """
    Проводит эксперимент с адаптивным изменением Dropout probability
    и BatchNorm momentum от эпохи к эпохе.
    Сохраняет график точностей в директорию plots/reg/.
    :return: None
    """
    model = build_model(with_dropout=0.0, with_batchnorm=True)
    history = {'train_accs': [], 'test_accs': []}

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.to(DEVICE)

    for epoch in range(EPOCHS):
        p = 0.5 - 0.4 * (epoch / (EPOCHS - 1))
        mom = 0.1 + 0.8 * (epoch / (EPOCHS - 1))
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = p
            if isinstance(m, nn.BatchNorm1d):
                m.momentum = mom

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, DEVICE, is_test=False)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, DEVICE, is_test=True)
        history['train_accs'].append(train_acc)
        history['test_accs'].append(test_acc)
        logging.info(f"Epoch {epoch+1}: p={p:.3f}, momentum={mom:.3f}, Test Acc={test_acc:.4f}")

    plt.figure()
    plt.plot(history['train_accs'], label='train_acc')
    plt.plot(history['test_accs'], label='test_acc')
    plt.title('Adaptive Regularization Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, 'adaptive_regularization.png'))
    plt.close()
    logging.info('Saved adaptive regularization plot')


def experiment_adaptive_layers() -> None:
    """
    Проводит адаптивный эксперимент, где dropout probability и momentum
    batch normalization изменяются индивидуально для каждого слоя.
    Сохраняет графики изменения параметров и распределений весов слоёв.
    :return: None
    """
    model = build_model(with_dropout=0.5, with_batchnorm=True)
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    p_history = {0: [], 1: [], 2: []}
    mom_history = {0: [], 1: [], 2: []}

    history = {'train_accs': [], 'test_accs': []}

    for epoch in range(EPOCHS):
        p_values = {
            0: 0.5 - 0.4 * (epoch / (EPOCHS - 1)),
            1: 0.3 - 0.25 * (epoch / (EPOCHS - 1)),
            2: 0.2 - 0.19 * (epoch / (EPOCHS - 1)),
        }
        mom_values = {
            0: 0.1 + 0.8 * (epoch / (EPOCHS - 1)),
            1: 0.2 + 0.5 * (epoch / (EPOCHS - 1)),
            2: 0.3 + 0.2 * (epoch / (EPOCHS - 1)),
        }

        for m in model.modules():
            if isinstance(m, nn.Dropout) and hasattr(m, "dropout_id"):
                layer_id = m.dropout_id
                m.p = max(p_values[layer_id], 0.01)
                p_history[layer_id].append(m.p)

            if isinstance(m, nn.BatchNorm1d) and hasattr(m, "bn_id"):
                layer_id = m.bn_id
                m.momentum = mom_values[layer_id]
                mom_history[layer_id].append(m.momentum)

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, DEVICE, is_test=False)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, DEVICE, is_test=True)
        history['train_accs'].append(train_acc)
        history['test_accs'].append(test_acc)

        logging.info(f"Epoch {epoch+1}: Test Acc={test_acc:.4f}")

    for k, values in p_history.items():
        plt.plot(values, label=f"Dropout Layer {k}")
    plt.title('Dropout p per Layer')
    plt.xlabel('Epoch')
    plt.ylabel('Dropout p')
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, 'adaptive_dropout_layers.png'))
    plt.close()

    for k, values in mom_history.items():
        plt.plot(values, label=f"BatchNorm Layer {k}")
    plt.title('BatchNorm Momentum per Layer')
    plt.xlabel('Epoch')
    plt.ylabel('Momentum')
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, 'adaptive_batchnorm_layers.png'))
    plt.close()

    plt.figure()
    plt.plot(history['train_accs'], label='train_acc')
    plt.plot(history['test_accs'], label='test_acc')
    plt.title('Adaptive Regularization Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, 'adaptive_regularization_layers.png'))
    plt.close()

    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            weights = layer.weight.detach().cpu().view(-1).numpy()
            plt.hist(weights, bins=50)
            plt.title(f"Linear Layer {i} weights distribution")
            plt.savefig(os.path.join(PLOT_DIR, f"layer_{i}_weights_hist.png"))
            plt.close()

    logging.info('Finished adaptive layer-wise experiment.')


if __name__ == '__main__':
    # 3.1 Сравнение техник регуляризации
    print(f'{"-" * 70}\n3.1 Сравнение техник регуляризации\n{"-" * 70}')
    experiment_variants()
    # 3.2 Адаптивная регуляризация
    print(f'{"-" * 70}\n3.2 Адаптивная регуляризация\n{"-" * 70}')
    experiment_adaptive()
    experiment_adaptive_layers()
