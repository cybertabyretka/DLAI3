import json
import logging
import os
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FullyConnectedModel(nn.Module):
    """
    Полносвязная нейронная сеть с конфигурацией из JSON или словаря.
    """
    def __init__(
            self,
            config_path: Optional[str] = None,
            input_size: Optional[int] = None,
            num_classes: Optional[int] = None,
            **kwargs
    ) -> None:
        """
        Инициализация FullyConnectedModel.
        :param config_path: Путь к JSON-файлу с конфигурацией модели. Если None, используется kwargs.
        :param input_size: Размер входного вектора. По умолчанию 28*28 (MNIST).
        :param num_classes: Количество классов для классификации. По умолчанию 10.
        :param kwargs: Конфигурация слоёв, если не используется config_path.
        """
        super().__init__()

        if config_path:
            self.config = self.load_config(config_path)
        else:
            self.config = kwargs

        self.input_size = input_size or self.config.get('input_size', 28 * 28)
        self.num_classes = num_classes or self.config.get('num_classes', 10)

        self.layers = self._build_layers()

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Загружает конфигурацию модели из JSON-файла.
        :param config_path: Путь к JSON-файлу.
        :return: Словарь с конфигурацией модели.
        """
        with open(config_path, 'r') as f:
            return json.load(f)

    def _build_layers(self) -> nn.Sequential:
        """
        Создаёт последовательность слоёв модели на основе конфигурации.
        :return: Секвенциальная модель (nn.Sequential).
        """
        layers = []
        prev_size = self.input_size

        layer_config = self.config.get('layers', [])

        for layer_spec in layer_config:
            layer_type = layer_spec['type']

            if layer_type == 'linear':
                out_size = layer_spec['size']
                layers.append(nn.Linear(prev_size, out_size))
                prev_size = out_size

            elif layer_type == 'relu':
                layers.append(nn.ReLU())

            elif layer_type == 'sigmoid':
                layers.append(nn.Sigmoid())

            elif layer_type == 'tanh':
                layers.append(nn.Tanh())

            elif layer_type == 'dropout':
                rate = layer_spec.get('rate', 0.5)
                layers.append(nn.Dropout(rate))

            elif layer_type == 'batch_norm':
                layers.append(nn.BatchNorm1d(prev_size))

            elif layer_type == 'layer_norm':
                layers.append(nn.LayerNorm(prev_size))

        layers.append(nn.Linear(prev_size, self.num_classes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход по модели.
        :param x: Входной тензор размерности (batch_size, …).
        :return: Выходной тензор размерности (batch_size, num_classes).
        """
        x = x.view(x.size(0), -1)
        return self.layers(x)


def run_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        device: str = 'cuda:0',
        is_test: bool = False
) -> Tuple[float, float]:
    """
    Запускает одну эпоху обучения или тестирования.
    :param model: Модель для обучения или теста.
    :param data_loader: DataLoader для прохода по данным.
    :param criterion: Функция потерь.
    :param optimizer: Оптимизатор. Если None, выполняется только тест.
    :param device: Устройство для вычислений.
    :param is_test: Флаг тестового режима.
    :return: Среднее значение loss и accuracy за эпоху.
    """
    if is_test:
        model.eval()
    else:
        model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)

        if not is_test and optimizer is not None:
            optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        if not is_test and optimizer is not None:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    return total_loss / len(data_loader), correct / total


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 10,
        lr: float = 0.001,
        device: str = 'cuda:0'
) -> Dict[str, Any]:
    """
    Обучает модель и возвращает историю обучения.
    :param model: Модель для обучения.
    :param train_loader: DataLoader для обучающей выборки.
    :param test_loader: DataLoader для тестовой выборки.
    :param epochs: Количество эпох обучения.
    :param lr: Скорость обучения.
    :param device: Устройство для обучения.
    :return: Словарь с историей значений loss и accuracy.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_test=False)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_test=True)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        logging.info(
            f'Epoch {epoch + 1}/{epochs}:'
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}'
            f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }


def save_model(path: str, model: torch.nn.Module) -> None:
    """
    Сохраняет веса модели в файл.
    :param path: Путь к файлу для сохранения.
    :param model: Обученная модель.
    """
    state_dict = {
        'model': model.state_dict(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)


def load_model(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[int, float, float]:
    """
    Загружает веса модели и состояние оптимизатора.
    :param path: Путь к сохранённому файлу.
    :param model: Экземпляр модели для загрузки весов.
    :param optimizer: Экземпляр оптимизатора для загрузки состояния.
    :return: Кортеж (номер эпохи, лучшая тестовая ошибка, лучшая тестовая точность).
    """
    state_dict = torch.load(path)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    return state_dict['epoch'], state_dict['best_test_loss'], state_dict['best_test_acc']


def count_parameters(model: nn.Module) -> int:
    """
    Считает количество обучаемых параметров модели.
    :param model: Модель.
    :return: Количество параметров.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
