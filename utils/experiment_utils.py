import os
from typing import Optional, Dict, Any

from torch.utils.data import DataLoader

from utils.model_utils import FullyConnectedModel, train_model, save_model
from utils.visualization_utils import plot_training_history


def make_experiment(
        model: FullyConnectedModel,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 10,
        device: str = 'cpu',
        save_path_graphic: Optional[str] = None,
        save_path_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Запускает эксперимент: обучение модели, тестирование и сохранение результатов.
    :param model: Инициализированная модель.
    :param train_loader: DataLoader для обучающей выборки.
    :param test_loader: DataLoader для тестовой выборки.
    :param epochs: Количество эпох обучения. По умолчанию 10.
    :param device: Устройство для обучения (например, 'cpu' или 'cuda:0').
    :param save_path_graphic: Путь для сохранения графика обучения. По умолчанию None.
    :param save_path_model: Путь для сохранения весов модели. По умолчанию None.
    :return: Словарь с метриками обучения и тестирования.
    """
    history = train_model(model, train_loader, test_loader, epochs=epochs, device=device)
    if save_path_model is not None:
        os.makedirs(os.path.dirname(save_path_model), exist_ok=True)
        save_model(save_path_model, model)
    if save_path_graphic is not None:
        plot_training_history(history, save_path=save_path_graphic)
    result_metrics = {
        'train_losses': history['train_losses'],
        'train_accs': history['train_accs'],
        'test_losses': history['test_losses'],
        'test_accs': history['test_accs'],
        'train_accuracy': history['train_accs'][-1],
        'test_accuracy': history['test_accs'][-1],
    }
    return result_metrics


def fix_config(config: Dict[str, Any], dataset_type: str) -> Dict[str, Any]:
    """
    Исправляет конфигурацию модели в зависимости от выбранного датасета.
    :param config: Исходная конфигурация модели.
    :param dataset_type: Тип датасета ('mnist' или 'cifar10').
    :return: Изменённая конфигурация модели.
    """
    if dataset_type == 'cifar10':
        config['input_size'] = 32 * 32 * 3
    elif dataset_type == 'mnist':
        config['input_size'] = 28 * 28
    return config


def make_reg_config(base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Создаёт конфигурацию модели с добавлением регуляризации (BatchNorm и Dropout) после каждого полносвязного слоя.
    :param base_cfg: Исходная конфигурация модели.
    :return: Новая конфигурация модели с регуляризаторами.
    """
    layers = []
    for spec in base_cfg['layers']:
        if spec['type'] == 'linear':
            layers.append(spec)
            layers.append({"type": "batch_norm"})
            layers.append({"type": "dropout", "rate": 0.5})
        else:
            layers.append(spec)
    return {"num_classes": base_cfg["num_classes"], "layers": layers}
