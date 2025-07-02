import os
from typing import Optional, Dict, List

import matplotlib.pyplot as plt


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """
    Строит графики обучения модели по истории значений loss и accuracy.

    :param history: Словарь с историей обучения. Должен содержать ключи:
     - **'train_losses'**: список значений потерь на обучении по эпохам
     - **'test_losses'**: список значений потерь на тесте по эпохам
     - **'train_accs'**: список значений точности на обучении по эпохам
     - **'test_accs'**: список значений точности на тесте по эпохам
    :param save_path: Путь для сохранения графика в файл. Если None, график отображается на экране.
    :return: None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()

    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
    else:
        plt.show()
