import logging

import torch.cuda as cuda

from utils.datasets_utils import get_cifar_loaders, get_mnist_loaders
from utils.experiment_utils import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f'CUDA доступна: {cuda.is_available()}')
logging.info(f'Количество GPU: {cuda.device_count()}')
logging.info(f'Текущее устройство: {cuda.current_device()}')
logging.info(f'Название GPU: {cuda.get_device_name(0)}')

# 1.1 Сравнение моделей разной глубины
print(f'{"-" * 70}\n1.1 Сравнение моделей разной глубины\n{"-" * 70}')
config_1_layer = {
    "num_classes": 10,
    "layers": []
}
config_2_layers = {
    "num_classes": 10,
    "layers": [
        {"type": "linear", "size": 256},
        {"type": "relu"}
    ]
}
config_3_layers = {
    "num_classes": 10,
    "layers": [
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "linear", "size": 128},
        {"type": "relu"}
    ]
}
config_5_layers = {
    "num_classes": 10,
    "layers": [
        {"type": "linear", "size": 512},
        {"type": "relu"},
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "linear", "size": 128},
        {"type": "relu"},
        {"type": "linear", "size": 128},
        {"type": "relu"},
    ]
}
config_7_layers = {
    "num_classes": 10,
    "layers": [
        {"type": "linear", "size": 512},
        {"type": "relu"},
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "linear", "size": 128},
        {"type": "relu"},
        {"type": "linear", "size": 128},
        {"type": "relu"},
        {"type": "linear", "size": 64},
        {"type": "relu"},
    ]
}
model_1_layer_mnist = FullyConnectedModel(**fix_config(config_1_layer, dataset_type='mnist'))
model_2_layers_mnist = FullyConnectedModel(**fix_config(config_2_layers, dataset_type='mnist'))
model_3_layers_mnist = FullyConnectedModel(**fix_config(config_3_layers, dataset_type='mnist'))
model_5_layers_mnist = FullyConnectedModel(**fix_config(config_5_layers, dataset_type='mnist'))
model_7_layers_mnist = FullyConnectedModel(**fix_config(config_7_layers, dataset_type='mnist'))

model_1_layer_cifar = FullyConnectedModel(**fix_config(config_1_layer, dataset_type='cifar10'))
model_2_layers_cifar = FullyConnectedModel(**fix_config(config_2_layers, dataset_type='cifar10'))
model_3_layers_cifar = FullyConnectedModel(**fix_config(config_3_layers, dataset_type='cifar10'))
model_5_layers_cifar = FullyConnectedModel(**fix_config(config_5_layers, dataset_type='cifar10'))
model_7_layers_cifar = FullyConnectedModel(**fix_config(config_7_layers, dataset_type='cifar10'))

mnist_train, mnist_test = get_mnist_loaders(batch_size=1024)
cifar_train, cifar_test = get_cifar_loaders(batch_size=1024)

mnist_epochs = 10
cifar_epochs = 10

logging.info('Начаты эксперименты с глубиной моделей')

print(f'{"-" * 70}\nЭксперименты на mnist\n{"-" * 70}')

logging.info('Глубина: 1, Датасет: mnist')
make_experiment(
    model=model_1_layer_mnist,
    train_loader=mnist_train,
    test_loader=mnist_test,
    epochs=mnist_epochs,
    device='cuda:0',
    save_path_graphic='plots/depth_experiments/1layer_mnist.png'
)

logging.info('Глубина: 2, Датасет: mnist')
make_experiment(
    model=model_2_layers_mnist,
    train_loader=mnist_train,
    test_loader=mnist_test,
    epochs=mnist_epochs,
    device='cuda:0',
    save_path_graphic='plots/depth_experiments/2layers_mnist.png'
)

logging.info('Глубина: 3, Датасет: mnist')
make_experiment(
    model=model_3_layers_mnist,
    train_loader=mnist_train,
    test_loader=mnist_test,
    epochs=mnist_epochs,
    device='cuda:0',
    save_path_graphic='plots/depth_experiments/3layers_mnist.png'
)

logging.info('Глубина: 5, Датасет: mnist')
make_experiment(
    model=model_5_layers_mnist,
    train_loader=mnist_train,
    test_loader=mnist_test,
    epochs=mnist_epochs,
    device='cuda:0',
    save_path_graphic='plots/depth_experiments/5layers_mnist.png'
)

logging.info('Глубина: 7, Датасет: mnist')
make_experiment(
    model=model_7_layers_mnist,
    train_loader=mnist_train,
    test_loader=mnist_test,
    epochs=mnist_epochs,
    device='cuda:0',
    save_path_graphic='plots/depth_experiments/7layers_mnist.png'
)

print(f'{"-" * 70}\nЭксперименты на cifar10\n{"-" * 70}')

logging.info('Глубина: 1, Датасет: cifar')
make_experiment(
    model=model_1_layer_cifar,
    train_loader=cifar_train,
    test_loader=cifar_test,
    epochs=cifar_epochs,
    device='cuda:0',
    save_path_graphic='plots/depth_experiments/1layer_cifar.png'
)

logging.info('Глубина: 2, Датасет: cifar')
make_experiment(
    model=model_2_layers_cifar,
    train_loader=cifar_train,
    test_loader=cifar_test,
    epochs=cifar_epochs,
    device='cuda:0',
    save_path_graphic='plots/depth_experiments/2layers_cifar.png'
)

logging.info('Глубина: 3, Датасет: cifar')
make_experiment(
    model=model_3_layers_cifar,
    train_loader=cifar_train,
    test_loader=cifar_test,
    epochs=cifar_epochs,
    device='cuda:0',
    save_path_graphic='plots/depth_experiments/3layers_cifar.png'
)

logging.info('Глубина: 5, Датасет: cifar')
make_experiment(
    model=model_5_layers_cifar,
    train_loader=cifar_train,
    test_loader=cifar_test,
    epochs=cifar_epochs,
    device='cuda:0',
    save_path_graphic='plots/depth_experiments/5layers_cifar.png'
)

logging.info('Глубина: 7, Датасет: cifar')
make_experiment(
    model=model_7_layers_cifar,
    train_loader=cifar_train,
    test_loader=cifar_test,
    epochs=cifar_epochs,
    device='cuda:0',
    save_path_graphic='plots/depth_experiments/7layers_cifar.png'
)

# 1.2 Анализ переобучения
print(f'{"-" * 70}\n1.2 Анализ переобучения\n{"-" * 70}')
config_1_layer = make_reg_config(config_1_layer)
config_2_layers = make_reg_config(config_2_layers)
config_3_layers = make_reg_config(config_3_layers)
config_5_layers = make_reg_config(config_5_layers)
config_7_layers = make_reg_config(config_7_layers)

print(f'{"-" * 70}\nЭксперименты на mnist\n{"-" * 70}')

logging.info('Глубина: 1, Датасет: mnist')
make_experiment(
    model=model_1_layer_mnist,
    train_loader=mnist_train,
    test_loader=mnist_test,
    epochs=mnist_epochs,
    device='cuda:0',
    save_path_graphic='plots/depth_experiments/reg/1layer_mnist.png'
)

logging.info('Глубина: 2, Датасет: mnist')
make_experiment(
    model=model_2_layers_mnist,
    train_loader=mnist_train,
    test_loader=mnist_test,
    epochs=mnist_epochs,
    device='cuda:0',
    save_path_graphic='plots/depth_experiments/reg/2layers_mnist.png'
)

logging.info('Глубина: 3, Датасет: mnist')
make_experiment(
    model=model_3_layers_mnist,
    train_loader=mnist_train,
    test_loader=mnist_test,
    epochs=mnist_epochs,
    device='cuda:0',
    save_path_graphic='plots/depth_experiments/reg/3layers_mnist.png'
)

logging.info('Глубина: 5, Датасет: mnist')
make_experiment(
    model=model_5_layers_mnist,
    train_loader=mnist_train,
    test_loader=mnist_test,
    epochs=mnist_epochs,
    device='cuda:0',
    save_path_graphic='plots/depth_experiments/reg/5layers_mnist.png'
)

logging.info('Глубина: 7, Датасет: mnist')
make_experiment(
    model=model_7_layers_mnist,
    train_loader=mnist_train,
    test_loader=mnist_test,
    epochs=mnist_epochs,
    device='cuda:0',
    save_path_graphic='plots/depth_experiments/reg/7layers_mnist.png'
)

print(f'{"-" * 70}\nЭксперименты на cifar10\n{"-" * 70}')

logging.info('Глубина: 1, Датасет: cifar')
make_experiment(
    model=model_1_layer_cifar,
    train_loader=cifar_train,
    test_loader=cifar_test,
    epochs=cifar_epochs,
    device='cuda:0',
    save_path_graphic='plots/depth_experiments/reg/1layer_cifar.png'
)

logging.info('Глубина: 2, Датасет: cifar')
make_experiment(
    model=model_2_layers_cifar,
    train_loader=cifar_train,
    test_loader=cifar_test,
    epochs=cifar_epochs,
    device='cuda:0',
    save_path_graphic='plots/depth_experiments/reg/2layers_cifar.png'
)

logging.info('Глубина: 3, Датасет: cifar')
make_experiment(
    model=model_3_layers_cifar,
    train_loader=cifar_train,
    test_loader=cifar_test,
    epochs=cifar_epochs,
    device='cuda:0',
    save_path_graphic='plots/depth_experiments/reg/3layers_cifar.png'
)

logging.info('Глубина: 5, Датасет: cifar')
make_experiment(
    model=model_5_layers_cifar,
    train_loader=cifar_train,
    test_loader=cifar_test,
    epochs=cifar_epochs,
    device='cuda:0',
    save_path_graphic='plots/depth_experiments/reg/5layers_cifar.png'
)

logging.info('Глубина: 7, Датасет: cifar')
make_experiment(
    model=model_7_layers_cifar,
    train_loader=cifar_train,
    test_loader=cifar_test,
    epochs=cifar_epochs,
    device='cuda:0',
    save_path_graphic='plots/depth_experiments/reg/7layers_cifar.png'
)
