# DLAI3
## Домашнее задание к уроку 3: Полносвязные сети
### Задание 1: Эксперименты с глубиной сети
#### 1.1 Сравнение моделей разной глубины
Мною были созданы и обучены 4 нейросети со слоями следующих размерностей:
 - (input_size)
 - (input_size, 256)
 - (input_size, 256, 128)
 - (input_size, 512, 256, 128, 128)
 - (input_size, 512, 256, 256, 128, 128, 64)

Модели были обучены на двух следующих датасетах:
 - mnist
 - cifar10

Для каждой модели были сохранены кривые обучения, которые можно найти в папке plots/depth_experiments/.

Для mnist время обучения оставалось примерно константным для всех моделей, 10-11 секунд на одну эпоху. Обучение происходило на GPU NVIDIA GeForce GTX 1650 Ti.

Время обучения для cifar10 немного увеличилось с 9 секунд на однослойной модели до 12 секунд на одну эпоху на 7-слойной модели. Для обучения использовалась всё та же GPU NVIDIA GeForce GTX 1650 Ti.

#### 1.2 Анализ переобучения
(input_size, 512, 256, 256, 128, 128, 64), mnist:

![7layers_mnist.png](plots%2Fdepth_experiments%2F7layers_mnist.png)

На графике представлена кривая обучения на модели с 7 слоями на датасете mnist. Как можно увидеть, переобучения не случилось. В целом, для mnist подходят все обученные до этого модели, качество лишь немного падает до 90% на однослойной.

(input_size, 512, 256, 256, 128, 128, 64), cifar10:

![7layers_cifar.png](plots%2Fdepth_experiments%2F7layers_cifar.png)

Теперь посмотрим на график обучения модели с тем же количеством слоёв, но уже на датасете cifar10. Видим, что модель явно начала переобучаться. На последних эпохах качество на тесте даже начало ухудшаться. Как мне кажется, начало переобучения можно считать с 8-9 эпох, потому что именно на них качество на тесте перестёт расти и начинает падать.

Перейдём к тестированию моделей, дополненных слоями Dropout и BatchNorm.

(input_size, 512, 256, 256, 128, 128, 64), mnist:

![7layers_mnist.png](plots%2Fdepth_experiments%2Freg%2F7layers_mnist.png)

Начнём с рассмотрения графика обучения для mnist. Если сравнить график для mnist, представленный до этого, то можно заметить, что удалось слегка улучшить качество на тесте на поздних эпохах. Переобучения нет.

(input_size, 512, 256, 256, 128, 128, 64), cifar10:

![7layers_cifar.png](plots%2Fdepth_experiments%2Freg%2F7layers_cifar.png)

Как можно увидеть на графике, от переобучения на cifar10 избавиться не удалось. Как мне кажется, проблема в том, что модель просто слишком слабая для данного датасета, либо сам датасет требует предобработки.

### Задание 2: Эксперименты с шириной сети
#### 2.1 Сравнение моделей разной ширины
Мною были созданы модели с различной шириной слоев:
 - Узкие слои: (64, 32, 16)
 - Средние слои: (256, 128, 64)
 - Широкие слои: (1024, 512, 256)
 - Очень широкие слои: (2048, 1024, 512)

| model_width | dataset | w1   | w2   | w3  | train time (s) | max_test_accuracy | num_parameters |
|-------------|---------|------|------|-----|----------------|-------------------|----------------|
| narrow      | mnist   | 64   | 32   | 16  | 133.51         | 0.9583            | 53018          |
| narrow      | cifar10 | 64   | 32   | 16  | 125.50         | 0.4957            |	199450         |
| medium      | mnist   | 256  | 128  | 64  | 125.57         | 0.9792            | 242762         |
| medium      | cifar10 | 256  | 128  | 64  | 129.43         | 0.5379            | 828490         |
| wide        | mnist   | 1024 | 512  | 256 | 155.83         |	0.9757            | 1462538        |
| wide        | cifar10 | 1024 | 512  | 256 | 170.64         | 0.5496            | 3805450        |
| extra_wide  | mnist   | 2048 | 1024 | 512 | 157.15         | 0.9789            | 4235786        |
| extra_wide  | cifar10 | 2048 | 1024 | 512 | 154.92         | 0.5442            | 8921610        |

После обучения моделей на обоих датасетах можно сделать следующие выводы:
 - Увеличение количества параметров дало сильный прирост только с "узкого" до "среднего". После этого точность, как на mnist, так и на cifar10 колебалась в одном небольшом интервале.
 - Время обучения на 10 эпохах значительно увеличилось (на 20 секунд) только при скачке от "среднего" до "широкого" размера слоёв
 - Количество параметров увеличивалось на всех этапах, но большее количество не гарантировало лучшее качество.

#### 2.2 Оптимизация архитектуры
Мною были рассмотрены следующие размеры слоёв:
 - 64
 - 128
 - 256
 - 512
 - 1024

Все эти размеры были перебраны, руководствуясь следующими правилами:
 - w1 < w2 < w3 (расширение)
 - w1 > w2 > w3 (сужение)
 - w1 = w2 = w3 (постоянная)

Графики (heatmap) можно найти по пути plots/grid_search/
Выводы, которые можно сделать по ним:
 - При постоянном количестве весов лучший результат как на mnist, так и на cifar10 получается при (512, 512, 512).

![3dscatter_mnist_constant_annotated.png](plots%2Fgrid_search%2F3dscatter_mnist_constant_annotated.png)

 - При сужении слоёв лучший результат на обоих датасетах получается при наибольших количествах весов (1024, 512, 256). Также для датасета mnist можно выделить точку (1024, 128, 64). Она достаточна близка к максимальной точности.

![3dscatter_mnist_contracting_annotated.png](plots%2Fgrid_search%2F3dscatter_mnist_contracting_annotated.png)

 - При расширении слоёв лучший результат на обоих датасетах опять получается при наибольших количествах весов (256, 512, 1024). Так же для датасета cifar10 можно выделить точку (128, 512, 1024). Она достаточна близка к максимальной точности.

![3dscatter_cifar10_expanding_annotated.png](plots%2Fgrid_search%2F3dscatter_cifar10_expanding_annotated.png)

### Задание 3: Эксперименты с регуляризацией
#### 3.1 Сравнение техник регуляризации
Исследовал различные техники регуляризации (использовал архитектуру (input_size, 256, 128, 64)):
 - Без регуляризации
 - Только Dropout (разные коэффициенты: 0.1, 0.3, 0.5)
 - Только BatchNorm
 - Dropout + BatchNorm
 - L2 регуляризация (weight decay)

| variant     | dataset | final_test_acc | stability_std | train time (s) |
|-------------|---------|----------------|---------------|----------------|
| no_reg      | mnist   | 0.9722         | 0.0137        | 125.54         |
| no_reg      | cifar10 | 0.5396         | 0.0262        | 139.62         |
| dropout_0.1 | mnist   | 0.9806         | 0.0149        | 126.21         |
| dropout_0.1 | cifar10 | 0.5415         | 0.0321        | 141.02         |
| dropout_0.3 | mnist   | 0.9802         | 0.0132        | 126.14         |
| dropout_0.3 | cifar10 | 0.5257         | 0.0341        | 140.46         |
| dropout_0.5 | mnist   | 0.9742         | 0.0145        | 126.98         |
| dropout_0.5 | cifar10 | 0.4805         | 0.0305        | 141.90         |
| batchnorm   | mnist   | 0.9820         | 0.0042        | 135.73         |
| batchnorm   | cifar10 | 0.5484         | 0.0254        | 145.09         |
| drop_bn     | mnist   | 0.9771         | 0.0148        | 131.65         |
| drop_bn     | cifar10 | 0.5004         | 0.0418        | 145.11         |
| l2_wd       | mnist   | 0.9771         | 0.0142        | 127.41         |
| l2_wd       | cifar10 | 0.5315         | 0.0230        | 149.28         |

Топ точности на cifar10 (accuracy):
1. Только BatchNorm (0.5484)
2. Только Dropout(0.1) (0.5415)
3. Dropout(0.5) + BatchNorm (0.5004)

Топ точности на mnist (accuracy):
1. Только BatchNorm (0.9820)
2. Только Dropout(0.1) (0.9806)
3. Только Dropout(0.3) (0.9802)

Топ стабильности обучения на cifar10 (стандартное отклонение accuracy во время обучения)
1. L2 регуляризация (0.0230)
2. Только BatchNorm (0.0254)
3. Без регуляризации (0.0262)

Топ стабильности обучения на mnist (стандартное отклонение accuracy во время обучения)
1. Только BatchNorm (0.0042)
2. Только Dropout(0.3) (0.0132)
3. Без регуляризации (0.0138)

Можно увидеть, что BatchNorm находится в обоих топах по стабильности, что говорит о повышении стабильности обучения при его использовании. Так же в обоих топах находится отсутствие регуляризации.

Ниже представлен график распределения весов для L2 регуляризации на cifar10.

![l2_wd_weights.png](plots%2Freg%2Fcifar10%2Fl2_wd_weights.png)

Ниже представлен график распределения весов для BatchNorm на mnist.

![batchnorm_weights.png](plots%2Freg%2Fmnist%2Fbatchnorm_weights.png)

#### 3.2 Адаптивная регуляризация
Провёл анализ влияния адаптивных техник на различные слои сетей. Ниже представлены графики распрделенения весов на разных слоях для mnist.

![layer_1_weights_hist.png](plots%2Freg%2Fmnist_%2Flayer_1_weights_hist.png)

![layer_5_weights_hist.png](plots%2Freg%2Fmnist_%2Flayer_5_weights_hist.png)

![layer_9_weights_hist.png](plots%2Freg%2Fmnist_%2Flayer_9_weights_hist.png)

![layer_13_weights_hist.png](plots%2Freg%2Fmnist_%2Flayer_13_weights_hist.png)

Как можно увидеть, распределение весов постепенно смещается в левую часть графика и в конце теряет форму нормального распределения, разбиваясь на два "возвышения".
