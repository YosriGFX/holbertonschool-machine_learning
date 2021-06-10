# 0x12. Transformer Applications
### 0. Dataset
```
$ cat 0-main.py
#!/usr/bin/env python3

Dataset = __import__('0-dataset').Dataset
import tensorflow as tf

data = Dataset()
for pt, en in data.data_train.take(1):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))
for pt, en in data.data_valid.take(1):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))
print(type(data.tokenizer_pt))
print(type(data.tokenizer_en))
$ ./0-main.py
e quando melhoramos a procura , tiramos a única vantagem da impressão , que é a serendipidade .
and when you improve searchability , you actually take away the one advantage of print , which is serendipity .
tinham comido peixe com batatas fritas ?
did they eat fish and chips ?
<class 'tensorflow_datasets.core.deprecated.text.subword_text_encoder.SubwordTextEncoder'>
<class 'tensorflow_datasets.core.deprecated.text.subword_text_encoder.SubwordTextEncoder'>
$

```

### 1. Encode Tokens
```
$ cat 1-main.py
#!/usr/bin/env python3

Dataset = __import__('1-dataset').Dataset
import tensorflow as tf

data = Dataset()
for pt, en in data.data_train.take(1):
    print(data.encode(pt, en))
for pt, en in data.data_valid.take(1):
    print(data.encode(pt, en))
$ ./1-main.py
([30138, 6, 36, 17925, 13, 3, 3037, 1, 4880, 3, 387, 2832, 18, 18444, 1, 5, 8, 3, 16679, 19460, 739, 2, 30139], [28543, 4, 56, 15, 1266, 20397, 10721, 1, 15, 100, 125, 352, 3, 45, 3066, 6, 8004, 1, 88, 13, 14859, 2, 28544])
([30138, 289, 15409, 2591, 19, 20318, 26024, 29997, 28, 30139], [28543, 93, 25, 907, 1366, 4, 5742, 33, 28544])
$

```

### 2. TF Encode
```
$ cat 2-main.py
#!/usr/bin/env python3

Dataset = __import__('2-dataset').Dataset
import tensorflow as tf

data = Dataset()
print('got here')
for pt, en in data.data_train.take(1):
    print(pt, en)
for pt, en in data.data_valid.take(1):
    print(pt, en)
$ ./2-main.py
tf.Tensor(
[30138     6    36 17925    13     3  3037     1  4880     3   387  2832
    18 18444     1     5     8     3 16679 19460   739     2 30139], shape=(23,), dtype=int64) tf.Tensor(
[28543     4    56    15  1266 20397 10721     1    15   100   125   352
     3    45  3066     6  8004     1    88    13 14859     2 28544], shape=(23,), dtype=int64)
tf.Tensor([30138   289 15409  2591    19 20318 26024 29997    28 30139], shape=(10,), dtype=int64) tf.Tensor([28543    93    25   907  1366     4  5742    33 28544], shape=(9,), dtype=int64)
$

```

### 3. Pipeline
```
$ cat 3-main.py
#!/usr/bin/env python3

Dataset = __import__('3-dataset').Dataset
import tensorflow as tf

tf.compat.v1.set_random_seed(0)
data = Dataset(32, 40)
for pt, en in data.data_train.take(1):
    print(pt, en)
for pt, en in data.data_valid.take(1):
    print(pt, en)
$ ./3-main.py
tf.Tensor(
[[30138  1029   104 ...     0     0     0]
 [30138    40     8 ...     0     0     0]
 [30138    12    14 ...     0     0     0]
 ...
 [30138    72 23483 ...     0     0     0]
 [30138  2381   420 ...     0     0     0]
 [30138     7 14093 ...     0     0     0]], shape=(32, 39), dtype=int64) tf.Tensor(
[[28543   831   142 ...     0     0     0]
 [28543    16    13 ...     0     0     0]
 [28543    19     8 ...     0     0     0]
 ...
 [28543    18    27 ...     0     0     0]
 [28543  2648   114 ... 28544     0     0]
 [28543  9100 19214 ...     0     0     0]], shape=(32, 37), dtype=int64)
tf.Tensor(
[[30138   289 15409 ...     0     0     0]
 [30138    86   168 ...     0     0     0]
 [30138  5036     9 ...     0     0     0]
 ...
 [30138  1157 29927 ...     0     0     0]
 [30138    33   837 ...     0     0     0]
 [30138   126  3308 ...     0     0     0]], shape=(32, 32), dtype=int64) tf.Tensor(
[[28543    93    25 ...     0     0     0]
 [28543    11    20 ...     0     0     0]
 [28543    11  2850 ...     0     0     0]
 ...
 [28543    11   406 ...     0     0     0]
 [28543     9   152 ...     0     0     0]
 [28543     4   272 ...     0     0     0]], shape=(32, 35), dtype=int64)
$

```

### 4. Create Masks

```
$ cat 4-main.py
#!/usr/bin/env python3

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
import tensorflow as tf

tf.compat.v1.set_random_seed(0)
data = Dataset(32, 40)
for inputs, target in data.data_train.take(1):
    print(create_masks(inputs, target))
$ ./4-main.py
(<tf.Tensor: shape=(32, 1, 1, 39), dtype=float32, numpy=
array([[[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       ...,


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],

       [[[0., 0., 0., ..., 1., 1., 1.]]]], dtype=float32)>, <tf.Tensor: shape=(32, 1, 37, 37), dtype=float32, numpy=
array([[[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.]]],
          ...,


       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 0., 1., 1.],
         [0., 0., 0., ..., 0., 1., 1.],
         [0., 0., 0., ..., 0., 1., 1.]]],


       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.]]]], dtype=float32)>, <tf.Tensor: shape=(32, 1, 1, 39), dtype=float32, numpy=
array([[[[0., 0., 0., ..., 1., 1., 1.]]],
 [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       ...,


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]]], dtype=float32)>)
$

```

### 5. Train

```
$ cat 5-main.py
#!/usr/bin/env python3
import tensorflow as tf
train_transformer = __import__('5-train').train_transformer

tf.compat.v1.set_random_seed(0)
transformer = train_transformer(4, 128, 8, 512, 32, 40, 2)
print(type(transformer))
$ ./5-main.py
Epoch 1, batch 0: loss 10.26855754852295 accuracy 0.0
Epoch 1, batch 50: loss 10.23129940032959 accuracy 0.0009087905054911971

...

Epoch 1, batch 1000: loss 7.164522647857666 accuracy 0.06743457913398743
Epoch 1, batch 1050: loss 7.076988220214844 accuracy 0.07054812461137772
Epoch 1: loss 7.038494110107422 accuracy 0.07192815840244293
Epoch 2, batch 0: loss 5.177524089813232 accuracy 0.1298387050628662
Epoch 2, batch 50: loss 5.189461708068848 accuracy 0.14023463428020477

...

Epoch 2, batch 1000: loss 4.870367527008057 accuracy 0.15659810602664948
Epoch 2, batch 1050: loss 4.858142375946045 accuracy 0.15731287002563477
Epoch 2: loss 4.852652549743652 accuracy 0.15768977999687195
<class '5-transformer.Transformer'>
$

```

> Copyright © 2021 Holberton School. All rights reserved.

![Yosri Ghorbel](https://pbs.twimg.com/media/E3YEO7kXwAU9x6x?format=png&name=4096x4096)
