# 0x11. Attention
### 0. RNN Encoder
```
$ cat 0-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
RNNEncoder = __import__('0-rnn_encoder').RNNEncoder

encoder = RNNEncoder(1024, 128, 256, 32)
print(encoder.batch)
print(encoder.units)
print(type(encoder.embedding))
print(type(encoder.gru))

initial = encoder.initialize_hidden_state()
print(initial)
x = tf.convert_to_tensor(np.random.choice(1024, 320).reshape((32, 10)))
outputs, hidden = encoder(x, initial)
print(outputs)
print(hidden)
$ ./0-main.py
32
256
<class 'tensorflow.python.keras.layers.embeddings.Embedding'>
<class 'tensorflow.python.keras.layers.recurrent.GRU'>
Tensor("zeros:0", shape=(32, 256), dtype=float32)
Tensor("rnn_encoder/gru/transpose_1:0", shape=(32, 10, 256), dtype=float32)
Tensor("rnn_encoder/gru/while/Exit_2:0", shape=(32, 256), dtype=float32)
$
```

### 1. Self Attention
```
$ cat 1-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention

attention = SelfAttention(256)
print(attention.W)
print(attention.U)
print(attention.V)
s_prev = tf.convert_to_tensor(np.random.uniform(size=(32, 256)), preferred_dtype='float32')
hidden_states = tf.convert_to_tensor(np.random.uniform(size=(32, 10, 256)), preferred_dtype='float32')
context, weights = attention(s_prev, hidden_states)
print(context)
print(weights)
$ ./1-main.py
<tensorflow.python.keras.layers.core.Dense object at 0x12309d3c8>
<tensorflow.python.keras.layers.core.Dense object at 0xb28536b38>
<tensorflow.python.keras.layers.core.Dense object at 0xb28536e48>
Tensor("self_attention/Sum:0", shape=(32, 256), dtype=float64)
Tensor("self_attention/transpose_1:0", shape=(32, 10, 1), dtype=float64)
$
```

### 2. RNN Decoder
```
$ cat 2-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
RNNDecoder = __import__('2-rnn_decoder').RNNDecoder

decoder = RNNDecoder(2048, 128, 256, 32)
print(decoder.embedding)
print(decoder.gru)
print(decoder.F)
x = tf.convert_to_tensor(np.random.choice(2048, 32).reshape((32, 1)))
s_prev = tf.convert_to_tensor(np.random.uniform(size=(32, 256)).astype('float32'))
hidden_states = tf.convert_to_tensor(np.random.uniform(size=(32, 10, 256)).astype('float32'))
y, s = decoder(x, s_prev, hidden_states)
print(y)
print(s)
$ ./2-main.py
<tensorflow.python.keras.layers.embeddings.Embedding object at 0x1321113c8>
<tensorflow.python.keras.layers.recurrent.GRU object at 0xb375aab00>
<tensorflow.python.keras.layers.core.Dense object at 0xb375d5128>
Tensor("rnn_decoder/dense/BiasAdd:0", shape=(32, 2048), dtype=float32)
Tensor("rnn_decoder/gru/while/Exit_2:0", shape=(32, 256), dtype=float32)
$
```

### 3. Positional Encoding
```
$ cat 4-main.py
#!/usr/bin/env python3

import numpy as np
positional_encoding = __import__('4-positional_encoding').positional_encoding

PE = positional_encoding(30, 512)
print(PE.shape)
print(PE)
$ ./4-main.py
(30, 512)
[[ 0.00000000e+00  1.00000000e+00  0.00000000e+00 ...  1.00000000e+00
   0.00000000e+00  1.00000000e+00]
 [ 8.41470985e-01  5.40302306e-01  8.21856190e-01 ...  9.99999994e-01
   1.03663293e-04  9.99999995e-01]
 [ 9.09297427e-01 -4.16146837e-01  9.36414739e-01 ...  9.99999977e-01
   2.07326584e-04  9.99999979e-01]
 ...
 [ 9.56375928e-01 -2.92138809e-01  7.91416314e-01 ...  9.99995791e-01
   2.79890525e-03  9.99996083e-01]
 [ 2.70905788e-01 -9.62605866e-01  9.53248145e-01 ...  9.99995473e-01
   2.90256812e-03  9.99995788e-01]
 [-6.63633884e-01 -7.48057530e-01  2.94705106e-01 ...  9.99995144e-01
   3.00623096e-03  9.99995481e-01]]
$
```

### 4. Scaled Dot Product Attention
```
$ cat 5-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention

np.random.seed(0)
Q = tf.convert_to_tensor(np.random.uniform(size=(50, 10, 256)).astype('float32'))
K = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype('float32'))
V = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 512)).astype('float32'))
output, weights = sdp_attention(Q, K, V)
print(output)
print(weights)
$ ./5-main.py
Tensor("MatMul_1:0", shape=(50, 10, 512), dtype=float32)
Tensor("Softmax:0", shape=(50, 10, 15), dtype=float32)
$
```

### 5. Multi Head Attention
```
$ cat 6-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention

mha = MultiHeadAttention(512, 8)
print(mha.dm)
print(mha.h)
print(mha.depth)
print(mha.Wq)
print(mha.Wk)
print(mha.Wv)
print(mha.linear)
Q = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype('float32'))
K = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype('float32'))
V = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype('float32'))
output, weights = mha(Q, K, V, None)
print(output)
print(weights)
$ ./6-main.py
512
8
64
<tensorflow.python.keras.layers.core.Dense object at 0xb2c585b38>
<tensorflow.python.keras.layers.core.Dense object at 0xb2c585e48>
<tensorflow.python.keras.layers.core.Dense object at 0xb2c5b1198>
<tensorflow.python.keras.layers.core.Dense object at 0xb2c5b14a8>
Tensor("multi_head_attention/dense_3/BiasAdd:0", shape=(50, 15, 512), dtype=float32)
Tensor("multi_head_attention/Softmax:0", shape=(50, 8, 15, 15), dtype=float32)
$
```

### 6. Transformer Encoder Block
```
$ cat 7-main
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock

eblock = EncoderBlock(512, 8, 2048)
print(eblock.mha)
print(eblock.dense_hidden)
print(eblock.dense_output)
print(eblock.layernorm1)
print(eblock.layernorm2)
print(eblock.dropout1)
print(eblock.dropout2)
x = tf.random.uniform((32, 10, 512))
output = eblock(x, True, None)
print(output)
$ ./7-main.py
<6-multihead_attention.MultiHeadAttention object at 0x12c61b390>
<tensorflow.python.keras.layers.core.Dense object at 0xb31ae1860>
<tensorflow.python.keras.layers.core.Dense object at 0xb31ae1b70>
<tensorflow.python.keras.layers.normalization.LayerNormalization object at 0xb31ae1e80>
<tensorflow.python.keras.layers.normalization.LayerNormalization object at 0xb31aea128>
<tensorflow.python.keras.layers.core.Dropout object at 0xb31aea390>
<tensorflow.python.keras.layers.core.Dropout object at 0xb31aea518>
Tensor("encoder_block/layer_normalization_1/batchnorm/add_1:0", shape=(32, 10, 512), dtype=float32)
$
```

### 7. Transformer Decoder Block
```
$ cat 8-main.py
#!/usr/bin/env python3

import tensorflow as tf
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock

dblock = DecoderBlock(512, 8, 2048)
print(dblock.mha1)
print(dblock.mha2)
print(dblock.dense_hidden)
print(dblock.dense_output)
print(dblock.layernorm1)
print(dblock.layernorm2)
print(dblock.layernorm3)
print(dblock.dropout1)
print(dblock.dropout2)
print(dblock.dropout3)
x = tf.random.uniform((32, 15, 512))
hidden_states = tf.random.uniform((32, 10, 512))
output = dblock(x, hidden_states, False, None, None)
print(output)
$ ./8-main.py
<6-multihead_attention.MultiHeadAttention object at 0x1313f4400>
<6-multihead_attention.MultiHeadAttention object at 0xb368bc9b0>
<tensorflow.python.keras.layers.core.Dense object at 0xb368c37b8>
<tensorflow.python.keras.layers.core.Dense object at 0xb368c3ac8>
<tensorflow.python.keras.layers.normalization.LayerNormalization object at 0xb368c3dd8>
<tensorflow.python.keras.layers.normalization.LayerNormalization object at 0xb368cb080>
<tensorflow.python.keras.layers.normalization.LayerNormalization object at 0xb368cb2e8>
<tensorflow.python.keras.layers.core.Dropout object at 0xb368cb550>
<tensorflow.python.keras.layers.core.Dropout object at 0xb368cb6d8>
<tensorflow.python.keras.layers.core.Dropout object at 0xb368cb828>
Tensor("decoder_block/layer_normalization_2/batchnorm/add_1:0", shape=(32, 15, 512), dtype=float32)
$
```

### 8. Transformer Encoder
```
$ cat 9-main.py
#!/usr/bin/env python3

import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder

encoder = Encoder(6, 512, 8, 2048, 10000, 1000)
print(encoder.dm)
print(encoder.N)
print(encoder.embedding)
print(encoder.positional_encoding)
print(encoder.blocks)
print(encoder.dropout)
x = tf.random.uniform((32, 10))
output = encoder(x, True, None)
print(output)
$ ./9-main.py
512
6
<tensorflow.python.keras.layers.embeddings.Embedding object at 0xb2981acc0>
[[ 0.00000000e+00  1.00000000e+00  0.00000000e+00 ...  1.00000000e+00
   0.00000000e+00  1.00000000e+00]
 [ 8.41470985e-01  5.40302306e-01  8.21856190e-01 ...  9.99999994e-01
   1.03663293e-04  9.99999995e-01]
 [ 9.09297427e-01 -4.16146837e-01  9.36414739e-01 ...  9.99999977e-01
   2.07326584e-04  9.99999979e-01]
 ...
 [-8.97967480e-01 -4.40061818e-01  4.26195541e-01 ...  9.94266169e-01
   1.03168405e-01  9.94663903e-01]
 [-8.55473152e-01  5.17847165e-01  9.86278111e-01 ...  9.94254673e-01
   1.03271514e-01  9.94653203e-01]
 [-2.64607527e-02  9.99649853e-01  6.97559894e-01 ...  9.94243164e-01
   1.03374623e-01  9.94642492e-01]]
ListWrapper([<7-transformer_encoder_block.EncoderBlock object at 0xb2981aef0>, <7-transformer_encoder_block.EncoderBlock object at 0xb29850ba8>, <7-transformer_encoder_block.EncoderBlock object at 0xb298647b8>, <7-transformer_encoder_block.EncoderBlock object at 0xb29e502e8>, <7-transformer_encoder_block.EncoderBlock object at 0xb29e5add8>, <7-transformer_encoder_block.EncoderBlock object at 0xb29e6c908>])
<tensorflow.python.keras.layers.core.Dropout object at 0xb29e7c470>
Tensor("encoder/encoder_block_5/layer_normalization_11/batchnorm/add_1:0", shape=(32, 10, 512), dtype=float32)
$
```

### 9. Transformer Decoder
```
$ cat 10-main.py
#!/usr/bin/env python3

import tensorflow as tf
Decoder = __import__('10-transformer_decoder').Decoder

decoder = Decoder(6, 512, 8, 2048, 12000, 1500)
print(decoder.dm)
print(decoder.N)
print(decoder.embedding)
print(decoder.positional_encoding)
print(decoder.blocks)
print(decoder.dropout)
x = tf.random.uniform((32, 15))
hidden_states = tf.random.uniform((32, 10, 512))
output = decoder(x, hidden_states, True, None, None)
print(output)
$ ./10-main.py
512
6
<tensorflow.python.keras.layers.embeddings.Embedding object at 0xb2cdede48>
[[ 0.00000000e+00  1.00000000e+00  0.00000000e+00 ...  1.00000000e+00
   0.00000000e+00  1.00000000e+00]
 [ 8.41470985e-01  5.40302306e-01  8.21856190e-01 ...  9.99999994e-01
   1.03663293e-04  9.99999995e-01]
 [ 9.09297427e-01 -4.16146837e-01  9.36414739e-01 ...  9.99999977e-01
   2.07326584e-04  9.99999979e-01]
 ...
 [ 9.99516416e-01 -3.10955511e-02 -8.59441209e-01 ...  9.87088496e-01
   1.54561841e-01  9.87983116e-01]
 [ 5.13875021e-01 -8.57865061e-01 -6.94580536e-02 ...  9.87071278e-01
   1.54664258e-01  9.87967088e-01]
 [-4.44220699e-01 -8.95917390e-01  7.80301396e-01 ...  9.87054048e-01
   1.54766673e-01  9.87951050e-01]]
ListWrapper([<8-transformer_decoder_block.DecoderBlock object at 0xb2ce0f0b8>, <8-transformer_decoder_block.DecoderBlock object at 0xb2ce29ef0>, <8-transformer_decoder_block.DecoderBlock object at 0xb2d711b00>, <8-transformer_decoder_block.DecoderBlock object at 0xb2d72c710>, <8-transformer_decoder_block.DecoderBlock object at 0xb2d744320>, <8-transformer_decoder_block.DecoderBlock object at 0xb2d755ef0>])
<tensorflow.python.keras.layers.core.Dropout object at 0xb2d76db38>
Tensor("decoder/decoder_block_5/layer_normalization_17/batchnorm/add_1:0", shape=(32, 15, 512), dtype=float32)
$
```

### 10. Transformer Network
```
$ cat 11-main.py
#!/usr/bin/env python3

import tensorflow as tf
Transformer = __import__('11-transformer').Transformer

transformer = Transformer(6, 512, 8, 2048, 10000, 12000, 1000, 1500)
print(transformer.encoder)
print(transformer.decoder)
print(transformer.linear)
x = tf.random.uniform((32, 10))
y = tf.random.uniform((32, 15))
output = transformer(x, y, True, None, None, None)
print(output)
$ ./11-main.py
<9-transformer_encoder.Encoder object at 0xb2edc5128>
<10-transformer_decoder.Decoder object at 0xb2f412b38>
<tensorflow.python.keras.layers.core.Dense object at 0xb2fd68898>
Tensor("transformer/dense_96/BiasAdd:0", shape=(32, 15, 12000), dtype=float32)
$
```

> Copyright © 2021 Holberton School. All rights reserved.

![Yosri Ghorbel](https://pbs.twimg.com/media/E3YEO7kXwAU9x6x?format=png&name=4096x4096)
