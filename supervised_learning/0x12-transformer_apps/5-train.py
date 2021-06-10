#!/usr/bin/env python3'''
from numpy.lib.function_base import gradient
import tensorflow as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    '''CustomSchedule class schedules learning rate'''
    def __init__(self, d_model, warmup_steps=4000):
        '''Initializer'''
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        '''Instance call'''
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(label, prediction):
    '''loss_function'''
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )
    mask = tf.math.logical_not(tf.math.equal(label, 0))
    loss = loss_object(label, prediction)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss)/tf.reduce_sum(mask)


def accuracy_function(label, prediction):
    '''ccuracy_function'''
    accuracies = tf.equal(label, tf.argmax(prediction, axis=2))

    mask = tf.math.logical_not(tf.math.equal(label, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    '''train_transformer'''
    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    ds = Dataset(batch_size, max_len)

    transformer = Transformer(
        N, dm, h, hidden,
        ds.tokenizer_pt.vocab_size(), ds.tokenizer_en.vocab_size(),
        1000, 1500
    )
    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        '''train_transformer'''
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        encoder_mask, combined_mask, decoder_mask = create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = transformer(
                inp, tar_inp,
                True,
                encoder_mask,
                combined_mask,
                decoder_mask
            )
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables)
        )
        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        for (batch, (inputs, target)) in enumerate(ds.data_train):
            train_step(inputs, target)
            if batch % 50 == 0:
                loss = train_loss.result()
                accr = train_accuracy.result()
                print('Epoch {}, batch {}: loss {} accuracy {}').format(
                        epoch, batch, loss, accr
                    )
        loss = train_loss.result()
        accr = train_accuracy.result()
        print('Epoch {}: loss {} accuracy {}'.format(epoch, loss, accr))
