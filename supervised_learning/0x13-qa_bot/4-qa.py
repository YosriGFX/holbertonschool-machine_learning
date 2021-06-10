#!/usr/bin/env python3
'''4. Multi-reference Question Answering'''
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(coprus_path):
    '''answers questions from multiple reference texts'''
    model = hub.load(
        'https://tfhub.dev/see--/bert-uncased-tf2-qa/1'
    )
    embed = hub.load(
        'https://tfhub.dev/google/universal-sentence-encoder-large/5'
    )
    tz = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    exit_list = ['exit', 'quit', 'goodbye', 'bye']
    while True:
        print(
            'Q:', end=' '
        )
        Question = input()
        if Question.lower() in [
            'exit',
            'quit',
            'goodbye',
            'bye'
        ]:
            Answer = 'Goodbye'
            print('A: {}'.format(Answer))
            break
        documents = []
        for file in os.listdir(coprus_path):
            if file.endswith(".md"):
                filename = os.path.join(
                    coprus_path,
                    file
                )
                with open(filename) as f:
                    documents.append(
                        f.read()
                    )
        argmax = np.argmax(
            np.dot(
                embed([Question]).numpy(),
                (embed(documents).numpy()).T
            )
        )
        question_tz = tz.tokenize(Question)
        reference_tz = tz.tokenize(documents[argmax])
        tokens = ['[CLS]'] + question_tz + ['[SEP]'] + reference_tz + ['[SEP]']
        input_ids = tz.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        type_ids = [0] * (len(question_tz) + 2) + [1] * (len(reference_tz) + 1)
        input_ids, input_mask, type_ids = map(
            lambda x:
                tf.expand_dims(tf.convert_to_tensor(x, dtype=tf.int32), 0),
                (input_ids, input_mask, type_ids)
        )
        outputs = model([input_ids, input_mask, type_ids])
        short_start = tf.argmax(outputs[0][0][1:]) + 1
        short_end = tf.argmax(outputs[1][0][1:]) + 1
        answer_tokens = tokens[short_start: short_end + 1]
        if not answer_tokens:
            print('A: Sorry, I do not understand your Question.')
        else:
            print(
                'A: {}'.format(
                    tz.convert_tokens_to_string(
                        answer_tokens
                    )
                )
            )
