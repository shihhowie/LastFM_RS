import json
import random
import os

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten

from helpers import timer

from redis_connection import r

SEED = 2021
with open("mappings/aid_to_artistid.json") as f:
    aid_map = json.load(f)
vocab_size = len(aid_map)


class Artist2Vec(Model):
    def __init__(self, vocab_size, embedding_dim, num_ns=4):
        super(Artist2Vec, self).__init__()
        self.target_embedding = Embedding(
            vocab_size, embedding_dim, input_length=1, name="a2v_embedding"
        )
        self.context_embedding = Embedding(
            vocab_size, embedding_dim, input_length=num_ns + 1
        )
        self.dots = Dot(axes=(3, 2))
        self.flatten = Flatten()

    def call(self, pair):
        target, context = pair
        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        dots = self.dots([context_emb, word_emb])
        return self.flatten(dots)


def generate_training_date(sequences, window_size, num_ns, vocab_size, seed):
    targets, contexts, labels = [], [], []

    for sequence in sequences:
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            window_size=window_size,
            negative_samples=0,
        )
        # print(positive_skip_grams[:5])
        for target, context in positive_skip_grams:
            positive_class = tf.expand_dims(tf.constant([context], dtype="int64"), 1)
            negative_samples, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=positive_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=SEED,
                name="negative_sampling",
            )
            negative_samples = tf.expand_dims(negative_samples, 1)
            context = tf.concat([positive_class, negative_samples], 0)
            label = tf.constant(([1] + [0] * num_ns), dtype="int64")

            # print(context)
            targets.append(target)
            labels.append(label)
            contexts.append(context)
    return targets, contexts, labels


def process_sequence(sequences, max_tokens, vocab_size):
    """
    * padding might not be required if window size is 2
    Pads sequence with tokens fewer than max_tokens
        - padding uses max_id+1 since 0 is already used
    Split sequence with longer than max_tokens length
    """
    padding = vocab_size + 1
    processed_sequence = []
    for sequence in sequences:
        if len(sequence) <= max_tokens:
            sequence = sequence + [padding] * (max_tokens - len(sequence))
            processed_sequence.append(sequence)
        else:
            while len(sequence) > max_tokens:
                subseq = sequence[:max_tokens]
                sequence = sequence[max_tokens - 1 :]
                processed_sequence.append(subseq)
            if len(sequence) > 1:
                sequence = sequence + [padding] * (max_tokens - len(sequence))
                processed_sequence.append(sequence)
    return processed_sequence


def prepare_data():
    sessions = []
    with open("../data/session_data.txt") as f:
        for line in f:
            line = line.split()
            line = list(map(lambda x: int(x), line))
            sessions.append(line)
    sessions = list(filter(lambda x: len(x) > 1, sessions))
    # print(sessions)
    return sessions


def extract_embeddings(model):
    embeddings = model.layers[0].get_weights()[0]
    embedding_map = {aid: embeddings[int(aid)].tolist() for aid in aid_map}
    with open("artist_embeddings.json", "w") as f:
        json.dump(embedding_map, f)


def train_model(
    num_ns=4,
    window_size=2,
    embedding_dim=32,
    batch_size=256,
    buffer_size=10000,
    epochs=5,
):
    sessions = prepare_data()
    random.shuffle(sessions)
    targets, contexts, labels = generate_training_date(
        sessions[:20000], window_size, num_ns, vocab_size, SEED
    )
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    dataset = dataset.cache().prefetch(buffer_size=buffer_size)
    artist2vec = Artist2Vec(vocab_size, embedding_dim, num_ns=num_ns)
    artist2vec.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    artist2vec.fit(dataset, epochs=epochs, callbacks=[tensorboard_callback])
    return artist2vec


def store_embedding_to_redis():

    curr_path = os.path.dirname(os.path.abspath(__file__))

    with open(f"{curr_path}/embeddings/artist_embeddings.json") as f:
        aid_embedding = json.load(f)
    with r.pipeline() as pipe:
        for aid, a_embedding in aid_embedding.items():
            r.rpush(f"aid:{aid}", *a_embedding)
        pipe.execute()


if __name__ == "__main__":
    # model = train_model(epochs=10)
    # extract_embeddings(model)
    pass
