import json

import tensorflow as tf

SEED = 2021


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


if __name__ == "__main__":
    with open("mappings/aid_to_artistid.json") as f:
        aid_map = json.load(f)
    vocab_size = int(max(aid_map))
    sessions = []
    with open("../data/session_data.txt") as f:
        for line in f:
            line = line.split()
            line = list(map(lambda x: int(x), line))
            sessions.append(line)
    sessions = list(filter(lambda x: len(x) > 1, sessions))
    # sessions = process_sequence(sessions[:100], 5, vocab_size)
    # print(sessions)
    generate_training_date(sessions, 2, 4, vocab_size, SEED)
