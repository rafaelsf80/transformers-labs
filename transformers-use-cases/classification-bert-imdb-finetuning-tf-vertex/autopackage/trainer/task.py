import argparse

import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification

CHECKPOINT = "bert-base-cased"

def get_args():
  '''Parses args.'''

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--epochs',
      required=False,
      default=3,
      type=int,
      help='number of epochs')
  parser.add_argument(
      '--job_dir',
      required=True,
      type=str,
      help='bucket to store saved model, include gs://')
  args = parser.parse_args()
  return args


def create_datasets():
    '''Creates a tf.data.Dataset for train and evaluation.'''

    raw_datasets = load_dataset('imdb')
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    tokenized_datasets = raw_datasets.map((lambda examples: tokenize_function(examples, tokenizer)), batched=True)

    # To speed up training, we use only a portion of the data.
    # Use full_train_dataset and full_eval_dataset if you want to train on all the data.
    small_train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets['test'].shuffle(seed=42).select(range(1000))
    full_train_dataset = tokenized_datasets['train']
    full_eval_dataset = tokenized_datasets['test']

    tf_train_dataset = small_train_dataset.remove_columns(['text']).with_format("tensorflow")
    tf_eval_dataset = small_eval_dataset.remove_columns(['text']).with_format("tensorflow")

    train_features = {x: tf_train_dataset[x] for x in tokenizer.model_input_names}
    train_tf_dataset = tf.data.Dataset.from_tensor_slices((train_features, tf_train_dataset["label"]))
    train_tf_dataset = train_tf_dataset.shuffle(len(tf_train_dataset)).batch(8)

    eval_features = {x: tf_eval_dataset[x] for x in tokenizer.model_input_names}
    eval_tf_dataset = tf.data.Dataset.from_tensor_slices((eval_features, tf_eval_dataset["label"]))
    eval_tf_dataset = eval_tf_dataset.batch(8)

    return train_tf_dataset, eval_tf_dataset


def tokenize_function(examples, tokenizer):
    '''Tokenizes text examples.'''

    return tokenizer(examples['text'], padding='max_length', truncation=True)


def main():
    args = get_args()
    train_tf_dataset, eval_tf_dataset = create_datasets()
    model = TFAutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy(),
    )

    model.fit(train_tf_dataset, validation_data=eval_tf_dataset, epochs=args.epochs)
    model.save(f'{args.job_dir}/model_output')


if __name__ == "__main__":
    main()