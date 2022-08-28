import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
import re
import string
from common.tf_base import TFTrainer, ModelOperator


class TextClassifierTrainer(TFTrainer):
    def __init__(self, data_dir, config={}):
        # merge dictionaries with second overwriting the first
        config = {**{
            'batch_size': 32,
            'seed': 42,
            'validation_split': 0.2,
            'max_features': 10000,
            'sequence_length': 250,
        }, **config}

        vectorize_layer = layers.TextVectorization(
            standardize=self.__custom_standardization,
            max_tokens=config['max_features'],
            output_mode='int',
            output_sequence_length=config['sequence_length'])

        definition = {
            'data_dir': data_dir,
            'vectorize_layer': vectorize_layer,
            'config': config
        }
        super().__init__(definition)

    def _get_data(self):
        dataset_dir = self.definition['data_dir']
        train_dir = dataset_dir / 'train'
        test_dir = dataset_dir / 'test'

        batch_size = self.definition['config']['batch_size']
        seed = self.definition['config']['seed']
        validation_split = self.definition['config']['validation_split']

        raw_train_ds = tf.keras.utils.text_dataset_from_directory(
            train_dir,
            batch_size=batch_size,
            validation_split=validation_split,
            subset='training',
            seed=seed)
        raw_val_ds = tf.keras.utils.text_dataset_from_directory(
            train_dir,
            batch_size=batch_size,
            validation_split=validation_split,
            subset='validation',
            seed=seed)
        raw_test_ds = tf.keras.utils.text_dataset_from_directory(
            test_dir,
            batch_size=batch_size)

        print("Label 0 corresponds to", raw_train_ds.class_names[0])
        print("Label 1 corresponds to", raw_train_ds.class_names[1])

        return raw_train_ds, raw_val_ds, raw_test_ds

    def _preprocess(self, data):
        raw_train_ds, raw_val_ds, raw_test_ds = data
        vectorize_layer = self.definition['vectorize_layer']

        # Make a text-only dataset (without labels), then call adapt
        train_text = raw_train_ds.map(lambda x, y: x)
        vectorize_layer.adapt(train_text)

        def vectorize_text(text, label):
            text = tf.expand_dims(text, -1)
            return vectorize_layer(text), label

        train_ds = raw_train_ds.map(vectorize_text)
        val_ds = raw_val_ds.map(vectorize_text)
        test_ds = raw_test_ds.map(vectorize_text)

        return train_ds, val_ds, test_ds

    def _get_tf_model(self):
        max_features = self.definition['config']['max_features']
        embedding_dim = 16
        model = tf.keras.Sequential([
            layers.Embedding(max_features + 1, embedding_dim),
            layers.Dropout(0.2),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.2),
            layers.Dense(1)])

        model.summary()

        model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                      optimizer='adam',
                      metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
        return model

    def _train_tf_model(self, model, processed_data):
        train_ds, val_ds, test_ds = processed_data

        epochs = 10
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs)

    def __custom_standardization(self, input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html,
                                        '[%s]' % re.escape(string.punctuation),
                                        '')


class TextClassifierOperator(ModelOperator):
    def __init__(self, model, definition={}):
        # merge dictionaries with second overwriting the first
        definition = {**{}, **definition}

        super().__init__(model, definition)

    def evaluate(self, params):
        """
        Params must provide the test dataset

        :param params: params like (train_ds, val_ds, test_ds)
        :return: loss and accuracy like (test_loss, test_acc)
        """
        _, _, test_ds = params

        # evaluate the model
        loss, accuracy = self.model.evaluate(test_ds)
        return loss, accuracy

    def use(self, input_data, config=None):
        raw_process_model = tf.keras.Sequential([
            config['vectorize_layer'],
            self.model,
            layers.Activation('sigmoid')
        ])

        raw_process_model.compile(
            loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
        )

        predictions = raw_process_model.predict(input_data)
        return predictions
