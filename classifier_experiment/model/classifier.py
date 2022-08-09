import tensorflow as tf
from common.tf_base import TFTrainer, ModelOperator


class ClassifierTrainer(TFTrainer):
    def __init__(self, data, config={}):
        # merge dictionaries with second overwriting the first
        config = {**{
            'input_shape': (28, 28),
            'nodes': 128,
            'EPOCHS': 10,
            'num_classes': 10
        }, **config}

        definition = {
            'data': data,
            'config': config
        }
        super().__init__(definition)

    def _get_data(self):
        return self.definition['data']

    def _preprocess(self, data):
        (train_images, train_labels), (test_images, test_labels) = data
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        return (train_images, train_labels), (test_images, test_labels)

    def _get_tf_model(self):
        # get definitions
        nodes = self.definition['config']['nodes']
        input_shape = self.definition['config']['input_shape']
        num_classes = self.definition['config']['num_classes']

        # define network structure
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(nodes, activation='relu'),
            tf.keras.layers.Dense(num_classes)
        ])
        # define network optimizer and metrics
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

    def _train_tf_model(self, model, processed_data):
        # get definitions
        epochs = self.definition['config']['EPOCHS']

        # get data
        (train_images, train_labels), (test_images, test_labels) = processed_data

        # train model
        model.fit(train_images, train_labels, epochs=epochs)


class ClassifierOperator(ModelOperator):
    def __init__(self, model, definition={}):
        # merge dictionaries with second overwriting the first
        definition = {**{}, **definition}

        super().__init__(model, definition)

    def evaluate(self, params):
        """
        Params must provide the test images and labels

        :param params: params like (test_images, test_labels)
        :return: loss and accuracy like (test_loss, test_acc)
        """
        # get images and labels
        (test_images, test_labels) = params

        # evaluate the model
        test_loss, test_acc = self.model.evaluate(test_images, test_labels, verbose=2)
        return test_loss, test_acc

    def use(self, input_data, config=None):
        probability_model = tf.keras.Sequential([self.model,
                                                 tf.keras.layers.Softmax()])
        predictions = probability_model.predict(input_data)
        return predictions
