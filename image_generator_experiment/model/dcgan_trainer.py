import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from pathlib import Path
from common.tf_base import TFTrainer, ModelOperator

from IPython import display

class DCGANTrainer(TFTrainer):
    def __init__(self, data, config={}):
        # merge dictionaries with second overwriting the first
        config = {**{
            'buffer_size': 60000,
            'batch_size': 256,
            'EPOCHS': 50,
            'checkpoint_frequency': 15,
            'restore_ckpt': False,
            'noise_dim': 100,
            'cache_dir': './image_generator_experiment/cache/',
        }, **config}

        definition = {
            'data': data,
            'config': config
        }

        # This method returns a helper function to compute cross entropy loss
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        super().__init__(definition)

    def _get_data(self):
        return self.definition['data']

    def _preprocess(self, data):
        # Grab data definitions
        (train_images, _), (_, _) = data
        BUFFER_SIZE = self.definition['config']['buffer_size']
        BATCH_SIZE = self.definition['config']['batch_size']

        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

        # Batch and shuffle the data
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

        return train_dataset

    def _get_tf_model(self):
        generator = self.__make_generator_model()
        discriminator = self.__make_discriminator_model()
        return generator, discriminator

    def _train_tf_model(self, model, processed_data):
        (generator, discriminator) = model

        # generator test
        # noise = tf.random.normal([1, 100])
        # generated_image = generator(noise, training=False)
        # plt.imshow(generated_image[0, :, :, 0], cmap='gray')
        # plt.show()

        # discriminator test
        # decision = discriminator(generated_image)
        # print(decision)

        generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        checkpoint_dir = Path(self.definition['config']['cache_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_prefix = Path(self.definition['config']['cache_dir']) / 'ckpt'
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         generator=generator,
                                         discriminator=discriminator)
        ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir.__str__(), max_to_keep=5)

        EPOCHS = self.definition['config']['EPOCHS']
        LATEST_EPOCH = 0

        if self.definition['config']['restore_ckpt'] and ckpt_manager.latest_checkpoint:
            LATEST_EPOCH = int(ckpt_manager.latest_checkpoint.split('-')[-1])
            print("Found latest epoch is {}".format(LATEST_EPOCH))
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir.__str__()))

        noise_dim = self.definition['config']['noise_dim']
        num_examples_to_generate = 16

        # You will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF)
        tf.random.set_seed(0)
        seed = tf.random.normal([num_examples_to_generate, noise_dim])

        for epoch in range(LATEST_EPOCH, EPOCHS):
            start = time.time()

            total_batches = len(processed_data)
            load_bar_length = 50
            for i_batch, image_batch in enumerate(processed_data):
                load_bar = '[{}{}{}]'.format(
                    '=' * int(load_bar_length * i_batch/total_batches),
                    ">",
                    '.' * int(load_bar_length * (1 - i_batch/total_batches)))
                print("\rEPOCH {}, Batch {}/{}\t{}".format(epoch, i_batch, total_batches, load_bar), end='')
                self.train_step(model,
                                (generator_optimizer, discriminator_optimizer),
                                image_batch)
            print()

            # Any reference to epoch after this do +1 since just trained one step

            # Produce images for the GIF as you go
            display.clear_output(wait=True)
            self.generate_and_save_images(generator,
                                     epoch + 1,
                                     seed)

            # Save the model every n epochs
            if (epoch + 1) % self.definition['config']['checkpoint_frequency'] == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch, time.time() - start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        self.generate_and_save_images(generator,
                                 EPOCHS,
                                 seed)

        history = 1
        return history

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, models, optimizers, images):
        generator, discriminator = models
        generator_optimizer, discriminator_optimizer = optimizers
        BATCH_SIZE = self.definition['config']['batch_size']
        noise_dim = self.definition['config']['noise_dim']

        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    def __make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        return model

    def __make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[28, 28, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generate_and_save_images(self, model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plot_name = os.path.join(self.definition['config']['cache_dir'], 'image_at_epoch_{:04d}.png'.format(epoch))
        plt.savefig(plot_name)
        plt.show()


class GeneratorOperator(ModelOperator):
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

        pass

    def use(self, input_data, config=None):
        pass

    def plot(self, history):
        pass


class DiscriminatorOperator(ModelOperator):
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

        pass

    def use(self, input_data, config=None):
        pass

    def plot(self, history):
        pass