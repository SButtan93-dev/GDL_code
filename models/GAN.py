
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import RandomNormal

import numpy as np
import json
import os
import pickle as pkl
import matplotlib.pyplot as plt


class GAN():
    def __init__(self
        , input_dim
        , discriminator_conv_filters
        , discriminator_conv_kernel_size
        , discriminator_conv_strides
        , discriminator_batch_norm_momentum
        , discriminator_activation
        , discriminator_dropout_rate
        , discriminator_learning_rate
        , generator_initial_dense_layer_size
        , generator_upsample
        , generator_conv_filters
        , generator_conv_kernel_size
        , generator_conv_strides
        , generator_batch_norm_momentum
        , generator_activation
        , generator_dropout_rate
        , generator_learning_rate
        , optimiser
        , z_dim
        , virtual_batch_size=None
        , preview_rows=5
        , preview_cols=5
        ):

        self.name = 'gan'

        self.input_dim = input_dim
        self.discriminator_conv_filters = discriminator_conv_filters
        self.discriminator_conv_kernel_size = discriminator_conv_kernel_size
        self.discriminator_conv_strides = discriminator_conv_strides
        self.discriminator_batch_norm_momentum = discriminator_batch_norm_momentum
        self.discriminator_activation = discriminator_activation
        self.discriminator_dropout_rate = discriminator_dropout_rate
        self.discriminator_learning_rate = discriminator_learning_rate

        self.generator_initial_dense_layer_size = generator_initial_dense_layer_size
        self.generator_upsample = generator_upsample
        self.generator_conv_filters = generator_conv_filters
        self.generator_conv_kernel_size = generator_conv_kernel_size
        self.generator_conv_strides = generator_conv_strides
        self.generator_batch_norm_momentum = generator_batch_norm_momentum
        self.generator_activation = generator_activation
        self.generator_dropout_rate = generator_dropout_rate
        self.generator_learning_rate = generator_learning_rate

        self.virtual_batch_size = virtual_batch_size
        self.optimiser = optimiser
        self.z_dim = z_dim

        self.preview_rows = preview_rows
        self.preview_cols = preview_cols

        self.n_layers_discriminator = len(discriminator_conv_filters)
        self.n_layers_generator = len(generator_conv_filters)

        self.weight_init = RandomNormal(mean=0., stddev=0.02)

        self.d_losses = []
        self.g_losses = []

        self.epoch = 0

        self._build_discriminator()
        self._build_generator()

        self._build_adversarial()

    def get_activation(self, activation):
        if activation == 'leaky_relu':
            layer = LeakyReLU(alpha = 0.2)
        else:
            layer = Activation(activation)
        return layer

    def _build_discriminator(self):

        ### THE discriminator
        discriminator_input = Input(shape=self.input_dim, name='discriminator_input')

        x = discriminator_input

        for i in range(self.n_layers_discriminator):
            if i<1:
                x = Conv2D(
                    filters = self.discriminator_conv_filters[i]
                    , kernel_size = self.discriminator_conv_kernel_size[i]
                    , strides = self.discriminator_conv_strides[i]
                    , padding = 'same'
                    , name = 'discriminator_conv_' + str(i)
                    , kernel_initializer = self.weight_init
                    )(x)
                x = self.get_activation(self.discriminator_activation)(x)
            else:
                if self.discriminator_dropout_rate:
                    x = Dropout(rate = self.discriminator_dropout_rate)(x)

                x = Conv2D(
                    filters = self.discriminator_conv_filters[i]
                    , kernel_size = self.discriminator_conv_kernel_size[i]
                    , strides = self.discriminator_conv_strides[i]
                    , padding = 'same'
                    , name = 'discriminator_conv_' + str(i)
                    , kernel_initializer = self.weight_init
                    )(x)

                if self.discriminator_batch_norm_momentum and i > 0:
                    x = BatchNormalization(momentum = self.discriminator_batch_norm_momentum, virtual_batch_size = self.virtual_batch_size)(x)

                x = self.get_activation(self.discriminator_activation)(x)

        x = Dropout(rate = self.discriminator_dropout_rate)(x)
        x = Flatten()(x)

        discriminator_output = Dense(1, activation='sigmoid', kernel_initializer = self.weight_init)(x)

        self.discriminator = Model(discriminator_input, discriminator_output)


    def _build_generator(self):

        ### THE generator

        generator_input = Input(shape=(self.z_dim,), name='generator_input')

        x = generator_input

        x = Dense(np.prod(self.generator_initial_dense_layer_size), kernel_initializer = self.weight_init)(x)

        if self.generator_batch_norm_momentum:
            x = BatchNormalization(momentum = self.generator_batch_norm_momentum, virtual_batch_size = self.virtual_batch_size)(x)

        x = self.get_activation(self.generator_activation)(x)

        x = Reshape(self.generator_initial_dense_layer_size)(x)

        if self.generator_dropout_rate:
            x = Dropout(rate = self.generator_dropout_rate)(x)


        for i in range(self.n_layers_generator):
            # upsample to desired dimensions (using 'size' parameter)
            if self.generator_upsample[i] > 1:
                x = UpSampling2D(
                    size=(self.generator_upsample[i],self.generator_upsample[i]))(x)
                x = Conv2D(
                    filters = self.generator_conv_filters[i]
                    , kernel_size = self.generator_conv_kernel_size[i]
                    , padding = 'same'
                    , name = 'generator_conv_' + str(i)
                    , kernel_initializer = self.weight_init
                )(x)
            else:
                x = Conv2D(
                    filters = self.generator_conv_filters[i]
                    , kernel_size = self.generator_conv_kernel_size[i]
                    , padding = 'same'
                    , name = 'generator_conv_' + str(i)
                    , kernel_initializer = self.weight_init
                    )(x)

            if i < self.n_layers_generator - 1:
                if self.generator_batch_norm_momentum:
                    x = BatchNormalization(momentum = self.generator_batch_norm_momentum, virtual_batch_size = self.virtual_batch_size)(x)

                x = self.get_activation(self.generator_activation)(x)
            else:
                x = Activation('tanh')(x)


        # if self.generator_upsample>1:
        #     # first n-2 elements of array are to set up Conv2D layers with 2x
        #     # upsampling per loop
        #     # n-1 element is for final upsampling to reach desired dimensions
        #     # final element is number of channels (e.g. RGB)
        #     for i in range(self.n_layers_generator-2):
        #         x = UpSampling2D()(x)
        #         x = Conv2D(
        #             filters = self.generator_conv_filters[i]
        #             , kernel_size = self.generator_conv_kernel_size[i]
        #             , padding = 'same'
        #             , name = 'generator_conv_' + str(i)
        #             , kernel_initializer = self.weight_init
        #         )(x)
        #         if self.generator_batch_norm_momentum:
        #             x = BatchNormalization(momentum = self.generator_batch_norm_momentum)(x)
        #
        #         x = self.get_activation(self.generator_activation)(x)
        #
        #     # upsample to desired dimensions (using 'size' parameter)
        #     x = UpSampling2D(
        #         size=(self.generator_upsample,self.generator_upsample)
        #     )(x)
        #     x = Conv2D(
        #         filters = self.generator_conv_filters[self.n_layers_generator-2]
        #         , kernel_size = self.generator_conv_kernel_size[self.n_layers_generator-2]
        #         , padding = 'same'
        #         , name = 'generator_conv_' + str(self.n_layers_generator-2)
        #         , kernel_initializer = self.weight_init
        #     )(x)
        #     if self.generator_batch_norm_momentum:
        #         x = BatchNormalization(momentum = self.generator_batch_norm_momentum)(x)
        #
        #     x = self.get_activation(self.generator_activation)(x)
        #
        #     # Final CNN layer
        #     x = Conv2D(
        #         filters = self.generator_conv_filters[self.n_layers_generator-1]
        #         , kernel_size = self.generator_conv_kernel_size[self.n_layers_generator-1]
        #         , padding = 'same'
        #         , name = 'generator_conv_' + str(self.n_layers_generator-1)
        #         , kernel_initializer = self.weight_init
        #     )(x)
        #     x = Activation('tanh')(x)
        # else:
        #     for i in range(self.n_layers_generator):
        #         x = Conv2DTranspose(
        #             filters = self.generator_conv_filters[i]
        #             , kernel_size = self.generator_conv_kernel_size[i]
        #             , padding = 'same'
        #             , strides = self.generator_conv_strides[i]
        #             , name = 'generator_conv_' + str(i)
        #             , kernel_initializer = self.weight_init
        #             )(x)
        #
        #         if i < self.n_layers_generator - 1:
        #             if self.generator_batch_norm_momentum:
        #                 x = BatchNormalization(momentum = self.generator_batch_norm_momentum)(x)
        #
        #             x = self.get_activation(self.generator_activation)(x)
        #         else:
        #             x = Activation('tanh')(x)


        # for i in range(self.n_layers_generator):
        #     if self.generator_upsample[i] == 2:
        #         x = UpSampling2D()(x)
        #         x = Conv2D(
        #             filters = self.generator_conv_filters[i]
        #             , kernel_size = self.generator_conv_kernel_size[i]
        #             , padding = 'same'
        #             , name = 'generator_conv_' + str(i)
        #             , kernel_initializer = self.weight_init
        #         )(x)
        #     else:
        #
        #         x = Conv2DTranspose(
        #             filters = self.generator_conv_filters[i]
        #             , kernel_size = self.generator_conv_kernel_size[i]
        #             , padding = 'same'
        #             , strides = self.generator_conv_strides[i]
        #             , name = 'generator_conv_' + str(i)
        #             , kernel_initializer = self.weight_init
        #             )(x)
        #
        #     if i < self.n_layers_generator - 1:
        #
        #         if self.generator_batch_norm_momentum:
        #             x = BatchNormalization(momentum = self.generator_batch_norm_momentum)(x)
        #
        #         x = self.get_activation(self.generator_activation)(x)
        #
        #
        #     else:
        #
        #         x = Activation('tanh')(x)
        #




        generator_output = x

        self.generator = Model(generator_input, generator_output)


    def get_opti(self, lr):
        if self.optimiser == 'adam':
            opti = Adam(lr=lr, beta_1=0.5)
        elif self.optimiser == 'rmsprop':
            opti = RMSprop(lr=lr)
        else:
            opti = Adam(lr=lr)

        return opti

    def set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val


    def _build_adversarial(self):

        ### COMPILE DISCRIMINATOR

        self.discriminator.compile(
        optimizer=self.get_opti(self.discriminator_learning_rate)
        , loss = 'binary_crossentropy'
        ,  metrics = ['accuracy']
        )

        ### COMPILE THE FULL GAN

        self.set_trainable(self.discriminator, False)

        model_input = Input(shape=(self.z_dim,), name='model_input')
        model_output = self.discriminator(self.generator(model_input))
        self.model = Model(model_input, model_output)

        self.model.compile(optimizer=self.get_opti(self.generator_learning_rate) , loss='binary_crossentropy', metrics=['accuracy']
        , experimental_run_tf_function=False
        )

        self.set_trainable(self.discriminator, True)



    def train_discriminator(self, x_train, batch_size, using_generator):

        # One-sided label smoothing. Using targets for real examples in the discriminator
        # The idea is to replace the target for the real examples with a value
        # slightly less than one (e.g. 0.9). This prevents extreme extrapolation
        # behavior in the discriminator (e.g. keeping disciminator from approaching 0 loss too rapidly).
        # So we want a value of 0.9 or targets with a stochastic range (e.g. 0.9-1.0)
        # from Goodfellow, I. (2016). NIPS 2016 Tutorial: Generative Adversarial Networks.
        # https://arxiv.org/abs/1701.00160
        # I'm doing this for the fake examples too.
        # apparently flipping the labels (Fake=True, Real=False) also helps
        fake = np.random.uniform(low=0.9, high=1.0, size=batch_size)
        valid = np.random.uniform(low=0.0, high=0.1, size=batch_size)
        #fake = np.ones((batch_size,1))
        #valid = np.zeros((batch_size,1))

        if using_generator:
            true_imgs = next(x_train)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        d_loss_real, d_acc_real = self.discriminator.train_on_batch(true_imgs, valid)
        d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(gen_imgs, fake)
        d_loss =  0.5 * (d_loss_real + d_loss_fake)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)

        return [d_loss, d_loss_real, d_loss_fake, d_acc, d_acc_real, d_acc_fake]


    def train_generator(self, batch_size):
        valid = np.ones((batch_size,1))
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)


    def train(self, x_train, batch_size, epochs, run_folder
    , print_every_n_batches = 50
    , using_generator = False):

        for epoch in range(self.epoch, self.epoch + epochs):

            d = self.train_discriminator(x_train, batch_size, using_generator)
            g = self.train_generator(batch_size)

            print ("%d [D loss: (%.3f)(R %.3f, F %.3f)] [D acc: (%.3f)(%.3f, %.3f)] [G loss: %.3f] [G acc: %.3f]" % (epoch, d[0], d[1], d[2], d[3], d[4], d[5], g[0], g[1]))

            self.d_losses.append(d)
            self.g_losses.append(g)

            if epoch % print_every_n_batches == 0:
                self.sample_images(run_folder)
                self.model.save_weights(os.path.join(run_folder, 'weights/weights-%d.h5' % (epoch)))
                self.model.save_weights(os.path.join(run_folder, 'weights/weights.h5'))
                self.save_model(run_folder)

            self.epoch += 1


    def sample_images(self, run_folder):
        #r, c = 5, 5
        noise = np.random.normal(
            0, 1, (self.preview_rows * self.preview_cols, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(
            self.preview_rows, self.preview_cols, figsize=(15,15))
        cnt = 0

        for i in range(self.preview_rows):
            for j in range(self.preview_cols):
                axs[i,j].imshow(np.squeeze(gen_imgs[cnt, :,:,:]), cmap = 'gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(run_folder, "images/sample_%d.png" % self.epoch))
        plt.close()



    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder ,'viz/model.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.discriminator, to_file=os.path.join(run_folder ,'viz/discriminator.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.generator, to_file=os.path.join(run_folder ,'viz/generator.png'), show_shapes = True, show_layer_names = True)



    def save(self, folder):

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pkl.dump([
                self.input_dim
                , self.discriminator_conv_filters
                , self.discriminator_conv_kernel_size
                , self.discriminator_conv_strides
                , self.discriminator_batch_norm_momentum
                , self.discriminator_activation
                , self.discriminator_dropout_rate
                , self.discriminator_learning_rate
                , self.generator_initial_dense_layer_size
                , self.generator_upsample
                , self.generator_conv_filters
                , self.generator_conv_kernel_size
                , self.generator_conv_strides
                , self.generator_batch_norm_momentum
                , self.generator_activation
                , self.generator_dropout_rate
                , self.generator_learning_rate
                , self.optimiser
                , self.z_dim
                , self.virtual_batch_size
                , self.preview_rows
                , self.preview_cols
                ], f)

        self.plot_model(folder)

    def save_model(self, run_folder):
        self.model.save(os.path.join(run_folder, 'model.h5'))
        self.discriminator.save(os.path.join(run_folder, 'discriminator.h5'))
        self.generator.save(os.path.join(run_folder, 'generator.h5'))

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
