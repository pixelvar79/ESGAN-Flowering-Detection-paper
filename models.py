from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, LeakyReLU, Dropout, Input, Reshape, Conv2DTranspose, Lambda, Activation
from keras.optimizers import adam_v2
from keras import backend

def custom_activation(output):
    logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
    result = logexpsum / (logexpsum + 1.0)
    return result

def define_discriminator(in_shape=(72, 72, 3), n_classes=2):
    in_image = Input(shape=in_shape)
    fe = Conv2D(32, (3, 3), padding='same')(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = MaxPooling2D(pool_size=(2, 2))(fe)
    fe = Conv2D(64, (3, 3), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = MaxPooling2D(pool_size=(2, 2))(fe)
    fe = Conv2D(128, (3, 3), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = MaxPooling2D(pool_size=(2, 2))(fe)
    fe = Conv2D(256, (3, 3), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = MaxPooling2D(pool_size=(2, 2))(fe)
    fe = Flatten()(fe)
    fe = Dropout(0.4)(fe)
    fe = Dense(n_classes)(fe)
    c_out_layer = Activation('softmax')(fe)
    c_model = Model(in_image, c_out_layer)
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=adam_v2.Adam(lr=0.00001, beta_1=0.01), metrics=['accuracy'])
    d_out_layer = Lambda(custom_activation)(fe)
    d_model = Model(in_image, d_out_layer)
    d_model.compile(loss='binary_crossentropy', optimizer=adam_v2.Adam(lr=0.00001, beta_1=0.01))
    return d_model, c_model

def define_generator(latent_dim):
    in_lat = Input(shape=(latent_dim,))
    n_nodes = 256 * 18 * 18
    X = Dense(n_nodes)(in_lat)
    X = LeakyReLU(alpha=0.2)(X)
    X = Reshape((18, 18, 256))(X)
    X = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    out_layer = Conv2D(3, (3, 3), activation='tanh', padding='same')(X)
    model = Model(in_lat, out_layer)
    return model

def define_gan(g_model, d_model):
    d_model.trainable = False
    gan_output = d_model(g_model.output)
    model = Model(g_model.input, gan_output)
    opt = adam_v2.Adam(lr=0.00001, beta_1=0.02)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model
