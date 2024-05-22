import os
import pandas as pd
import numpy as np
from tifffile import imread, imwrite
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.preprocessing import LabelEncoder
from numpy.random import randint
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dropout, Input, Reshape, Conv2DTranspose, LeakyReLU, Lambda, Activation
from keras.optimizers import adam_v2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, f1_score, jaccard_score, precision_score, recall_score
from PIL import Image as Img
import gc
import random
import joblib
import pickle
import sklearn
from keras.datasets.mnist import load_data
from matplotlib import pyplot
from keras import backend

# Set directories
# Set base directory
base_dir = 'D:\\OneDrive - University of Illinois - Urbana\\TF\\'
# Define subdirectories
dir_out = os.path.join(base_dir, 'Data\\GANS_FL\\output\\head50\\gan')
dir_img = os.path.join(base_dir, 'Data\\MSA2019_RGB_SEASON\\IMAGES_ALL')
dir_gt = os.path.join(base_dir, 'Data\\GANS_FL\\data')
dir_restmodels = os.path.join(base_dir, 'Data\\GANS_FL\\output\\head50\\smallcnn')

# Change working directory
os.chdir(dir_out)

# Display max rows setting for pandas
pd.set_option('display.max_rows', 100)

# Import utility functions
import Utils_RGB_head_class
from Utils_RGB_head_class import load_dataset, load_image
import Utils_RGB_head_class1
from Utils_RGB_head_class1 import load_dataset1, load_image1

# Load data
def load_data():
    X, y, y1, y11 = Utils_RGB_head_class.load_dataset(dir_img, dir_gt, task='classification2')
    return X, y, y1, y11

X, y, y1, y11 = load_data()

dates = ('247', '262', '279')

# Prepare data
def preparing_data():
    tf.keras.backend.clear_session()
    gc.collect()
    
    slices = ((dates[0], slice(61, 64)),
              (dates[1], slice(67, 70)),
              (dates[2], slice(73, 76)))
    
    list_data = []
    labels = []

    for names, slicing in slices:
        list_data.append(X[:, :, :, slicing])
        labels.append(names)
    
    X1 = np.concatenate([list_data[0], list_data[1], list_data[2]], 0)
    
    def rescale(arr):
        arr_min = arr.min()
        arr_max = arr.max()
        return (arr - arr_min) / (arr_max - arr_min)

    LIST_IMG = [rescale(im) for im in X1]
    X111 = np.stack(LIST_IMG, axis=0)
    
    yy = pd.Series(np.concatenate((y, y1, y11), axis=0))
    
    return X111, yy

X111, y111 = preparing_data()

# Custom activation function
def custom_activation(output):
    logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
    result = logexpsum / (logexpsum + 1.0)
    return result

# Define discriminator
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

d_model, c_model = define_discriminator()

# Define generator
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

g_model = define_generator(100)

# Define GAN
def define_gan(g_model, d_model):
    d_model.trainable = False
    gan_output = d_model(g_model.output)
    model = Model(g_model.input, gan_output)
    opt = adam_v2.Adam(lr=0.00001, beta_1=0.02)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

gan_model = define_gan(g_model, d_model)

# Load real samples
def load_real_samples():
    X1111 = (X111 - 0.5) / 0.5
    encoder = LabelEncoder()
    encoder.fit(y111)
    y1111 = encoder.transform(y111)
    x_train, x_test, y_train, y_test = train_test_split(X1111, y1111, train_size=0.8, random_state=123)
    return [x_train, x_test, y_train, y_test]

dataset = load_real_samples()

# Select supervised samples
def select_supervised_samples(dataset, n_samples, n_classes=2):
    X, _, y, _ = dataset
    X_list, y_list = [], []
    n_per_class = int(n_samples / n_classes)
    for i in range(n_classes):
        X_with_class = X[y == i]
        ix = randint(0, len(X_with_class), n_per_class)
        [X_list.append(X_with_class[j]) for j in ix]
        [y_list.append(i) for j in ix]
    return np.asarray(X_list), np.asarray(y_list)

# Generate real samples
def generate_real_samples(dataset, n_samples):
    images, labels = dataset
    ix = randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    y = np.ones((n_samples, 1))
    return [X, labels], y

def generate_real_samples1(dataset, n_samples):
    images, _, labels, _ = dataset
    ix = randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    y = np.ones((n_samples, 1))
    return [X, labels], y

# Generate latent points
def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# Use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input)
    y = np.zeros((n_samples, 1))
    return X, y

# Create and save a plot of generated images
def save_plot(examples, epoch, n=10):
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i])
    filename = f'generated_plot_e{epoch+1:03d}.png'
    plt.savefig(filename)
    plt.close()

# Evaluate the discriminator, plot generated images, and save the model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    X_real, y_real = generate_real_samples1(dataset, n_samples)
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    X_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(X_fake, y_fake, verbose=0)
    print(f'>Accuracy real: {acc_real*100:.0f}%, fake: {acc_fake*100:.0f}%')
    save_plot(X_fake, epoch)
    filename = f'generator_model_{epoch+1:03d}.h5'
    g_model.save(filename)

# Train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            X_real, y_real = generate_real_samples1(dataset, half_batch)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            print(f'>{i+1}, {j+1}/{bat_per_epo}, d1={d_loss1:.3f}, d2={d_loss2:.3f}, g={g_loss:.3f}')
        if (i+1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)

# Set the size of the latent space
latent_dim = 100
train(g_model, d_model, gan_model, dataset, latent_dim)
