import numpy as np
from numpy import zeros
from numpy.random import randint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_real_samples(X111, y111):
    X1111 = (X111 - 0.5) / 0.5
    encoder = LabelEncoder()
    encoder.fit(y111)
    y1111 = encoder.transform(y111)
    x_train, x_test, y_train, y_test = train_test_split(X1111, y1111, train_size=0.8, random_state=123)
    return [x_train, x_test, y_train, y_test]

def select_supervised_samples(dataset, n_samples, n_classes=2):
    X, _, y, _ = dataset
    #X_list, y_list = []
    X_list = []
    y_list = []
    n_per_class = int(n_samples / n_classes)
    for i in range(n_classes):
        X_with_class = X[y == i]
        ix = randint(0, len(X_with_class), n_per_class)
        [X_list.append(X_with_class[j]) for j in ix]
        [y_list.append(i) for j in ix]
    return np.asarray(X_list), np.asarray(y_list)

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


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict(z_input)
    # create class labels
    y = zeros((n_samples, 1))
    return images, y