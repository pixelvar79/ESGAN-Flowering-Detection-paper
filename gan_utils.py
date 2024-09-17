import numpy as np
from numpy import zeros
from numpy.random import randint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and preprocess real samples
# This function normalizes the input images, encodes the labels, and splits the data into training and testing sets.
def load_real_samples(X111, y111):
    
    # Normalize the images to the range [-1, 1]
    X1111 = (X111 - 0.5) / 0.5
    # Encode the labels as integers
    encoder = LabelEncoder()
    encoder.fit(y111)
    y1111 = encoder.transform(y111)
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X1111, y1111, train_size=0.8, random_state=123)
    return [x_train, x_test, y_train, y_test]


def load_real_samples1(X, y):
    
    
# Select supervised samples from the dataset
# This function selects a specified number of samples for each class from the dataset.
def select_supervised_samples(dataset, n_samples, n_classes=2):
    
    X, _, y, _ = dataset
    X_list = []
    y_list = []
    # Calculate the number of samples per class
    n_per_class = int(n_samples / n_classes)
    for i in range(n_classes):
        # Select samples for the current class
        X_with_class = X[y == i]
        ix = randint(0, len(X_with_class), n_per_class)
        [X_list.append(X_with_class[j]) for j in ix]
        [y_list.append(i) for j in ix]
    return np.asarray(X_list), np.asarray(y_list)

# Generate real samples with labels for supervised training
# selects a specified number of real samples from the dataset and assigns them a label of 1.
def generate_real_samples(dataset, n_samples):
    images, labels = dataset
    # Randomly select samples
    ix = randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    # Create class labels (1 for real)
    y = np.ones((n_samples, 1))
    return [X, labels], y

# Generate real samples with labels for unsupervised training
# randomly selects a specified number of real samples from the dataset and assigns them a label of 1.
def generate_real_samples1(dataset, n_samples):
    images, _, labels, _ = dataset
    # Randomly select samples
    ix = randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    # Create class labels (1 for real)
    y = np.ones((n_samples, 1))
    return [X, labels], y

# Generate latent points
# generates a specified number of points in the latent space, which are used as input for the generator.
def generate_latent_points(latent_dim, n_samples):
    # Generate random points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# Use the generator to generate fake examples with class labels
# uses the generator to create a specified number of fake samples and assigns them a label of 0.
def generate_fake_samples(generator, latent_dim, n_samples):
    # Generate points in latent space
    z_input = generate_latent_points(latent_dim, n_samples)
    # Predict outputs (fake images)
    images = generator.predict(z_input)
    # Create class labels (0 for fake)
    y = zeros((n_samples, 1))
    return images, y