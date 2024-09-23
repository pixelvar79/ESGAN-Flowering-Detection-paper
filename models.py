from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, MaxPooling2D, Flatten, Dropout, Dense, Activation, Lambda, Reshape, Conv2DTranspose, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import load_model
import joblib
import gc
import tensorflow as tf
import os

# Custom activation function
# This function ensures that the output values are constrained within a specific range.
# This is considered useful in adversarial training scenarios where the discriminator's 
# output needs to be in a certain range to stabilize training.
def custom_activation(output):
    logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
    result = logexpsum / (logexpsum + 1.0)
    return result

# Define the discriminator model
def define_discriminator(in_shape=(72, 72, 3), n_classes=2):
    # Input layer for the image
    in_image = Input(shape=in_shape)
    
    # First convolutional layer with LeakyReLU activation
    fe = Conv2D(32, (3, 3), padding='same')(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    
    # Max pooling layer to reduce spatial dimensions
    fe = MaxPooling2D(pool_size=(2, 2))(fe)
    
    # Second convolutional layer with LeakyReLU activation
    fe = Conv2D(64, (3, 3), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    
    # Max pooling layer to reduce spatial dimensions
    fe = MaxPooling2D(pool_size=(2, 2))(fe)
    
    # Third convolutional layer with LeakyReLU activation
    fe = Conv2D(128, (3, 3), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    
    # Max pooling layer to reduce spatial dimensions
    fe = MaxPooling2D(pool_size=(2, 2))(fe)
    
    # Fourth convolutional layer with LeakyReLU activation
    fe = Conv2D(256, (3, 3), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    
    # Max pooling layer to reduce spatial dimensions
    fe = MaxPooling2D(pool_size=(2, 2))(fe)
    
    # Flatten the feature maps to a 1D vector
    fe = Flatten()(fe)
    
    # Dropout layer to reduce overfitting
    fe = Dropout(0.4)(fe)
    
    # Fully connected layer with n_classes outputs
    fe = Dense(n_classes)(fe)
    
    # Classification output with softmax activation
    c_out_layer = Activation('softmax')(fe)
    c_model = Model(in_image, c_out_layer)
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.00001, beta_1=0.01), metrics=['accuracy'])
    
    # Discriminator output with custom activation
    d_out_layer = Lambda(custom_activation)(fe)
    d_model = Model(in_image, d_out_layer)
    d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00001, beta_1=0.01))
    
    return d_model, c_model

# Define the generator model
def define_generator(latent_dim):
    # Input layer for the latent vector
    in_lat = Input(shape=(latent_dim,))
    
    # Fully connected layer to reshape the latent vector
    n_nodes = 256 * 18 * 18
    X = Dense(n_nodes)(in_lat)
    X = LeakyReLU(alpha=0.2)(X)
    X = Reshape((18, 18, 256))(X)
    
    # First deconvolutional layer to upsample the feature maps
    X = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    
    # Second deconvolutional layer to upsample the feature maps
    X = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    
    # Output layer with tanh activation to generate the final image
    out_layer = Conv2D(3, (3, 3), activation='tanh', padding='same')(X)
    model = Model(in_lat, out_layer)
    
    return model

# Define the GAN model
def define_gan(g_model, d_model):
    # Make the discriminator weights non-trainable
    d_model.trainable = False
    
    # Connect the generator and discriminator
    gan_output = d_model(g_model.output)
    model = Model(g_model.input, gan_output)
    
    # Compile the GAN model
    opt = Adam(lr=0.00001, beta_1=0.02)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    
    return model

# Define and train the RandomForest model
def train_rf(x_train1, y_train1, nsample, outdir, iter):
    print(f'training RF model {nsample}...')
    
    cv = model_selection.KFold(n_splits=10)
    classif = RandomForestClassifier(random_state=123)
    
    # Define the hyperparameters to search
    param_grid = {'n_estimators': [400, 500, 600], 'max_depth': [5, 10, 20]}
    
    gridforest = GridSearchCV(classif, param_grid, cv=cv, n_jobs=-1, verbose=2)
    gridforest.fit(x_train1, y_train1)
    best_params = gridforest.best_params_
    model = RandomForestClassifier()
    model.set_params(**best_params)
    
    model.fit(x_train1, y_train1)
    
    MODEL_NAME = f'RF_{nsample}_{iter}.pkl'
    
    MODEL_PATH = os.path.join(outdir, MODEL_NAME)  # Construct the full path

    
    joblib.dump(model, MODEL_PATH)
    
    return model

# Define and train the KNN model
def train_knn(x_train1, y_train1, nsample, outdir, iter):
    print(f'training KNN model {nsample}...')
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train1, y_train1)
    
    MODEL_NAME = f'KNN_{nsample}_{iter}.pkl'
    MODEL_PATH = os.path.join(outdir, MODEL_NAME) 
    
    joblib.dump(model, MODEL_PATH)
    
    return model


## PARAMETERS FOR TRANSF LEARNING AND SMALLCNN MODELS
EPOCHS = 200
SIZE = 16
MODE = 'max'
METRIC_VAR = 'val_accuracy'
LR = 0.001
DC = 0.0001

# Define and train the Transfer Learning CNN model
def train_transf(x_train1, y_train1, x_val, y_val, nsample, outdir, iter):
    print(f'training ResNEt50 model {nsample}...')
   
    MODEL_NAME = f'TRANSF_{nsample}_{iter}.h5'
    
    MODEL_PATH = os.path.join(outdir, MODEL_NAME) 
    
    #optimizer = tf.keras.optimizers.Adam(learning_rate=LR, decay=DC)
    es = callbacks.EarlyStopping(monitor=METRIC_VAR, verbose=1, mode=MODE, min_delta=0.01, patience=5)
    mc = callbacks.ModelCheckpoint(MODEL_PATH, monitor=METRIC_VAR, mode=MODE, save_best_only=True, verbose=1)
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    
    avg = GlobalAveragePooling2D()(base_model.output)
    
    output = Dense(1, activation='sigmoid')(avg)
    
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    
    for layer in base_model.layers:
        layer.trainable = True
        
    optimizer = tf.keras.optimizers.Adam(lr=0.0001, decay=0.00001)
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(x_train1, y_train1, epochs=EPOCHS, batch_size=SIZE, validation_data=(x_val, y_val), verbose=2, callbacks=[mc, es])
    
    # Load the best model saved by ModelCheckpoint
    best_model = load_model(MODEL_PATH)
    
    return best_model

# Define and train the Basic CNN model
def train_smallcnn(x_train1, y_train1, x_val, y_val, nsample, outdir, iter):
    print(f'training small CNN model {nsample}...')
    
    MODEL_NAME = f'SMALLCNN_{nsample}_{iter}.h5'
    
    MODEL_PATH = os.path.join(outdir, MODEL_NAME) 
    
    #OPT = tf.keras.optimizers.Adam(learning_rate=LR, decay=DC)
    es = callbacks.EarlyStopping(monitor=METRIC_VAR, verbose=1, mode=MODE, min_delta=0.01, patience=5)
    mc = callbacks.ModelCheckpoint(MODEL_PATH, monitor=METRIC_VAR, mode=MODE, save_best_only=True, verbose=1)
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(72, 72, 3)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    opt = SGD(lr=0.001, momentum=0.9)
    
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(x_train1, y_train1, epochs=EPOCHS, batch_size=SIZE, validation_data=(x_val, y_val), verbose=2, callbacks=[mc, es])
    
    # Load the best model saved by ModelCheckpoint
    best_model = load_model(MODEL_PATH)
    
    return best_model

