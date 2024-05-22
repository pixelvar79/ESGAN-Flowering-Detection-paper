import os
import tensorflow as tf
import gc
import matplotlib.pyplot as plt
import numpy as np

from directories import dir_img, dir_gt, dir_out
from data_utils import load_dataset, preparing_data
from models import define_discriminator, define_generator, define_gan
from gan_utils import load_real_samples, select_supervised_samples, generate_real_samples, generate_real_samples1, generate_latent_points
print(dir_img)
def train_gan(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            
            X_fake, _ = generate_latent_points(latent_dim, half_batch)
            y_fake = np.zeros((half_batch, 1))
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            
            print(f'>{i+1}, {j+1}/{bat_per_epo}, d1={d_loss1:.3f}, d2={d_loss2:.3f}, g={g_loss:.3f}')
        
        if (i+1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)

def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    [X_real, labels_real], _ = generate_real_samples(dataset, n_samples)
    _, acc_real = d_model.evaluate(X_real, np.ones((n_samples, 1)), verbose=0)
    
    x_fake, _ = generate_latent_points(latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(x_fake, np.zeros((n_samples, 1)), verbose=0)
    
    print(f'Accuracy real: {acc_real*100:.0f}%, fake: {acc_fake*100:.0f}%')
    
    save_plot(x_fake, epoch)
    g_model.save(os.path.join(dir_out, f'generator_model_{epoch+1}.h5'))

def save_plot(examples, epoch, n=10):
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :, 0])
    filename = os.path.join(dir_out, f'generated_plot_e{epoch+1}.png')
    plt.savefig(filename)
    plt.close()

# Set directories
os.chdir(dir_out)

# Load data
X, y, y1, y11 = load_dataset(dir_img, dir_gt, task='classification2')

# Prepare data
X111, yy = preparing_data(X, y, y1, y11)

# Load real samples
dataset = load_real_samples(X111, yy)

# Define models
latent_dim = 100
d_model, c_model = define_discriminator()
g_model = define_generator(latent_dim)
gan_model = define_gan(g_model, d_model)

# Train the GAN
# train_gan(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128)
