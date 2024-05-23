import os
import tensorflow as tf
import gc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from numpy import ones
from numpy import zeros
from PIL import Image as Img
from directories import dir_img, dir_gt, dir_out
from data_utils import load_dataset, preparing_data
from models import define_discriminator, define_generator, define_gan
from gan_utils import load_real_samples, select_supervised_samples, generate_real_samples, generate_real_samples1, generate_latent_points, generate_fake_samples

# train the generator and discriminator
def train(g_model, d_model, c_model, gan_model, dataset, latent_dim, n_epochs=20):
    # select supervised dataset 
    print(n_batchs)
    X_sup, y_sup = select_supervised_samples(dataset,sample1)
    #print('Sup samples shape', X_sup.shape, y_sup.shape)
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batchs)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batchs / 2)
    print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batchs, half_batch, bat_per_epo, n_steps))
    # manually enumerate epochs
    for i in range(n_steps): 
        # update supervised discriminator (c)
        gc.collect()
        [Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], half_batch)
        c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
        
        # update unsupervised discriminator (d)
        [X_real, _], y_real = generate_real_samples1(dataset, half_batch)
        d_loss1 = d_model.train_on_batch(X_real, y_real)
 
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
   
        # update generator (g)
        X_gan, y_gan = generate_latent_points(latent_dim, n_batchs), ones((n_batchs, 1))
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        # summarize loss on this batch
        print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
        
        # evaluate the model performance every so often
        if (i+1) % (bat_per_epo * 1) == 0:
            summarize_performance(i, g_model, c_model, latent_dim, dataset)
            
def rescale(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min)

def summarize_performance(step, g_model, c_model, latent_dim, dataset, n_samples=16):
    # prepare fake examples
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot images
    fig=plt.figure(figsize=(20,24))
    for i in range(16):
        # define subplot
        pyplot.subplot(4, 4, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        #pyplot.imshow(X[i, 2, 1, 0])#, cmap='gray_r')
        
        arr = 255.0 * rescale(X[i])
        arr = arr[:,:,[2,1,0]]
        img = Img.fromarray(arr.astype('uint8'), 'RGB')
        #plt.title('Flower ' + str(int(plot_list[idx])))
        pyplot.imshow(img)

    # save plot to file
    filename1 = 'generated_plot_%04d_1.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # evaluate the classifier model
    _, Xtest, _, ytest = dataset
    _, acc = c_model.evaluate(Xtest, ytest, verbose=0)
    print('Classifier Accuracy: %.3f%%' % (acc * 100))
    # save the generator model
    #filename2 = f'g_model_%04d_GAN_{sample1}_{j}.h5' % (step+1)
    #g_model.save(filename2)
    # save the classifier model
    filename3 = f'c_model_%04d_GAN_{sample1}_{j}_111111.h5' % (step+1)
    c_model.save(filename3)
    print('>Saved: %s' % (filename3))

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

print(f'input images dimension is: {X111.shape}')
print(f'input response variable dimension is {yy.shape}')

# Load real samples
#dataset = load_real_samples(X111, yy)

# size of the latent space
latent_dim = 100
# create the discriminator models
d_model, c_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
#percents = (2400,)
percents = (30,60,100,300,900,1800,2400,3000)
batchs = (10,16,30,40,50,50,70,70)
#batchs = (10,20,30,50,50,50,50,50)
#batchs = (50,)

for j in range(3):
    for sample1,n_batchs in zip(percents,batchs):
        #load random split of data
        dataset = load_real_samples(X111,yy)
        # train model
        train(g_model, d_model, c_model, gan_model, dataset, latent_dim)

