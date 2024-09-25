# import the necessary packages
import os
import gc
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
from numpy import ones
from PIL import Image as Img
import tensorflow as tf

from directories import dir_img, dir_gt, dir_out
from data_loader import load_dataset
from models import define_discriminator, define_generator, define_gan
from gan_utils import load_real_samples, select_supervised_samples, generate_real_samples, generate_real_samples1, generate_latent_points, generate_fake_samples

try:
    # Ensure TensorFlow uses the GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    def check_gpu():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print("GPUs are available:")
            for gpu in gpus:
                print(f"  {gpu}")
        else:
            print("No GPUs found.")
                
    check_gpu()

    #  train the models
    def train_gan(g_model, d_model, c_model, gan_model, dataset, latent_dim, n_epochs=50):
        
        print(n_batchs)
        X_sup, y_sup = select_supervised_samples(dataset, sample1)
        bat_per_epo = int(dataset[0].shape[0] / n_batchs)
        n_steps = bat_per_epo * n_epochs
        half_batch = int(n_batchs / 2)
        
        # Print the number of epochs, batches, steps, and half batch
        print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batchs, half_batch, bat_per_epo, n_steps))
        
        for i in range(n_steps):
            
            # Define file names based on the current step
            filename1 = 'generated_plot_%04d_1.png' % (i+1)
            filename3 = f'c_model_%04d_ESGAN_{sample1}_{j}.h5' % (i+1)
            
            # Check if files already exist, if so it skips the iteration
            if os.path.exists(filename1) and os.path.exists(filename3):
                print(f'Skipping iteration {i+1} as files already exist.')
                continue
            
            # Proceed with training if files do not exist
            gc.collect()
            
            # load real samples and train supervised classifier (flowering status)
            [Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], half_batch)
            c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
            
            # load real samples and train unsupervised classifier (original gan real/fake)
            [X_real, _], y_real = generate_real_samples1(dataset, half_batch)
            d_loss1 = d_model.train_on_batch(X_real, y_real)
            
            # generate fake samples and train unsupervised classifier (original gan real/fake)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss2 = d_model.train_on_batch(X_fake, y_fake)
            
            # prepare points in latent space as input for train generator
            X_gan, y_gan = generate_latent_points(latent_dim, n_batchs), np.ones((n_batchs, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            
            #print performance during training for each component of gan at each step
            print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
            if (i+1) % (bat_per_epo * 1) == 0:
                summarize_performance(i, g_model, c_model, latent_dim, dataset)

    def rescale(arr):
        arr_min = arr.min()
        arr_max = arr.max()
        return (arr - arr_min) / (arr_max - arr_min)

    # function that help visualize the convergence of the generator ability to create 'realistic' fake images
    def summarize_performance(step, g_model, c_model, latent_dim, dataset, n_samples=16):
        X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
        X = (X + 1) / 2.0
        fig = plt.figure(figsize=(20,24))
        for i in range(16):
            plt.subplot(4, 4, 1 + i)
            plt.axis('off')
            arr = 255.0 * rescale(X[i])
            arr = arr[:, :, [2,1,0]]
            img = Img.fromarray(arr.astype('uint8'), 'RGB')
            plt.imshow(img)
        filename1 = f'generated_plot_%04d_ESGAN_{sample1}_{j}.png' % (step+1)
        plt.savefig(filename1)
        plt.close()
        _, Xtest, _, ytest = dataset
        _, acc = c_model.evaluate(Xtest, ytest, verbose=0)
        print('Classifier Accuracy: %.3f%%' % (acc * 100))
        filename3 = f'c_model_%04d_ESGAN_{sample1}_{j}.h5' % (step+1)
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


    # implement training of the GAN setting parameters and varying the amount of 'labeled' data to GAN training 
    os.chdir(dir_out)
    
    #load images and corresponding ground truth
    X, y = load_dataset(dir_img, dir_gt)

    # set latent space size for generator
    latent_dim = 100
    
    # define models
    d_model, c_model = define_discriminator()
    g_model = define_generator(latent_dim)
    gan_model = define_gan(g_model, d_model)
    
    #varying labeled samples and batch size for GAN training according to the paper def, testing data size is kept fixed for all percentages
    
    # 30= 1%, 60=2%, 100=3%, 300=10%, 900=30%, 1800=60%, 2400=80%, 3000=100%
    percents = (30,60,100,300,900,1800,2400,3000) 
    
    # batch size for training data should be assigned according since it's taken from the amount of training labeled data available 
    batchs = (10,16,30,40,50,50,70,70) 

    # training the gan model for each percentage of labeled data
    for j in range(5):
        for sample1, n_batchs in zip(percents, batchs):
            
            dataset = load_real_samples(X, y)
            
            print(f'Training on iteration {j}, sample {sample1}, and batch {n_batchs}')
            
            try:
                train_gan(g_model, d_model, c_model, gan_model, dataset, latent_dim)
            except KeyboardInterrupt:
                print("Training interrupted. Exiting gracefully...")

except KeyboardInterrupt:
    print("Training interrupted. Exiting gracefully...")
