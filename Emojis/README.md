# Emoji Data
#### Model 1 - Deep Convolution GAN (Generative Adversarial Networks)
- Generate Emojis from samples of random noise
- DCGAN is a GAN that uses a convolutional neural network as the discriminator, and a network composed of transposed convolutions as the generator. 
- Workflow
    - Have m training examples - x_i
    - Generate noise on m training examples - z_i
    - Generate fake images using the generator
    - Put the fake images in the discriminator - discriminator value towards 1 indicates real image and towards 0 indicates fake image
    - Compute the discriminator loss using Least Squares: (sum of (D(x_i) - 1)  ^ 2 + sum of (D(G(z_i)) ^ 2) / (2 * m)
    - Update parameters of discriminator
    - Minimize the discriminator loss
    - Compute the generator loss using Least Squares: (D(G(z_i) - 1) ^ 2) / m
    - Minimize the generator loss and update parameters of generator
    - Goal is to make discriminator predict correctly which image is fake or not so loss should be going to 0.
    - Goal of Generator - make generators generate images closest to real so D(G(z_i)) value should be approaching 1 hence, minimize loss
#### Model 2 - Cycle GAN (Generative Adversarial Networks)
- Designed for unpaired image to image translation.
- Allows for the transformation of images from one domain to another without requiring paired examples (i.e., a corresponding image in the target domain)
- Workflow
    - Have m training examples from Domain X - x_i
    - Have m training examples from Domain Y - y_i
    - Compute discriminator loss on real images
    - Compute discriminator loss on fake images
    - Update the discriminators
    - Compute Y->X generator loss
    - Compute X->Y generator loss
    - Update the generators
