import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
import scipy.stats as st

def loss_fun(model, x, y):
    p = model.net(x.to(model.device))
    loss = torch.nn.CrossEntropyLoss()(p, y.to(model.device))
    return loss

def gkern(kernlen, nsig):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def lkern(kernlen, alpha=0.5, beta=0.5):
    """Returns a 2D Laplacian kernel array."""
    # Generate a 1D kernel
    x = np.linspace(-1, 1, kernlen)
    kern1d = np.zeros_like(x)
    kern1d[x == 0] = -2 * alpha
    kern1d[x != 0] = beta / (np.abs(x[x != 0]) ** alpha)
    # Outer product of the 1D kernel with itself to get a 2D kernel
    kernel = np.outer(kern1d, kern1d)
    # Normalize the kernel
    kernel = kernel / np.sum(np.abs(kernel))
    return kernel

def Translation_Invariant_Method(input, grad, kernel_size=5):
    
    # Generate Gaussian kernel
    sigma = kernel_size / np.sqrt(3)
    gaussian_kernel = torch.from_numpy(gkern(kernel_size, sigma)).type_as(input)
    # Reshape the kernel
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(input.size(1), 1, 1, 1)

    # Generate Laplacian kernel
    laplacian_kernel = torch.from_numpy(lkern(kernel_size)).type_as(input)

    laplacian_kernel = laplacian_kernel.view(1, 1, kernel_size, kernel_size)
    laplacian_kernel = laplacian_kernel.repeat(input.size(1), 1, 1, 1)

    # Define the convolution layer
    conv_layer = nn.Conv2d(in_channels=input.size(1),
                           out_channels=input.size(1),
                           kernel_size=kernel_size,
                           groups=input.size(1),
                           bias=False,
                           padding=(kernel_size - 1) // 2)
    
    # Initialize the convolution layer
    with torch.no_grad():
        conv_layer.weight.data = gaussian_kernel
        # conv_layer.weight.data = laplacian_kernel
        conv_layer.weight.requires_grad = False

    output = conv_layer(grad)
    return output

def ti_mi_fgsm_attack(model, image, label, epsilon, num_iterations = 10, momentum=1):

    # Initialize the adversarial example to the input data
    image = image.cuda()
    alpha = epsilon / num_iterations
    adv_image = image + torch.Tensor(np.random.uniform(-alpha, alpha, image.shape)).type_as(image)
    g = torch.zeros(image.size(0), 1, 1, 1).cuda()

    # Loop over the number of iterations
    for _ in range(num_iterations):

        # Compute the gradient 
        adv_image = adv_image.detach()
        adv_image.requires_grad = True
        model.net.zero_grad()
        loss = loss_fun(model, adv_image, label)
        loss.backward()
        grad = adv_image.grad.data

        # TIM
        grad = Translation_Invariant_Method(adv_image,grad)

        g = momentum * g + (grad/(torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True))).cuda()

        sign_grad = g.sign()
        adv_image = adv_image + alpha * sign_grad
        
        adv_image[:,0,:,:] = torch.min(torch.max(adv_image[:,0,:,:], image[:,0,:,:] - epsilon), image[:,0,:,:] + epsilon)
        adv_image[:,1,:,:] = torch.min(torch.max(adv_image[:,1,:,:], image[:,1,:,:] - epsilon), image[:,1,:,:] + epsilon)

        adv_image[:,0,:,:] = torch.clamp(adv_image[:,0,:,:], torch.min(image[:,0,:,:]), torch.max(image[:,0,:,:]))
        adv_image[:,1,:,:] = torch.clamp(adv_image[:,1,:,:], torch.min(image[:,1,:,:]), torch.max(image[:,1,:,:]))
    # Return the adversarial example
    return adv_image
