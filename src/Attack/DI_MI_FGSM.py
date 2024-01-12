import numpy as np
import torch
import torch.nn.functional as F
import random

random.seed(18)

def loss_fun(model, x, y):
    p = model.net(x.to(model.device))
    loss = torch.nn.CrossEntropyLoss()(p, y.to(model.device))
    return loss

def input_diversity(input_tensor, image_width = 88, image_resize = 97, prob = 0.5):
    rnd = torch.randint(low=image_width, high=image_resize, size=(1,))
    rescaled = torch.nn.functional.interpolate(input_tensor, size=rnd.item(), mode='nearest')
    h_rem = image_resize - rnd
    w_rem = image_resize - rnd
   
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,))
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,))
    pad_right = w_rem - pad_left
    padded = torch.nn.functional.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom, 0, 0), mode='constant', value=0.)
    
    resize = torch.nn.functional.interpolate(padded, size=(image_width, image_width), mode='nearest')

    if random.random()<prob:
        return resize
    else:
        return input_tensor

def di_mi_fgsm_attack(model, image, label, epsilon, num_iterations = 10, momentum=1):

    # Initialize the adversarial example to the input data
    image = image.cuda()
    alpha = epsilon / num_iterations
    adv_image = image + torch.Tensor(np.random.uniform(-alpha, alpha, image.shape)).type_as(image)
    
    g = torch.zeros(image.size(0), 1, 1, 1).cuda()

    # Loop over the number of iterations
    for _ in range(num_iterations):

        # Compute the loss
        adv_image = adv_image.detach()
        adv_image.requires_grad = True
        model.net.zero_grad()
        loss = loss_fun(model, adv_image, label)
        loss.backward()
        grad = adv_image.grad.data

        # Get the gradient
        g = momentum * g + (grad/(torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True))).cuda()

        sign_grad = g.sign()
        adv_image = adv_image + alpha * sign_grad
        
        adv_image[:,0,:,:] = torch.min(torch.max(adv_image[:,0,:,:], image[:,0,:,:] - epsilon), image[:,0,:,:] + epsilon)
        adv_image[:,1,:,:] = torch.min(torch.max(adv_image[:,1,:,:], image[:,1,:,:] - epsilon), image[:,1,:,:] + epsilon)

        adv_image[:,0,:,:] = torch.clamp(adv_image[:,0,:,:], torch.min(image[:,0,:,:]), torch.max(image[:,0,:,:]))
        adv_image[:,1,:,:] = torch.clamp(adv_image[:,1,:,:], torch.min(image[:,1,:,:]), torch.max(image[:,1,:,:]))
    # Return the adversarial example
    return adv_image