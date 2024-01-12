import numpy as np
import torch

def mi_fgsm_attack(model, image, label, epsilon, num_iterations = 10, momentum=1):

    # Initialize the adversarial example to the input data
    image = image.to(model.device)
    alpha = epsilon / num_iterations
    adv_image = image + torch.Tensor(np.random.uniform(-alpha, alpha, image.shape)).type_as(image)

    g = torch.zeros(image.size(0), 1, 1, 1).cuda()

    # Loop over the number of iterations
    for _ in range(num_iterations):

        # Compute the loss
        adv_image = adv_image.detach()
        adv_image.requires_grad = True
        model.optimize_attack(adv_image,label)

        # Get the gradient
        grad = adv_image.grad.data
        g = momentum * g + (grad/(torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True))).cuda()

        sign_grad = g.sign()
        adv_image = adv_image + alpha * sign_grad
        
        adv_image[:,0,:,:] = torch.min(torch.max(adv_image[:,0,:,:], image[:,0,:,:] - epsilon), image[:,0,:,:] + epsilon)
        adv_image[:,1,:,:] = torch.min(torch.max(adv_image[:,1,:,:], image[:,1,:,:] - epsilon), image[:,1,:,:] + epsilon)

        adv_image[:,0,:,:] = torch.clamp(adv_image[:,0,:,:], torch.min(image[:,0,:,:]), torch.max(image[:,0,:,:]))
        adv_image[:,1,:,:] = torch.clamp(adv_image[:,1,:,:], torch.min(image[:,1,:,:]), torch.max(image[:,1,:,:]))

    # Return the adversarial example
    return adv_image
