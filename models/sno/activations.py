import torch

def softplus(x, alpha=1.0, cut_off=35):
    a = (alpha*x < cut_off)
    return a*torch.log(1 + torch.exp(alpha*x*a))/alpha + torch.logical_not(a) * x

def complex_split_relu(x):
    return torch.real(x)*(torch.real(x)>0) + 1j*torch.imag(x)*(torch.imag(x)>0)

def complex_split_softplus(x, alpha=1.0):
    return softplus(torch.real(x), alpha=alpha) + 1j*softplus(torch.imag(x), alpha=alpha)

def complex_mul(equation, x, y):
    return torch.einsum(equation, x.real, y.real) - torch.einsum(equation, x.imag, y.imag) \
             + 1j * (torch.einsum(equation, x.imag, y.real) + torch.einsum(equation, x.real, y.imag))