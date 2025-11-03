from types import DynamicClassAttribute
import torch
import torchvision
import cv2
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def chain_rule():
    """
    Compute df/dz, df/dq, df/dx, and df/dy for f(x,y,z)=xy+z,
    where q=xy, at x=-2, y=5, z=-4.
    Return them in this order: df/dz, df/dq, df/dx, df/dy. 
    """ 
    x = -2.0
    y = 5.0

    df_dz = 1.0
    df_dq = 1.0
    df_dx = y
    df_dy = x

    return df_dz, df_dq, df_dx, df_dy 

def ReLU():
    """
    Compute dx and dw, and return them in order.
    Forward:
        y = ReLU(w0 * x0 + w1 * x1 + w2)

    Returns:
        dx -- gradient with respect to input x, as a vector [dx0, dx1]
        dw -- gradient with respect to weights (including the third term w2), 
              as a vector [dw0, dw1, dw2]
    """
    w0 = 2.0
    w1 = -3.0
    w2 = -3.0

    x0 = -1.0
    x1 = -2.0
    
    relu_output = max(0, w0 * x0 + w1 * x1 + w2)
    drelu_output = 1 if relu_output > 0 else 0

    doutput_dx0 = drelu_output * w0
    doutput_dx1 = drelu_output * w1

    doutput_dw0 = drelu_output * x0
    doutput_dw1 = drelu_output * x1
    doutput_dw2 = drelu_output * 1

    dx = [doutput_dx0, doutput_dx1]
    dw = [doutput_dw0, doutput_dw1, doutput_dw2]

    return dx, dw

def chain_rule_a():
    """
    In the lecture notes, the last three forward pass values are 
    a=0.37, b=1.37, and c=0.73.  
    Calculate these numbers to 4 decimal digits and return in order of a, b, c
    """
    w0 = 2.0
    w1 = -3.0
    w2 = -3.0

    x0 = -1.0
    x1 = -2.0

    q = -(w0 * x0 + w1 * x1 + w2)
    a = np.exp(q)
    b = 1 + a
    c = 1 / b

    a = float(round(a, 4))
    b = float(round(b, 4))
    c = float(round(c, 4))

    print(a, b, c)

    return a, b, c

def chain_rule_b():
    """
    In the lecture notes, the backward pass values are
    ±0.20, ±0.39, -0.59, and -0.53.  
    Calculate these numbers to 4 decimal digits 
    and return in order of gradients for w0, x0, w1, x1, w2.
    """
    w0 = torch.tensor(2.0, requires_grad=True)
    w1 = torch.tensor(-3.0, requires_grad=True)
    w2 = torch.tensor(-3.0, requires_grad=True)

    x0 = torch.tensor(-1.0, requires_grad=True)
    x1 = torch.tensor(-2.0, requires_grad=True)

    y = 1 / (1 + torch.exp(-(w0 * x0 + w1 * x1 + w2)))
    y.backward()

    gw0 = round(w0.grad.item(), 4) if w0.grad is not None else 0.0
    gw1 = round(w1.grad.item(), 4) if w1.grad is not None else 0.0
    gw2 = round(w2.grad.item(), 4) if w2.grad is not None else 0.0
    gx0 = round(x0.grad.item(), 4) if x0.grad is not None else 0.0
    gx1 = round(x1.grad.item(), 4) if x1.grad is not None else 0.0
    
    print(gw0, gx0, gw1, gx1, gw2)

    return torch.tensor([gw0, gx0, gw1, gx1, gw2])

class tanhNN(nn.Module):
    def __init__(self, w_int=(5.0, 2.0), b_init=-2.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(list(w_int)))
        self.b = nn.Parameter(torch.tensor(b_init))
    def forward(self, x):
        return torch.tanh(self.w @ x + self.b)

def backprop_a():
    """
    Let f(w,x) = torch.tanh(w0x0+w1x1+w2).  
    Assume the weight vector is w = [w0=5, w1=2], 
    the input vector is  x = [x0=-1,x1= 4],, and the bias is  w2  =-2.
    Use PyTorch to calculate the forward pass of the network, return y_hat = f(w,x).
    """
    w = torch.tensor([5.0, 2.0, -2.-0], requires_grad=True)
    x = torch.tensor([-1.0, 4.0], requires_grad=True)

    y_hat = torch.tanh(w[0] * x[0] + w[1] * x[1] + w[2])
    return y_hat

def backprop_b():
    """
    Use PyTorch Autograd to calculate the gradients 
    for each of the weights, and return the gradient of them 
    in order of w0, w1, and w2.
    """
    w = torch.tensor([5.0, 2.0, -2.-0], requires_grad=True)
    x = torch.tensor([-1.0, 4.0], requires_grad=True)

    y_hat = torch.tanh(w[0] * x[0] + w[1] * x[1] + w[2])
    ground_truth = torch.tensor(1.0)
    loss = (y_hat - ground_truth) ** 2
    loss.backward()

    gw0 = w.grad[0]
    gw1 = w.grad[1]
    gw2 = w.grad[2]

    return gw0, gw1, gw2

def backprop_c():
    """
    Assuming a learning rate of 0.1, 
    update each of the weights accordingly. 
    For simplicity, just do one iteration. 
    And return the updated weights in the order of w0, w1, and w2 
    """
    lr = torch.tensor(0.1)
    w = torch.tensor([5.0, 2.0, -2.-0], requires_grad=True)
    x = torch.tensor([-1.0, 4.0], requires_grad=True)

    y_hat = torch.tanh(w[0] * x[0] + w[1] * x[1] + w[2])
    ground_truth = torch.tensor(1.0)
    loss = (y_hat - ground_truth) ** 2
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad if w.grad is not None else 0.0

    w0 = w[0]
    w1 = w[1]
    w2 = w[2]

    return  w0, w1, w2 


def constructParaboloid(w=256, h=256):
    img = np.zeros((w, h), np.float32)
    for x in range(w):
        for y in range(h):
            # let's center the paraboloid in the img
            img[y, x] = (x - w / 2) ** 2 + (y - h / 2) ** 2
    return img


def newtonMethod(x0, y0):
    paraboloid = torch.tensor([constructParaboloid()]).squeeze()
    paraboloid = torch.unsqueeze(paraboloid, 0) 
    paraboloid = torch.unsqueeze(paraboloid, 0)   

    """
    Insert your code here
    """

    paraboloid = paraboloid.to(torch.float32).contiguous()   
    _, _, H, W = paraboloid.shape
    device = paraboloid.device
    dtype  = paraboloid.dtype

    kx1 = torch.tensor([[-0.5, 0.0, 0.5]], dtype=dtype, device=device).view(1,1,1,3)  
    ky1 = torch.tensor([[-0.5], [0.0], [0.5]], dtype=dtype, device=device).view(1,1,3,1)  
    kx2 = torch.tensor([[ 1.0, -2.0, 1.0 ]], dtype=dtype, device=device).view(1,1,1,3)    
    ky2 = torch.tensor([[ 1.0], [-2.0], [1.0]], dtype=dtype, device=device).view(1,1,3,1) 

    Ix  = F.conv2d(F.pad(paraboloid, (1,1,0,0), mode="replicate"), kx1)  
    Iy  = F.conv2d(F.pad(paraboloid, (0,0,1,1), mode="replicate"), ky1)  
    Ixx = F.conv2d(F.pad(paraboloid, (1,1,0,0), mode="replicate"), kx2)  
    Iyy = F.conv2d(F.pad(paraboloid, (0,0,1,1), mode="replicate"), ky2)  
    Ixy = F.conv2d(F.pad(Ix, (0,0,1,1), mode="replicate"), ky1)  

    x = torch.as_tensor(float(x0), dtype=dtype, device=device)   
    y = torch.as_tensor(float(y0), dtype=dtype, device=device)   

    max_iter = 50                      
    tol = 1e-6                          
    damping = 1.0                       
    epsI = torch.eye(2, dtype=dtype, device=device) * 1e-9   

    for _ in range(max_iter):
        xi = torch.round(x).to(torch.long)
        yi = torch.round(y).to(torch.long)

        xi = xi.clamp(0, W - 1)
        yi = yi.clamp(0, H - 1)

        gx  = Ix[0, 0, yi, xi]
        gy  = Iy[0, 0, yi, xi]
        H11 = Ixx[0, 0, yi, xi]
        H22 = Iyy[0, 0, yi, xi]
        H12 = Ixy[0, 0, yi, xi]

        g = torch.stack([gx, gy])
        Hmat = torch.stack([
            torch.stack([H11, H12]),
            torch.stack([H12, H22])
        ]) + epsI

        delta = torch.linalg.solve(Hmat, g)
        x_new = x - damping * delta[0]
        y_new = y - damping * delta[1]

        step = torch.hypot(x_new - x, y_new - y)
        x, y = x_new, y_new
        if step < tol:
            break


    final_x = x.item()
    final_y = y.item()
    return final_x, final_y


def sgd(x0, y0, lr=0.001):
    paraboloid = torch.tensor([constructParaboloid()]).squeeze()
    paraboloid = torch.unsqueeze(paraboloid, 0)
    paraboloid = torch.unsqueeze(paraboloid, 0)

    """
    Insert your code here
    """
    paraboloid = paraboloid.to(torch.float32).contiguous()  
    _, _, H, W = paraboloid.shape
    device = paraboloid.device
    dtype  = paraboloid.dtype

    kx1 = torch.tensor([[-0.5, 0.0, 0.5]], dtype=dtype, device=device).view(1,1,1,3)  
    ky1 = torch.tensor([[-0.5],[0.0],[0.5]], dtype=dtype, device=device).view(1,1,3,1) 

    Ix = F.conv2d(F.pad(paraboloid, (1,1,0,0), mode="replicate"), kx1)  
    Iy = F.conv2d(F.pad(paraboloid, (0,0,1,1), mode="replicate"), ky1)  

    lr = torch.tensor(lr, dtype=dtype, device=device)
    tol = torch.tensor(1e-6, dtype=dtype, device=device)

    x = torch.as_tensor(float(x0), dtype=dtype, device=device)
    y = torch.as_tensor(float(y0), dtype=dtype, device=device)

    max_iter = 5000
    batch_sz = 1000
    jitter_sd = 0.1    

    for time in range(max_iter):
        bx = (x + torch.randn(batch_sz, dtype=dtype, device=device) * jitter_sd).round().long().clamp(0, W-1)
        by = (y + torch.randn(batch_sz, dtype=dtype, device=device) * jitter_sd).round().long().clamp(0, H-1)
        gx = Ix[0,0,by, bx].mean()
        gy = Iy[0,0,by, bx].mean()

        x_new = x - lr * gx
        y_new = y - lr * gy

        step = torch.hypot(x_new - x, y_new - y)
        x, y = x_new, y_new
        if step < tol:
            print("converged at iteration: ", time)
            break

        x = x.clamp(0.0, W - 1.0)
        y = y.clamp(0.0, H - 1.0)

   
    final_x = x.item()
    final_y = y.item()
    return final_x, final_y

if __name__ == "__main__":
    backprop_a()
    backprop_b()
    backprop_c()
    newtonMethod(128, 128)
    sgd(12, 12, lr=1)
    # chain_rule_a()
    # chain_rule_b()

