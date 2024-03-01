import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

def cal_transmission2(haze_img, t, lambda_val, param):
    n_rows, n_cols = t.shape
    nsz = 3
    num = nsz * nsz
    
    d = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
        np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
        np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    ]
    
    for k in range(len(d)):
        d[k] = d[k] / np.linalg.norm(d[k])
    
    num_filters = len(d)
    w_fun = np.zeros((n_rows, n_cols))
    
    for k in range(num_filters):
        w_fun_k = cal_weight_fun(haze_img, d[k], param)
        w_fun += w_fun_k / num_filters
    
    tf = fft2(t)
    ds = np.zeros((n_rows, n_cols))
    
    for k in range(num_filters):
        d_k = np.fft.fft2(np.flipud(np.fliplr(d[k])), s=(n_rows, n_cols))
        ds += np.abs(d_k) ** 2
    
    beta = 1
    beta_rate = 2 * np.sqrt(2)
    beta_max = 2**8
    out_iter = 0
    
    while beta < beta_max:
        gamma = lambda_val / beta
        out_iter += 1
        du = 0
        
        for k in range(num_filters):
            dt_k = convolve2d(t, d[k], mode='same', boundary='wrap')
            u_k = np.maximum(np.abs(dt_k) - w_fun / beta / num_filters, 0) * np.sign(dt_k)
            du += fft2(convolve2d(u_k, np.flipud(np.fliplr(d[k])), mode='same', boundary='wrap'))
        
        t = np.abs(ifft2((gamma * tf + du) / (gamma + ds)))
        beta *= beta_rate
    
    return t

def cal_weight_fun(haze_img, D, param):
    sigma = param
    method = 'wrap'
    haze_img = haze_img / 255.0
    d_r = convolve2d(haze_img[:, :, 0], D, mode='same', boundary=method)
    d_g = convolve2d(haze_img[:, :, 1], D, mode='same', boundary=method)
    d_b = convolve2d(haze_img[:, :, 2], D, mode='same', boundary=method)
    w_fun = np.exp(-(d_r**2 + d_g**2 + d_b**2) / (2 * sigma))
    return w_fun
