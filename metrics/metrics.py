import numpy as np
import torch
import skimage
from skimage.metrics import structural_similarity as SSIM
from .inceptioninception import InceptionV3
from tqdm import tqdm
from torch.autograd import Variable
import os
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MAE
def compare_mae(real, fake):
    real = real.cpu().numpy()
    fake = fake.cpu().numpy()
    real, fake = real.astype(np.float32), fake.astype(np.float32)
    return np.sum(np.abs(real - fake)) / np.sum(real + fake)

###### Calculate metrics psnr ##########
def psnr(pred, target, pixel_max_cnt=255):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p

###### Calculate metrics ssim ##########
def ssim(pred, target):
    pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target[0]
    pred = pred[0]
    ssim = SSIM(target, pred, multichannel=True)
    return ssim


def fid(reals, fakes, num_worker=8, real_fid_path=None):
    
    dims = 2048
    batch_size = 4
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).cuda()

    if real_fid_path is None: 
        real_fid_path = 'places2_fid.pt'
        
    if os.path.isfile(real_fid_path): 
        data = pickle.load(open(real_fid_path, 'rb'))
        real_m, real_s = data['mu'], data['sigma']
    else: 
        reals = (np.array(reals).astype(np.float32) / 255.0).transpose((0, 3, 1, 2))
        real_m, real_s = calculate_activation_statistics(reals, model, batch_size, dims)
        with open(real_fid_path, 'wb') as f: 
            pickle.dump({'mu': real_m, 'sigma': real_s}, f)


    # calculate fid statistics for fake images
    fakes = (np.array(fakes).astype(np.float32) / 255.0).transpose((0, 3, 1, 2))
    fake_m, fake_s = calculate_activation_statistics(fakes, model, batch_size, dims)

    fid_value = calculate_frechet_distance(real_m, real_s, fake_m, fake_s)

    return fid_value


def calculate_activation_statistics(images, model, batch_size=64,
                                    dims=2048, cuda=True, verbose=False):

    act = get_activations(images, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def get_activations(images, model, batch_size=64, dims=2048, cuda=True, verbose=False):

    model.eval()

    d0 = images.shape[0]
    if batch_size > d0:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = d0

    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))
    for i in tqdm(range(n_batches), desc='Calculate activations'):
        if verbose:
            print('Propagating batch {%d}/{%d}'.format
                  (i + 1, n_batches), end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
        batch = Variable(batch)
        if torch.cuda.is_available:
            batch = batch.cuda()
        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)
    if verbose:
        print('Done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Implement of Frechet distance

    Args:
        mu1 ([type]): Numpy array containing the activations of a layer of the inception net
        sigma1 ([type]): The covariance matrix over activations for generated samples.
        mu2 ([type]): The sample mean over activations, precalculated on an representive data set.
        sigma2 ([type]): The covariance matrix over activations, precalculated on an representive data set.
        eps ([type], optional): [Epsilon]. Defaults to 1e-6.

    Raises:
        ValueError: [description]

    Returns:
        [type]: The Frechet Distance
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'The training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'The training and test covariances have different dimensions'
    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = np.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('FID calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = np.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)