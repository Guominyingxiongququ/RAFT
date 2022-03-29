# Copyright (c) OpenMMLab. All rights reserved.
import math

import cv2
import mmcv
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.special import gamma

def _cubic(x):
    """ Cubic function.

    Args:
        x (ndarray): The distance from the center position.

    Returns:
        ndarray: The weight corresponding to a particular distance.

    """

    x = np.array(x, dtype=np.float32)
    x_abs = np.abs(x)
    x_abs_sq = x_abs**2
    x_abs_cu = x_abs_sq * x_abs

    # if |x| <= 1: y = 1.5|x|^3 - 2.5|x|^2 + 1
    # if 1 < |x| <= 2: -0.5|x|^3 + 2.5|x|^2 - 4|x| + 2
    f = (1.5 * x_abs_cu - 2.5 * x_abs_sq + 1) * (x_abs <= 1) + (
        -0.5 * x_abs_cu + 2.5 * x_abs_sq - 4 * x_abs + 2) * ((1 < x_abs) &
                                                             (x_abs <= 2))

    return f

def sad(alpha, trimap, pred_alpha):
    if alpha.ndim != 2 or trimap.ndim != 2 or pred_alpha.ndim != 2:
        raise ValueError(
            'input alpha, trimap and pred_alpha should has two dimensions, '
            f'alpha {alpha.shape}, please check their shape: '
            f'trimap {trimap.shape}, pred_alpha {pred_alpha.shape}')
    assert (pred_alpha[trimap == 0] == 0).all()
    assert (pred_alpha[trimap == 255] == 255).all()
    alpha = alpha.astype(np.float64) / 255
    pred_alpha = pred_alpha.astype(np.float64) / 255
    sad_result = np.abs(pred_alpha - alpha).sum() / 1000
    return sad_result


def mse(alpha, trimap, pred_alpha):
    if alpha.ndim != 2 or trimap.ndim != 2 or pred_alpha.ndim != 2:
        raise ValueError(
            'input alpha, trimap and pred_alpha should has two dimensions, '
            f'alpha {alpha.shape}, please check their shape: '
            f'trimap {trimap.shape}, pred_alpha {pred_alpha.shape}')
    assert (pred_alpha[trimap == 0] == 0).all()
    assert (pred_alpha[trimap == 255] == 255).all()
    alpha = alpha.astype(np.float64) / 255
    pred_alpha = pred_alpha.astype(np.float64) / 255
    weight_sum = (trimap == 128).sum()
    if weight_sum != 0:
        mse_result = ((pred_alpha - alpha)**2).sum() / weight_sum
    else:
        mse_result = 0
    return mse_result


def gradient_error(alpha, trimap, pred_alpha, sigma=1.4):
    """Gradient error for evaluating alpha matte prediction.

    Args:
        alpha (ndarray): Ground-truth alpha matte.
        trimap (ndarray): Input trimap with its value in {0, 128, 255}.
        pred_alpha (ndarray): Predicted alpha matte.
        sigma (float): Standard deviation of the gaussian kernel. Default: 1.4.
    """
    if alpha.ndim != 2 or trimap.ndim != 2 or pred_alpha.ndim != 2:
        raise ValueError(
            'input alpha, trimap and pred_alpha should has two dimensions, '
            f'alpha {alpha.shape}, please check their shape: '
            f'trimap {trimap.shape}, pred_alpha {pred_alpha.shape}')
    if not ((pred_alpha[trimap == 0] == 0).all() and
            (pred_alpha[trimap == 255] == 255).all()):
        raise ValueError(
            'pred_alpha should be masked by trimap before evaluation')
    alpha = alpha.astype(np.float64)
    pred_alpha = pred_alpha.astype(np.float64)
    alpha_normed = np.zeros_like(alpha)
    pred_alpha_normed = np.zeros_like(pred_alpha)
    cv2.normalize(alpha, alpha_normed, 1., 0., cv2.NORM_MINMAX)
    cv2.normalize(pred_alpha, pred_alpha_normed, 1., 0., cv2.NORM_MINMAX)

    alpha_grad = gauss_gradient(alpha_normed, sigma).astype(np.float32)
    pred_alpha_grad = gauss_gradient(pred_alpha_normed,
                                     sigma).astype(np.float32)

    grad_loss = ((alpha_grad - pred_alpha_grad)**2 * (trimap == 128)).sum()
    # same as SAD, divide by 1000 to reduce the magnitude of the result
    return grad_loss / 1000


def connectivity(alpha, trimap, pred_alpha, step=0.1):
    """Connectivity error for evaluating alpha matte prediction.

    Args:
        alpha (ndarray): Ground-truth alpha matte with shape (height, width).
            Value range of alpha is [0, 255].
        trimap (ndarray): Input trimap with shape (height, width). Elements
            in trimap are one of {0, 128, 255}.
        pred_alpha (ndarray): Predicted alpha matte with shape (height, width).
            Value range of pred_alpha is [0, 255].
        step (float): Step of threshold when computing intersection between
            `alpha` and `pred_alpha`.
    """
    if alpha.ndim != 2 or trimap.ndim != 2 or pred_alpha.ndim != 2:
        raise ValueError(
            'input alpha, trimap and pred_alpha should has two dimensions, '
            f'alpha {alpha.shape}, please check their shape: '
            f'trimap {trimap.shape}, pred_alpha {pred_alpha.shape}')
    if not ((pred_alpha[trimap == 0] == 0).all() and
            (pred_alpha[trimap == 255] == 255).all()):
        raise ValueError(
            'pred_alpha should be masked by trimap before evaluation')
    alpha = alpha.astype(np.float32) / 255
    pred_alpha = pred_alpha.astype(np.float32) / 255

    thresh_steps = np.arange(0, 1 + step, step)
    round_down_map = -np.ones_like(alpha)
    for i in range(1, len(thresh_steps)):
        alpha_thresh = alpha >= thresh_steps[i]
        pred_alpha_thresh = pred_alpha >= thresh_steps[i]
        intersection = (alpha_thresh & pred_alpha_thresh).astype(np.uint8)

        # connected components
        _, output, stats, _ = cv2.connectedComponentsWithStats(
            intersection, connectivity=4)
        # start from 1 in dim 0 to exclude background
        size = stats[1:, -1]

        # largest connected component of the intersection
        omega = np.zeros_like(alpha)
        if len(size) != 0:
            max_id = np.argmax(size)
            # plus one to include background
            omega[output == max_id + 1] = 1

        mask = (round_down_map == -1) & (omega == 0)
        round_down_map[mask] = thresh_steps[i - 1]
    round_down_map[round_down_map == -1] = 1

    alpha_diff = alpha - round_down_map
    pred_alpha_diff = pred_alpha - round_down_map
    # only calculate difference larger than or equal to 0.15
    alpha_phi = 1 - alpha_diff * (alpha_diff >= 0.15)
    pred_alpha_phi = 1 - pred_alpha_diff * (pred_alpha_diff >= 0.15)

    connectivity_error = np.sum(
        np.abs(alpha_phi - pred_alpha_phi) * (trimap == 128))
    # same as SAD, divide by 1000 to reduce the magnitude of the result
    return connectivity_error / 1000


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def psnr(img1, img2, crop_border=0, input_order='HWC', convert_to=None):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PSNR calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: None.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
    if isinstance(convert_to, str) and convert_to.lower() == 'y':
        img1 = mmcv.bgr2ycbcr(img1 / 255., y_only=True) * 255.
        img2 = mmcv.bgr2ycbcr(img2 / 255., y_only=True) * 255.
    elif convert_to is not None:
        raise ValueError('Wrong color model. Supported values are '
                         '"Y" and None.')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    mse_value = np.mean((img1 - img2)**2)
    if mse_value == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse_value))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1, img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim(img1, img2, crop_border=0, input_order='HWC', convert_to=None):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the SSIM calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: None.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if isinstance(convert_to, str) and convert_to.lower() == 'y':
        img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
        img1 = mmcv.bgr2ycbcr(img1 / 255., y_only=True) * 255.
        img2 = mmcv.bgr2ycbcr(img2 / 255., y_only=True) * 255.
        img1 = np.expand_dims(img1, axis=2)
        img2 = np.expand_dims(img2, axis=2)
    elif convert_to is not None:
        raise ValueError('Wrong color model. Supported values are '
                         '"Y" and None')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


class L1Evaluation:
    """L1 evaluation metric.

    Args:
        data_dict (dict): Must contain keys of 'gt_img' and 'fake_res'. If
            'mask' is given, the results will be computed with mask as weight.
    """

    def __call__(self, data_dict):
        gt = data_dict['gt_img']
        pred = data_dict['fake_res']
        mask = data_dict.get('mask', None)

        from mmedit.models.losses.pixelwise_loss import l1_loss
        l1_error = l1_loss(pred, gt, weight=mask, reduction='mean')

        return l1_error


def estimate_aggd_param(block):
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters.

    Args:
        block (ndarray): 2D Image block.

    Returns:
        tuple: alpha (float), beta_l (float) and beta_r (float) for the AGGD
            distribution (Estimating the parames in Equation 7 in the paper).
    """
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)  # len = 9801
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (
        gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))

    left_std = np.sqrt(np.mean(block[block < 0]**2))
    right_std = np.sqrt(np.mean(block[block > 0]**2))
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(block)))**2 / np.mean(block**2)
    rhatnorm = (rhat * (gammahat**3 + 1) *
                (gammahat + 1)) / ((gammahat**2 + 1)**2)
    array_position = np.argmin((r_gam - rhatnorm)**2)

    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return (alpha, beta_l, beta_r)


def compute_feature(block):
    """Compute features.

    Args:
        block (ndarray): 2D Image block.

    Returns:
        list: Features with length of 18.
    """
    feat = []
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])

    # distortions disturb the fairly regular structure of natural images.
    # This deviation can be captured by analyzing the sample distribution of
    # the products of pairs of adjacent coefficients computed along
    # horizontal, vertical and diagonal orientations.
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for shift in shifts:
        shifted_block = np.roll(block, shift, axis=(0, 1))
        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted_block)
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])
    return feat


def niqe_core(img,
              mu_pris_param,
              cov_pris_param,
              gaussian_window,
              block_size_h=96,
              block_size_w=96):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.

    Ref: Making a "Completely Blind" Image Quality Analyzer.
    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

    Note that we do not include block overlap height and width, since they are
    always 0 in the official implementation.

    For good performance, it is advisable by the official implementation to
    divide the distorted image in to the same size patched as used for the
    construction of multivariate Gaussian model.

    Args:
        img (ndarray): Input image whose quality needs to be computed. The
            image must be a gray or Y (of YCbCr) image with shape (h, w).
            Range [0, 255] with float type.
        mu_pris_param (ndarray): Mean of a pre-defined multivariate Gaussian
            model calculated on the pristine dataset.
        cov_pris_param (ndarray): Covariance of a pre-defined multivariate
            Gaussian model calculated on the pristine dataset.
        gaussian_window (ndarray): A 7x7 Gaussian window used for smoothing the
            image.
        block_size_h (int): Height of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
        block_size_w (int): Width of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
    """
    # crop image
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

    distparam = []  # dist param is actually the multiscale features
    for scale in (1, 2):  # perform on two scales (1, 2)
        mu = convolve(img, gaussian_window, mode='nearest')

        sigma = np.sqrt(
            np.abs(
                convolve(np.square(img), gaussian_window, mode='nearest') -
                np.square(mu)))
        # normalize, as in Eq. 1 in the paper
        img_nomalized = (img - mu) / (sigma + 1)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                # process each block
                block = img_nomalized[idx_h * block_size_h //
                                      scale:(idx_h + 1) * block_size_h //
                                      scale, idx_w * block_size_w //
                                      scale:(idx_w + 1) * block_size_w //
                                      scale]
                feat.append(compute_feature(block))

        distparam.append(np.array(feat))

        # matlab-like bicubic downsample with anti-aliasing
        if scale == 1:
            resize = MATLABLikeResize(keys=None, scale=0.5)
            img = resize._resize(img[:, :, np.newaxis] / 255.)[:, :, 0] * 255.

    distparam = np.concatenate(distparam, axis=1)

    # fit a MVG (multivariate Gaussian) model to distorted patch features
    mu_distparam = np.nanmean(distparam, axis=0)
    distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
    cov_distparam = np.cov(distparam_no_nan, rowvar=False)

    # compute niqe quality, Eq. 10 in the paper
    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    quality = np.matmul(
        np.matmul((mu_pris_param - mu_distparam), invcov_param),
        np.transpose((mu_pris_param - mu_distparam)))

    return np.squeeze(np.sqrt(quality))


def niqe(img, crop_border, input_order='HWC', convert_to='y'):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.

    Ref: Making a "Completely Blind" Image Quality Analyzer.
    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

    We use the official params estimated from the pristine dataset.
    We use the recommended block size (96, 96) without overlaps.

    Args:
        img (ndarray): Input image whose quality needs to be computed.
            The input image must be in range [0, 255] with float/int type.
            The input_order of image can be 'HW' or 'HWC' or 'CHW'. (BGR order)
            If the input order is 'HWC' or 'CHW', it will be converted to gray
            or Y (of YCbCr) image according to the ``convert_to`` argument.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        input_order (str): Whether the input order is 'HW', 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether converted to 'y' (of MATLAB YCbCr) or 'gray'.
            Default: 'y'.

    Returns:
        float: NIQE result.
    """

    # we use the official params estimated from the pristine dataset.
    niqe_pris_params = np.load('mmedit/core/evaluation/niqe_pris_params.npz')
    mu_pris_param = niqe_pris_params['mu_pris_param']
    cov_pris_param = niqe_pris_params['cov_pris_param']
    gaussian_window = niqe_pris_params['gaussian_window']

    img = img.astype(np.float32)
    if input_order != 'HW':
        img = reorder_image(img, input_order=input_order)
        if convert_to == 'y':
            img = mmcv.bgr2ycbcr(img / 255., y_only=True) * 255.
        elif convert_to == 'gray':
            img = mmcv.bgr2gray(img / 255., cv2.COLOR_BGR2GRAY) * 255.
        img = np.squeeze(img)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border]

    # round to follow official implementation
    img = img.round()

    niqe_result = niqe_core(img, mu_pris_param, cov_pris_param,
                            gaussian_window)

    return niqe_result

def gaussian(x, sigma):
    """Gaussian function.

    Args:
        x (array_like): The independent variable.
        sigma (float): Standard deviation of the gaussian function.

    Return:
        ndarray or scalar: Gaussian value of `x`.
    """
    return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def dgaussian(x, sigma):
    """Gradient of gaussian.

    Args:
        x (array_like): The independent variable.
        sigma (float): Standard deviation of the gaussian function.

    Return:
        ndarray or scalar: Gradient of gaussian of `x`.
    """
    return -x * gaussian(x, sigma) / sigma**2


def gauss_filter(sigma, epsilon=1e-2):
    """Gradient of gaussian.

    Args:
        sigma (float): Standard deviation of the gaussian kernel.
        epsilon (float): Small value used when calculating kernel size.
            Default: 1e-2.

    Return:
        tuple[ndarray]: Gaussian filter along x and y axis.
    """
    half_size = np.ceil(
        sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
    size = np.int(2 * half_size + 1)

    # create filter in x axis
    filter_x = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            filter_x[i, j] = gaussian(i - half_size, sigma) * dgaussian(
                j - half_size, sigma)

    # normalize filter
    norm = np.sqrt((filter_x**2).sum())
    filter_x = filter_x / norm
    filter_y = np.transpose(filter_x)

    return filter_x, filter_y


def gauss_gradient(img, sigma):
    """Gaussian gradient.

    From https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/
    submissions/8060/versions/2/previews/gaussgradient/gaussgradient.m/
    index.html

    Args:
        img (ndarray): Input image.
        sigma (float): Standard deviation of the gaussian kernel.

    Return:
        ndarray: Gaussian gradient of input `img`.
    """
    filter_x, filter_y = gauss_filter(sigma)
    img_filtered_x = cv2.filter2D(
        img, -1, filter_x, borderType=cv2.BORDER_REPLICATE)
    img_filtered_y = cv2.filter2D(
        img, -1, filter_y, borderType=cv2.BORDER_REPLICATE)
    return np.sqrt(img_filtered_x**2 + img_filtered_y**2)

class MATLABLikeResize:
    """Resize the input image using MATLAB-like downsampling.

        Currently support bicubic interpolation only. Note that the output of
        this function is slightly different from the official MATLAB function.

        Required keys are the keys in attribute "keys". Added or modified keys
        are "scale" and "output_shape", and the keys in attribute "keys".

        Args:
            keys (list[str]): A list of keys whose values are modified.
            scale (float | None, optional): The scale factor of the resize
                operation. If None, it will be determined by output_shape.
                Default: None.
            output_shape (tuple(int) | None, optional): The size of the output
                image. If None, it will be determined by scale. Note that if
                scale is provided, output_shape will not be used.
                Default: None.
            kernel (str, optional): The kernel for the resize operation.
                Currently support 'bicubic' only. Default: 'bicubic'.
            kernel_width (float): The kernel width. Currently support 4.0 only.
                Default: 4.0.
    """

    def __init__(self,
                 keys,
                 scale=None,
                 output_shape=None,
                 kernel='bicubic',
                 kernel_width=4.0):

        if kernel.lower() != 'bicubic':
            raise ValueError('Currently support bicubic kernel only.')

        if float(kernel_width) != 4.0:
            raise ValueError('Current support only width=4 only.')

        if scale is None and output_shape is None:
            raise ValueError('"scale" and "output_shape" cannot be both None')

        self.kernel_func = _cubic
        self.keys = keys
        self.scale = scale
        self.output_shape = output_shape
        self.kernel = kernel
        self.kernel_width = kernel_width

    def _resize(self, img):
        weights = {}
        indices = {}

        # compute scale and output_size
        if self.scale is not None:
            scale = float(self.scale)
            scale = [scale, scale]
            output_size = get_size_from_scale(img.shape, scale)
        else:
            scale = get_scale_from_size(img.shape, self.output_shape)
            output_size = list(self.output_shape)

        # apply cubic interpolation along two dimensions
        order = np.argsort(np.array(scale))
        for k in range(2):
            key = (img.shape[k], output_size[k], scale[k], self.kernel_func,
                   self.kernel_width)
            weight, index = get_weights_indices(img.shape[k], output_size[k],
                                                scale[k], self.kernel_func,
                                                self.kernel_width)
            weights[key] = weight
            indices[key] = index

        output = np.copy(img)
        if output.ndim == 2:  # grayscale image
            output = output[:, :, np.newaxis]

        for k in range(2):
            dim = order[k]
            key = (img.shape[dim], output_size[dim], scale[dim],
                   self.kernel_func, self.kernel_width)
            output = resize_along_dim(output, weights[key], indices[key], dim)

        return output

    def __call__(self, results):
        for key in self.keys:
            is_single_image = False
            if isinstance(results[key], np.ndarray):
                is_single_image = True
                results[key] = [results[key]]

            results[key] = [self._resize(img) for img in results[key]]

            if is_single_image:
                results[key] = results[key][0]

        results['scale'] = self.scale
        results['output_shape'] = self.output_shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(keys={self.keys}, scale={self.scale}, '
            f'output_shape={self.output_shape}, '
            f'kernel={self.kernel}, kernel_width={self.kernel_width})')
        return repr_str

def resize_along_dim(img_in, weights, indices, dim):
    """Resize along a specific dimension.

    Args:
        img_in (ndarray): The input image.
        weights (ndarray): The weights used for interpolation, computed from
            [get_weights_indices].
        indices (ndarray): The indices used for interpolation, computed from
            [get_weights_indices].
        dim (int): Which dimension to undergo interpolation.

    Returns:
        ndarray: Interpolated (along one dimension) image.
    """

    img_in = img_in.astype(np.float32)
    w_shape = weights.shape
    output_shape = list(img_in.shape)
    output_shape[dim] = w_shape[0]
    img_out = np.zeros(output_shape)

    if dim == 0:
        for i in range(w_shape[0]):
            w = weights[i, :][np.newaxis, ...]
            ind = indices[i, :]
            img_slice = img_in[ind, :]
            img_out[i] = np.sum(np.squeeze(img_slice, axis=0) * w.T, axis=0)
    elif dim == 1:
        for i in range(w_shape[0]):
            w = weights[i, :][:, :, np.newaxis]
            ind = indices[i, :]
            img_slice = img_in[:, ind]
            img_out[:, i] = np.sum(np.squeeze(img_slice, axis=1) * w.T, axis=1)

    if img_in.dtype == np.uint8:
        img_out = np.clip(img_out, 0, 255)
        return np.around(img_out).astype(np.uint8)
    else:
        return img_out

def get_size_from_scale(input_size, scale_factor):
    """Get the output size given input size and scale factor.

    Args:
        input_size (tuple): The size of the input image.
        scale_factor (float): The resize factor.

    Returns:
        list[int]: The size of the output image.
    """

    output_shape = [
        int(np.ceil(scale * shape))
        for (scale, shape) in zip(scale_factor, input_size)
    ]

    return output_shape

def get_scale_from_size(input_size, output_size):
    """Get the scale factor given input size and output size.

    Args:
        input_size (tuple(int)): The size of the input image.
        output_size (tuple(int)): The size of the output image.

    Returns:
        list[float]: The scale factor of each dimension.
    """

    scale = [
        1.0 * output_shape / input_shape
        for (input_shape, output_shape) in zip(input_size, output_size)
    ]

    return scale

def get_weights_indices(input_length, output_length, scale, kernel,
                        kernel_width):
    """Get weights and indices for interpolation.

    Args:
        input_length (int): Length of the input sequence.
        output_length (int): Length of the output sequence.
        scale (float): Scale factor.
        kernel (func): The kernel used for resizing.
        kernel_width (int): The width of the kernel.

    Returns:
        list[ndarray]: The weights and the indices for interpolation.


    """
    if scale < 1:  # modified kernel for antialiasing

        def h(x):
            return scale * kernel(scale * x)

        kernel_width = 1.0 * kernel_width / scale
    else:
        h = kernel
        kernel_width = kernel_width

    # coordinates of output
    x = np.arange(1, output_length + 1).astype(np.float32)

    # coordinates of input
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)  # leftmost pixel
    p = int(np.ceil(kernel_width)) + 2  # maximum number of pixels

    # indices of input pixels
    ind = left[:, np.newaxis, ...] + np.arange(p)
    indices = ind.astype(np.int32)

    # weights of input pixels
    weights = h(u[:, np.newaxis, ...] - indices - 1)

    weights = weights / np.sum(weights, axis=1)[:, np.newaxis, ...]

    # remove all-zero columns
    aux = np.concatenate(
        (np.arange(input_length), np.arange(input_length - 1, -1,
                                            step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]

    return weights, indices