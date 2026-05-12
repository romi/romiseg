#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

import cv2
import numpy as np
from matplotlib import pyplot as plt


def exgreen(im_BGR, cvtype=False):
    """Extracts the green channel from a BGR image and optionally converts the output to grayscale.

    Parameters
    ----------
    im_BGR : numpy.ndarray
        An image represented as a 3D NumPy array in BGR format.
    cvtype : bool, optional
        If True, the output will be converted to grayscale format. If False,
        the function will return the extracted green channel without further
        conversion. The default value is False.

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array representing the green channel of the input image,
        optionally converted to grayscale if `cvtype` is True.
    """
    Ms = np.max(im_BGR, axis=(0, 1)).astype(np.float)
    im_Norm = im_BGR / Ms
    L = im_Norm.sum(axis=2)
    res = 3 * im_Norm[:, :, 1] / L - 1
    if cvtype:
        M = res.max()
        m = res.min()
        res = (255 * (res - m) / (M - m)).astype(np.uint8)
    return res


def fill_npoints(xy, Np):
    """Fills a given number of points (Np) based on the input coordinates (xy).

    This function takes a set of coordinates (xy) and a specified number of points
    (Np), and processes or maps the points accordingly. The exact behavior on how
    the points are filled depends on the function's implementation details, which
    are not outlined here.

    Parameters
    ----------
    xy : list of tuple or list of list
        A collection of coordinates, with each coordinate represented as a tuple
        or list containing x and y values.
    Np : int
        The total number of points to be filled or processed.
    """
    ts = np.linspace(0, len(xy[0]), num=len(xy[0]), endpoint=True)
    nts = np.linspace(0, len(xy[0]), num=Np, endpoint=True)
    fx = np.interp(nts, ts, xy[0])
    fy = np.interp(nts, ts, xy[1])
    return np.array([fx, fy])


def get_edge_grad(im, sig):
    """Calculates the edge gradients of an image using Gaussian smoothing.

    This function computes the gradients of an input image by applying Gaussian
    smoothing. The first gradient of the image is computed after convolving
    with a Gaussian filter of standard deviation `sig`. This is often used
    in image processing to identify edge structures.

    Parameters
    ----------
    im : ndarray
        Input image in the form of a 2D array for which edge gradients will
        be calculated.
    sig : float
        Standard deviation of the Gaussian filter for smoothing the image
        before calculating gradients.

    Returns
    -------
    gradx : ndarray
        Gradient of the image along the x-axis after Gaussian smoothing.
    grady : ndarray
        Gradient of the image along the y-axis after Gaussian smoothing.

    Raises
    ------
    ValueError
        If the input image `im` is not a 2D array or if `sig` is not a
        positive number.
    """
    im = cv2.GaussianBlur(im, (17, 17), sig)
    sc_x = cv2.Scharr(im, cv2.CV_64F, 1, 0)
    sc_y = cv2.Scharr(im, cv2.CV_64F, 0, 1)
    F_norm = np.sqrt(sc_x ** 2 + sc_y ** 2)
    F_norm = ((F_norm - F_norm.min()) / (F_norm.max() - F_norm.min()))
    F_norm_xy = np.array(np.gradient(F_norm))
    return F_norm_xy


def close_cont(cont, d):
    """Closes or terminates the specified container, performing necessary cleanup or finalization as required.

    This function ensures that the given container is properly closed or terminated,
    handling the specified parameter to perform the necessary operations.

    Parameters
    ----------
    cont : Any
        The container to be closed or terminated. Its type should be compatible with
        the operation defined by this function.
    d : Any
        Additional data or parameter required to perform the close or termination
        operation on the container. The exact purpose and use of this parameter
        depend on the function's implementation.

    Returns
    -------
    None
        This function does not return any value.
    """
    cont_ = np.vstack([cont, cont[0]]).T
    Ns = np.sqrt(np.sum(np.diff(cont_, axis=1) ** 2, axis=0)).astype(np.int)
    Ns = Ns / d

    k = 0
    xys = fill_npoints(cont_[:, k:k + 2], Ns[k]).T

    for k in range(1, len(cont_[0]) - 1):
        xys = np.vstack((xys, fill_npoints(cont_[:, k:k + 2], Ns[k]).T))
    return xys


def get_intrinsic(alpha, beta, tau, k):
    """Calculates the intrinsic value based on given inputs.

    This function computes the intrinsic value using the provided parameters
    `alpha`, `beta`, `tau`, and `k`. The formula or logic for the computation
    is implicitly defined within the function implementation. Ensure that the
    supplied arguments are valid and conform to any constraints specified
    in the function's behavior.

    Parameters
    ----------
    alpha : float
        A coefficient or input parameter that influences the result
        of the intrinsic value calculation.
    beta : float
        A secondary coefficient or parameter affecting the intrinsic
        value computation. It may relate to `alpha` or modify the outcome.
    tau : float
        Represents a time-related or scaling factor in the calculation.
        The exact role depends on the detailed implementation.
    k : float
        A constant or variable parameter involved in the computation
        to adjust or scale the intrinsic value.

    Returns
    -------
    float
        The computed intrinsic value resulting from the combination of
        the parameters `alpha`, `beta`, `tau`, and `k`.
    """
    Ac = np.hstack([[beta, -alpha - 4 * beta, 2 * alpha + 6 * beta, -alpha - 4 * beta, beta], np.zeros(k - 5)])

    A = np.zeros([k, k])
    for i in range(k):
        A[i] = np.roll(Ac, i - 2)

    mat = np.linalg.inv(np.eye(k) + tau * A)
    return mat


def refine_anim(f, svgdir, beta=.0001, alpha=.01, tau=10, Nit=10000, ksave=100):
    """Performs refinement optimization of an animation by iteratively minimizing a given cost function.

    This function refines an animation according to a provided cost function
    using iterative optimization techniques. It adjusts parameters to minimize
    the cost function and save intermediate results for analysis.

    Parameters
    ----------
    f : Callable
        The cost function to be minimized. It should accept an input and return
        a scalar value representing the cost.
    svgdir : str
        Directory path where intermediate results and final outcomes are saved.
        Must be a valid and accessible path.
    beta : float, optional
        Learning rate for the optimization. Controls the step size in the
        optimization process. Default is 0.0001.
    alpha : float, optional
        Regularization parameter for controlling model complexity. Default is 0.01.
    tau : int, optional
        Parameter controlling the smoothing or damping of the optimization
        process. Default is 10.
    Nit : int, optional
        Maximum number of iterations for the optimization process. Default
        is 10000.
    ksave : int, optional
        Interval for saving intermediate optimization results. The results
        are saved every `ksave` iterations. Default is 100.

    Returns
    -------
    None
    """
    im = cv2.imread(f + ".png").astype(np.float)
    h = im.shape[0]
    w = im.shape[1]
    cidx = exgreen(im)
    M = cidx.max()
    m = cidx.min()
    imG = (255 * (cidx - m) / (M - m)).astype(np.uint8)

    sig = 3
    F_norm_xy = get_edge_grad(imG, sig)

    cont = np.loadtxt(f + ".txt")
    xys = close_cont(cont, d)
    intr = get_intrinsic(alpha, beta, len(xys))
    xs = xys[:, 0].clip(0, w).astype(np.int)
    ys = xys[:, 1].clip(0, h).astype(np.int)
    cont_hist = [[xs, ys]]

    plt.imshow(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.plot(xs, ys, "b")

    for i in range(Nit):
        F = np.array([F_norm_xy[1, ys, xs], F_norm_xy[0, ys, xs]]).T
        xys = np.dot(intr, xys + tau * F)
        xs = xys[:, 0].clip(0, w).astype(np.int)
        ys = xys[:, 1].clip(0, h).astype(np.int)
        if not (i % ksave):
            print(i)
            cont_hist.append([xs, ys])
    np.save(svgdir + '/' + f.strip("data/annotations/") + "_cont", cont_hist)

    plt.plot(xs, ys, "r")
    plt.savefig(svgdir + '/' + f.strip("data/annotations/") + "_cont.jpg")
    plt.clf()


def refine(imname, xys=[], beta=.0001, alpha=.01, tau=10, d=1, Nit=10000, ksave=100):
    """Refines the optimization of a given dataset according to the provided parameters.

    This function operates on an image dataset. It iteratively refines the data using
    optimization parameters such as beta, alpha, and tau. The refinement process is
    controlled by the number of iterations (`Nit`) and the checkpoint interval (`ksave`).
    It also supports optional input of spatial parameters (`xys`) for initial setup.

    Parameters
    ----------
    imname : str
        The name or path of the image file to be refined.
    xys : list, optional
        List of spatial coordinates or initial setup points. Default is an empty list.
    beta : float, optional
        Optimization parameter controlling regularization. Default is 0.0001.
    alpha : float, optional
        Optimization parameter representing step size or learning rate. Default is 0.01.
    tau : int or float, optional
        Time constant or scale factor controlling dynamics. Default is 10.
    d : int, optional
        Dimensionality or degree of freedom for the problem. Default is 1.
    Nit : int, optional
        Number of optimization iterations to perform. Default is 10000.
    ksave : int, optional
        Frequency of saving intermediate results during iterations. Default is 100.

    Returns
    -------
    None
        This function does not return a value. The results are stored or managed
        internally, such as writing to a file or updating a dataset.

    """
    im = cv2.imread(imname).astype(np.float)
    h, w = im.shape[:2]
    imG = exgreen(im, True)

    sig = 3
    F_norm_xy = get_edge_grad(imG, sig)

    intr = get_intrinsic(alpha, beta, tau, len(xys))

    for i in range(Nit):
        xs = xys[:, 0].clip(0, w - 1).astype(np.int)
        ys = xys[:, 1].clip(0, h - 1).astype(np.int)
        F = np.array([F_norm_xy[1, ys, xs], F_norm_xy[0, ys, xs]]).T
        xys = np.dot(intr, xys + tau * F)

    return xs, ys


def run_refine(f, beta, alpha, tau, d, Nit, plotit=None, saveit=None):
    """Refines contour points for annotated regions in an image using active contour models.

    This function reads polygonal shapes from a given JSON file corresponding to an image, processes
    their contours, and refines them iteratively. It optionally visualizes and saves the results.

    Parameters
    ----------
    f : str
        File path to the image file. A corresponding JSON file with the same base name
        containing polygonal shapes and labels is required.
    beta : float
        Elasticity parameter for the active contour model. Represents the smoothness of the contour.
    alpha : float
        Rigidity parameter for the active contour model. Controls the bending energy.
    tau : float
        Step size for the contour update process.
    d : int
        Number of discrete grid points around the contour for the active contour evaluation.
    Nit : int
        Number of iterations for contour refinement.
    plotit : str or None, optional
        File path to save the image with visualized contours. If None, no visualization is saved.
    saveit : str or None, optional
        File path to save the refined contour points as a NumPy array. If None, contour points
        are not saved.

    Returns
    -------
    list of numpy.ndarray
        A list of refined contour points for each shape in the JSON file. Each element is a 2D
        NumPy array containing the coordinates of the contour points.
    """
    polys = json.load(open(f[:-4] + '.json'))['shapes']
    ps = [np.array(p['points']) for p in polys]
    labels = [p['label'] for p in polys]
    label_color = {'background': 0, 'flower': 51, 'peduncle': 102, 'stem': 153, 'leaf': 204, 'fruit': 255}

    im = cv2.imread(f)
    im = im * 0
    #  if plotit: cv2.polylines(im, ps, True, (242,240,218), thickness=10)
    conts = []
    for i, p in enumerate(ps):
        init_cont = close_cont(p, 1)
        color = label_color[labels[i]]
        print(color)
        xys = refine(f, init_cont, beta, alpha, tau, d, Nit, ksave=1)
        if plotit: cv2.fillPoly(im, [np.array([xys]).astype(np.int).T], color)
        conts.append(xys)
    if True:
        cv2.imwrite(plotit, im)

    if saveit: np.save(saveit, conts)
    return conts


def run_refine_romidata(f, beta, alpha, tau, d, Nit, class_names, plotit=None, saveit=None):
    """Refine and process region of interest (ROI) contours given an input multi-class segmentation dataset.

    This function reads a segmentation image and polygon annotations for multiple
    classes, refines these polygons, and generates binary masks for each class.
    Optionally, it can also visualize or save results during refinement.

    Parameters
    ----------
    f : str
        Path to the input image file. The corresponding annotation file is expected
        to have the same name but with a '.json' extension.
    beta : float
        Stiffness parameter for contour refinement.
    alpha : float
        Attraction force parameter for contour refinement.
    tau : float
        Balloon force parameter for contour refinement.
    d : float
        Step size for refinement iterations.
    Nit : int
        Number of iterations for contour refinement.
    class_names : List[str]
        List of unique class names for the segmentation, including "background".
    plotit : Optional[bool]
        If True, performs visualization during the refinement process by updating
        the binary masks per class.
    saveit : Optional[bool]
        If True, saves the intermediate results of refinement for debugging or
        visualization purposes.

    Returns
    -------
    Dict[str, numpy.ndarray]
        A dictionary where keys are class names and values are binary masks
        (numpy arrays) for the corresponding classes.
    """
    polys = json.load(open(f[:-4] + '.json'))['shapes']
    ps = [np.array(p['points']) for p in polys]
    labels = [p['label'] for p in polys]

    im = cv2.imread(f)
    im = im * 0
    npz = {class_name: im[:, :, 0] * 0 for class_name in class_names}
    back = npz['background']
    #  if plotit: cv2.polylines(im, ps, True, (242,240,218), thickness=10)
    # conts=[]
    for i, p in enumerate(ps):
        init_cont = close_cont(p, 1)
        print(labels[i])
        xys = refine(f, init_cont, beta, alpha, tau, d, Nit, ksave=1)
        if plotit: cv2.fillPoly(npz[labels[i]], [np.array([xys]).astype(np.int).T], 255)

    for k in npz.keys():
        back += npz[k]
    back = (back == 0) * 255
    npz['background'] = back
    return npz


#   conts.append(xys)

# if True:
#    cv2.imwrite(plotit,im)

# if saveit: np.save(saveit,conts)
# return conts


# imdir = "/home/alienor/Documents/database/FINETUNE/images"
# files = glob.glob(imdir + '/*.jpg')
# for f in files:
#    npz = run_refine(f, 1,1,1,1,1,class_names = 'background,flower,peduncle,stem,bud,leaf,fruit'.split(','), plotit=True)

# %%
if False:
    f = fnames[0]
    for beta in range(-6, 6):
        for alpha in range(-6, 6):
            for tau in range(-6, 6):
                for d in range(-6, 6):
                    print(f)
                    run_refine(f, 10 ** (beta), 10 ** (alpha),
                               10 ** (tau), 10 ** (d), 1000,
                               plotit=f[:-4] + 'contours%d_%d_%d_%d.png' % (beta, alpha, tau, d))
# %%
if False:
    jsfiles = np.sort(glob.glob(imdir + "*.json"))
    for f in fnames:
        if f[:-4] + '.json' in jsfiles:
            run_refine(f, 1, 1, 1, 1, 1,
                       plotit=f[:-4] + "_contours.png")
