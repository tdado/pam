from classes import CustomRidgeCV
from decoder import pearson_correlation_coefficient
from imports import *


def get_god():
    w_te = np.load("/home/tdado/god/t_te_best.npy")
    w_tr = np.load("/home/tdado/god/t_tr_best.npy")
    rois = ["V1", "V2", "V3", "V4", "LOC", "FFA", "PPA"]
    _H = [0, 2521, 2687, 2058, 1283, 2724, 2100, 900]
    H = np.cumsum(_H)

    # voxel selection
    x_tr = np.zeros((1200, H[-1]))
    for i, roi in enumerate(rois):
        x_tr = np.zeros((1200, H[-1]))
        for i, roi in enumerate(rois):
            _roi = np.mean(np.load(f"/home/tdado/god2/hyperaligned/{roi}_aligned.npy")[:, :1200], axis=0)
            roi_mean = np.mean(_roi, axis=0)
            roi_std = np.std(_roi, axis=0, ddof=1)
            roi_std[roi_std == 0] = 1
            x_tr[:, H[i]:H[i+1]] = (_roi - roi_mean) / roi_std

    y_cv = np.zeros((1200, H[-1]))
    t_cv = np.zeros((1200, H[-1]))
    for i, roi in enumerate(rois[:6]):
        x_roi = x_tr[:, H[i]:H[i+1]]
        kf = KFold(n_splits=10, shuffle=True, random_state=7)  # Adjusted to 5 folds
        for train_index, test_index in kf.split(w_tr):
            w_train, w_val = w_tr[train_index], w_tr[test_index]
            x_train, x_val = x_roi[train_index], x_roi[test_index]
            model = CustomRidgeCV(w_train, x_train, n_alphas=10).train()
            y_cv[test_index, H[i]:H[i+1]] = model.predict(w_val)
            t_cv[test_index, H[i]:H[i+1]] = x_val
    r, p = pearson_correlation_coefficient(y_cv, t_cv, 0) # 10-fold now

    # False Discovery Rate
    alpha_FDR = 0.05
    _h = [0]
    masks = []
    for roi_index, roi in enumerate(rois):
        p_region = p[H[roi_index]:H[roi_index+1]]
        sorted_i = np.argsort(p_region)
        sorted_p = np.sort(p_region)
        thr = (np.arange(1, len(p_region)+1) / len(p_region)) * alpha_FDR
        mask = np.zeros(len(p_region))

        # sequence of thresholds where each element in the array corresponds to 
        # the maximum allowable p-value for the i-th smallest observed p-value to 
        # be considered significant (a = 0.05)
        try:
            crit_i = np.where(sorted_p <= thr)[0].max()
            crit_p = sorted_p[crit_i]
            signif = sorted_i[sorted_p <= crit_p]
            mask[signif] = True
        except:
            pass
        _h += [int(mask.sum())]
        masks += [mask.astype("bool")]
    h = np.cumsum(_h)


    # separate and take mean
    x_tr = np.zeros((1200, np.sum(_h)))
    x_pt = np.zeros((50, np.sum(_h)))
    for roi_index, roi in enumerate(rois):
        _roi = np.load(f"/home/tdado/god2/hyperaligned/{roi}_aligned.npy")
        vox_range = slice(h[roi_index], h[roi_index+1])
        x_tr[:, vox_range] = _roi[:, :1200, masks[roi_index]].mean(axis=0)
        roi_mean = np.mean(x_tr[:, vox_range], axis=0)
        roi_std = np.std(x_tr[:, vox_range], axis=0, ddof=1) + 1e-8
        x_tr[:, vox_range] = (x_tr[:, vox_range] - roi_mean) / roi_std
        _x_pt = _roi[:, 1200:1250, masks[roi_index]].mean(axis=0)
        x_pt[:, vox_range] = (_x_pt - roi_mean) / roi_std
    return w_te, x_pt, w_tr, x_tr, h


def get_styxl():
    f1 = h5py.File("/home/tdado/images/GANs_StyleGAN_XL_normMUA.mat", "r")
    x_te = np.delete(np.array(f1["test_MUA"]), np.arange(320, 384), axis=1)
    x_tr = np.delete(np.array(f1["train_MUA"]), np.arange(320, 384), axis=1)
    w_te = np.load("/home/tdado/images/ws_te7.npy")[:, 0]
    w_tr = np.load("/home/tdado/images/ws_tr7.npy")[:, 0]
    h = np.cumsum([0, 448, 256, 256])
    return w_te, x_te, w_tr, x_tr, h