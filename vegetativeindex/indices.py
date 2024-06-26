import numpy as np

vegetative_indices = {
    "Indice de cor de vegetação (CIVE)": 3,
    "Excessive green (ExG)": 0,
    "Excessive red (ExR)": 1,
    "Normalized Difference Index (NDI)": 2,
    "Excessive rde minus excessive red (ExGR)": 4,
    "Vegetative index (VEG)": 5, 
    "Combined indices (COM)": 6,
    "Custom": 7
}

def get_exg(image):
    R, G, B = _split_rgb_channels(image)
    exg_image = (2*G - R - G)*-1
    return _normalize_image(exg_image)

def get_exr(image):
    R, G, _ = _split_rgb_channels(image)
    exr_image = 1.4*R - G
    return _normalize_image(exr_image)

def get_ndi(image):
    R, G, _ = _split_rgb_channels(image)
    ndi_image = (128*(((G - R) / (G + R)) + 1)) *-1
    return _normalize_image(ndi_image)

def get_cive(image):
    R, G, B = _split_rgb_channels(image)
    cive_img = (0.441 * R) - (0.811 * G) + (0.385 * B)
    return _normalize_image(cive_img)

def get_exgr(image):
    exgr_img = get_exr(image) - get_exg(image)
    return _normalize_image(exgr_img)

def get_veg(image):
    R, G, B = _split_rgb_channels(image)
    a = 0.667
    veg_img = G/(R**(a) * B**(1-a))
    return _normalize_image(veg_img)

def get_com(image):
    exg = 0.25 * get_exg(image)
    exgr = 0.3 * get_exgr(image)
    cive = 0.33 * get_cive(image)
    veg = 0.12 * get_veg(image)

    com_img = exg + exgr + cive + veg
    return _normalize_image(com_img)

def get_custom(image, _r, _g, _b):
    R, G, B = _split_rgb_channels(image)
    custom_img = (_r * R) - (_g * G) + (_b * B)
    return _normalize_image(custom_img)

def _normalize_image(image):
    return ((image + np.abs(image.min())) / (np.abs(image.min()) + image.max()) * 255).astype(np.uint8)

def _split_rgb_channels(image):
    return image[:,:,0]/np.max(image[:,:,0]), image[:,:,1]/np.max(image[:,:,1]), image[:,:,2]/np.max(image[:,:,2])
