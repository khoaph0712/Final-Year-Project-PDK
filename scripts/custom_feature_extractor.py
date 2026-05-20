"""Redesigned custom feature extractor for Final Year Project (FYP) waste classification.

Extracts EXACTLY 637 features:
1) Color (256 features):
   - RGB Histogram: 144 bins (3 channels * 48 bins)
   - HSV Histogram: 112 bins (H: 48 bins, S: 32 bins, V: 32 bins)
   Total Color = 144 + 112 = 256 features.

2) Texture (47 features):
   - Local Binary Patterns (LBP): 10 uniform LBP bins (10 features)
   - Gray-Level Co-occurrence Matrix (GLCM): 37 features
     * 9 statistical descriptors computed across 4 directions (0, 45, 90, 135 degrees) = 36 features
     * 1 global average entropy descriptor = 1 feature
   Total Texture = 10 + 37 = 47 features.

3) Shape/Geometric (10 features):
   - Hu Moments: 7 scale & rotation-invariant log-transformed moments (7 features)
   - Geometric measurements: Area, Perimeter, and Circularity (3 features)
   Total Shape = 10 features.

4) Edge/HOG (324 features):
   - Histogram of Oriented Gradients (HOG) with custom parameters:
     * Window Size: 64x64
     * Block Size: 32x32
     * Block Stride: 16x16
     * Cell Size: 16x16
     * Bins: 9
     Calculates to: 9 blocks * 4 cells/block * 9 bins = 324 features.

Total features = 256 + 47 + 10 + 324 = 637 features exactly.
"""

from __future__ import annotations
import cv2
import numpy as np


def compute_color_features(crop_bgr: np.ndarray) -> np.ndarray:
    """Compute normalized RGB and HSV histograms yielding exactly 256 features.
    
    RGB: 144 bins (3 channels * 48 bins per channel)
    HSV: 112 bins (Hue: 48 bins, Sat: 32 bins, Val: 32 bins)
    """
    resized_bgr = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    
    # 1. RGB Histogram (144 features)
    rgb_hists = []
    for chan in range(3):
        h = cv2.calcHist([resized_bgr], [chan], None, [48], [0, 256]).flatten()
        h = h / (h.sum() + 1e-9)
        rgb_hists.append(h)
    rgb_hist = np.concatenate(rgb_hists)
    
    # 2. HSV Histogram (112 features)
    resized_hsv = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
    
    h_hist = cv2.calcHist([resized_hsv], [0], None, [48], [0, 180]).flatten()
    s_hist = cv2.calcHist([resized_hsv], [1], None, [32], [0, 256]).flatten()
    v_hist = cv2.calcHist([resized_hsv], [2], None, [32], [0, 256]).flatten()
    
    h_hist = h_hist / (h_hist.sum() + 1e-9)
    s_hist = s_hist / (s_hist.sum() + 1e-9)
    v_hist = v_hist / (v_hist.sum() + 1e-9)
    
    hsv_hist = np.concatenate([h_hist, s_hist, v_hist])
    
    return np.concatenate([rgb_hist, hsv_hist])


def compute_lbp(gray_image: np.ndarray) -> np.ndarray:
    """Compute uniform Local Binary Patterns (LBP) for a 64x64 grayscale image.
    
    Uses circular P=8, R=1 neighborhood and maps 256 patterns to 10 uniform bins.
    """
    h, w = gray_image.shape
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
    
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1), (1, 1), (1, 0),
        (1, -1), (0, -1)
    ]
    
    for bit, (dy, dx) in enumerate(neighbors):
        shifted = gray_image[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx]
        center = gray_image[1 : h - 1, 1 : w - 1]
        mask = shifted >= center
        lbp += (mask.astype(np.uint8) << bit)
    
    lut = np.zeros(256, dtype=np.uint8)
    uniform_vals = []
    for val in range(256):
        binary = f"{val:08b}"
        transitions = 0
        for j in range(8):
            if binary[j] != binary[(j + 1) % 8]:
                transitions += 1
        if transitions <= 2:
            uniform_vals.append(val)
            
    for label, val in enumerate(uniform_vals):
        lut[val] = label
    for val in range(256):
        if val not in uniform_vals:
            lut[val] = 9
            
    lbp_mapped = lut[lbp]
    hist, _ = np.histogram(lbp_mapped, bins=10, range=(0, 10))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-9)
    return hist


def compute_glcm_features(gray_image: np.ndarray, Ng: int = 8) -> np.ndarray:
    """Compute symmetric GLCM features for a 64x64 gray-scale image.
    
    Quantizes gray levels to 8 levels (0..7). Computes GLCM for 4 directions
    (0, 45, 90, 135) at distance 1 and returns exactly 37 descriptors.
    """
    h, w = gray_image.shape
    gray_q = (gray_image / (256 / Ng)).astype(np.uint8)
    gray_q = np.clip(gray_q, 0, Ng - 1)
    
    directions = [
        (0, 1),   # 0 degrees (horizontal)
        (1, 1),   # 45 degrees (diagonal)
        (1, 0),   # 90 degrees (vertical)
        (1, -1)   # 135 degrees (diagonal)
    ]
    
    glcms = []
    for dy, dx in directions:
        glcm = np.zeros((Ng, Ng), dtype=np.float32)
        
        y_start, y_end = max(0, dy), min(h, h + dy)
        x_start, x_end = max(0, dx), min(w, w + dx)
        
        y_shift_start, y_shift_end = max(0, -dy), min(h, h - dy)
        x_shift_start, x_shift_end = max(0, -dx), min(w, w - dx)
        
        ref = gray_q[y_start:y_end, x_start:x_end]
        neighbor = gray_q[y_shift_start:y_shift_end, x_shift_start:x_shift_end]
        
        for r in range(Ng):
            mask_r = (ref == r)
            for c in range(Ng):
                glcm[r, c] = np.sum(mask_r & (neighbor == c))
                
        glcm_sym = glcm + glcm.T
        s = glcm_sym.sum()
        if s > 0:
            glcm_sym /= s
        glcms.append(glcm_sym)
        
    glcm_feats = []
    i_idx, j_idx = np.ogrid[:Ng, :Ng]
    entropies = []

    for idx, P in enumerate(glcms):
        contrast = np.sum(((i_idx - j_idx) ** 2) * P)
        homogeneity = np.sum(P / (1.0 + (i_idx - j_idx) ** 2))
        energy = np.sum(P ** 2)
        dissimilarity = np.sum(np.abs(i_idx - j_idx) * P)
        max_prob = np.max(P)
        entropy = -np.sum(P * np.log(P + 1e-9))
        entropies.append(entropy)
        
        mu_i = np.sum(i_idx * P)
        var_i = np.sum(((i_idx - mu_i) ** 2) * P)
        std_i = np.sqrt(var_i)
        
        mu_j = np.sum(j_idx * P)
        var_j = np.sum(((j_idx - mu_j) ** 2) * P)
        std_j = np.sqrt(var_j)
        
        if std_i * std_j > 1e-9:
            corr = np.sum((i_idx - mu_i) * (j_idx - mu_j) * P) / (std_i * std_j)
        else:
            corr = 0.0
            
        glcm_feats.extend([
            float(contrast),
            float(homogeneity),
            float(energy),
            float(dissimilarity),
            float(max_prob),
            float(entropy),
            float(mu_i),
            float(std_i),
            float(corr)
        ])
        
    avg_entropy = float(np.mean(entropies))
    glcm_feats.append(avg_entropy)
    
    return np.array(glcm_feats, dtype=np.float32)


def compute_shape_features(gray_image: np.ndarray) -> np.ndarray:
    """Compute exactly 10 shape/geometric features for a 64x64 grayscale crop.
    
    Features: 7 Hu Moments + 1 Area + 1 Perimeter + 1 Circularity.
    """
    moments = cv2.moments(gray_image)
    hu = cv2.HuMoments(moments).flatten()
    
    hu_log = np.zeros(7, dtype=np.float32)
    for i in range(7):
        val = hu[i]
        if abs(val) > 1e-20:
            hu_log[i] = -np.sign(val) * np.log10(np.abs(val))
            
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        area_raw = cv2.contourArea(c)
        perim_raw = cv2.arcLength(c, True)
        
        area = area_raw / (64.0 * 64.0)
        perimeter = perim_raw / (64.0 * 4.0)
        if perim_raw > 0:
            circularity = (4.0 * np.pi * area_raw) / (perim_raw ** 2 + 1e-9)
            circularity = min(1.0, circularity)
        else:
            circularity = 0.0
    else:
        area = 1.0
        perimeter = 1.0
        circularity = 0.0
        
    return np.concatenate([hu_log, np.array([area, perimeter, circularity], dtype=np.float32)])


def extract_637_features(crop_bgr: np.ndarray) -> np.ndarray:
    """Extract exactly 637 features from a BGR crop.
    
    Resizes internally to 64x64, then extracts:
    - 256 Color Features (RGB histogram 144 + HSV histogram 112)
    - 47 Texture Features (10 uniform LBP + 37 GLCM Descriptors)
    - 10 Shape/Geometric Features (7 Hu Moments + Area, Perimeter, Circularity)
    - 324 Edge Features (HOG descriptor)
    """
    resized_bgr = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    
    color_features = compute_color_features(resized_bgr)
    
    lbp_hist = compute_lbp(gray) # 10
    glcm_feats = compute_glcm_features(gray) # 37
    texture_features = np.concatenate([lbp_hist, glcm_feats]) # 47
    
    shape_features = compute_shape_features(gray) # 10
    
    hog = cv2.HOGDescriptor(
        _winSize=(64, 64),
        _blockSize=(32, 32),
        _blockStride=(16, 16),
        _cellSize=(16, 16),
        _nbins=9,
    )
    hog_features = hog.compute(gray).flatten().astype(np.float32) # 324
    
    feature_vector = np.concatenate([
        color_features,      # 256
        texture_features,    # 47
        shape_features,      # 10
        hog_features         # 324
    ])
    
    return feature_vector


if __name__ == "__main__":
    print("Running custom feature extractor self-test...")
    test_img = np.random.randint(0, 256, (120, 80, 3), dtype=np.uint8)
    feats = extract_637_features(test_img)
    print(f"Input image shape: {test_img.shape}")
    print(f"Extracted feature shape: {feats.shape}")
    print(f"Feature vector length matches 637 exactly: {len(feats) == 637}")
    
    print(f"  - Color features (0..255): length {len(feats[0:256])}")
    print(f"  - Texture features (256..302): length {len(feats[256:303])}")
    print(f"  - Shape features (303..312): length {len(feats[303:313])}")
    print(f"  - HOG features (313..636): length {len(feats[313:637])}")
