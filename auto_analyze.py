'''
Particle Analysis and Measurement System

This script processes coffee grounds images to detect ground coffee particles and analyze their size and shape.
Key functionalities include:
1) Image loading and visualization
2) Particle detection through thresholding
3) Cluster analysis for particle identification
4) Size and distribution measurements
5) Statistical visualization of results

Dependencies:
- numpy: numerical operations
- PIL: image processing
- matplotlib: visualization
- scipy: computational geometry
'''

# import required libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import ConvexHull
import cv2
import time
import sys
import os

# Add toolbar for zoom and pan
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import tkinter as tk

# Redirect stderr to devnull to suppress Tkinter errors
class SuppressTkinterErrors:
    def __enter__(self):
        self.stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, *args):
        sys.stderr = self.stderr

def plot_images(original_images, processed_images, original_titles, processed_titles, figsize=(15, 10)):
    '''
    Displays original and preprocessed images in a single figure with two rows.
    
    Args:
        original_images (list): List of original images to display
        processed_images (list): List of preprocessed images to display
        original_titles (list): Titles for original images
        processed_titles (list): Titles for preprocessed images
        figsize (tuple): Figure dimensions (width, height)
    '''
    # create new figure
    plt.figure(figsize=figsize)
    
    # plot original images on top row
    for i, (img, title) in enumerate(zip(original_images, original_titles)):
        plt.subplot(2, len(original_images), i+1)
        plt.imshow(img)
        plt.title(title)
        #plt.axis('off')
    
    # plot preprocessed images on bottom row
    for i, (img, title) in enumerate(zip(processed_images, processed_titles)):
        plt.subplot(2, len(processed_images), i+1+len(original_images))
        plt.imshow(img)
        plt.title(title)
        #plt.axis('off')
    
    # adjust layout and display
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def preprocess_image(img):
    '''
    Preprocesses the image to reduce noise and enhance contrast.
    
    Args:
        img (PIL.Image): Input image
    Returns:
        PIL.Image: Processed image
    '''
    # Convert PIL image to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Apply bilateral filter to reduce noise while preserving edges
    # Parameters: d=9 (diameter of pixel neighborhood), sigmaColor=75, sigmaSpace=75
    denoised = cv2.bilateralFilter(img_cv, 9, 75, 75)
    
    # Convert to LAB color space for better contrast enhancement
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE with more conservative parameters
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    lab_clahe = cv2.merge((l_clahe, a, b))
    res = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    # Optional: Apply slight Gaussian blur to reduce any remaining noise
    # Use a small kernel (3x3) to avoid losing too much detail
    res = cv2.GaussianBlur(res, (3, 3), 0)
    
    # Convert back to PIL format
    enhanced_pil = Image.fromarray(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    return enhanced_pil  

class ProcessingResults:
    '''
    Container class for storing analysis results.
    
    Attributes:
        mask_threshold: Binary mask from thresholding
        cluster_data: List of detected particle clusters
        nclusters: Number of clusters found
    '''
    def __init__(self):
        self.mask_threshold = None
        self.cluster_data = []
        self.nclusters = 0


def threshold_image(img, analysis_region, threshold_percent=58.8):
    '''
    Applies threshold to identify particles in an image within specified region.
    
    Args:
        img (PIL.Image): Input image to process
        analysis_region (tuple): (x1, y1, x2, y2) coordinates defining region
        threshold_percent (float): Percentage of median for threshold
        
    Returns:
        tuple: (cropped image, mask coordinates, image data, background median)
    '''
    # extract region coordinates
    x1, y1, x2, y2 = analysis_region
    
    # crop image to region of interest
    cropped = img.crop((x1, y1, x2, y2))
    
    # convert to numpy array and extract blue channel
    imdata_3d = np.array(cropped)
    imdata = imdata_3d[:, :, 2]
    
    # calculate background and threshold
    bg_median = np.median(imdata)
    thresh_val = bg_median * threshold_percent / 100
    
    # create binary mask
    mask = np.where(imdata < thresh_val)
    
    return cropped, mask, imdata, bg_median

def quick_cluster(xlist, ylist, xstart, ystart):
    # initialize arrays for checking points
    xcheck = np.array([xstart])
    ycheck = np.array([ystart])
    xlist_decay = np.copy(xlist)
    ylist_decay = np.copy(ylist)
    ilist_decay = np.arange(xlist.size)
    
    # find and remove starting point from decay lists
    istart = np.where((xlist_decay == xstart) & (ylist_decay == ystart))
    if istart[0].size != 0:
        xlist_decay = np.delete(xlist_decay, istart[0])
        ylist_decay = np.delete(ylist_decay, istart[0])
        ilist_decay = np.delete(ilist_decay, istart[0])
    iout = istart[0]
    
    # process points until no more neighbors found
    for _ in range(xlist.size):
        # find points within manhattan distance of 1
        isel = np.where((np.abs(xlist_decay - xcheck[0]) + np.abs(ylist_decay - ycheck[0])) <= 1.001)
        if isel[0].size == 0:
            if xcheck.size == 1:
                break
            xcheck = np.delete(xcheck, 0)
            ycheck = np.delete(ycheck, 0)
            continue
            
        # add found points to output and check lists
        iout = np.append(iout, ilist_decay[isel[0]])
        xcheck = np.append(xcheck, xlist_decay[isel[0]])
        ycheck = np.append(ycheck, ylist_decay[isel[0]])
        xcheck = np.delete(xcheck, 0)
        ycheck = np.delete(ycheck, 0)
        
        # check if we've found all points
        if isel[0].size == xlist_decay.size:
            break
            
        # remove processed points from decay lists
        xlist_decay = np.delete(xlist_decay, isel[0])
        ylist_decay = np.delete(ylist_decay, isel[0])
        ilist_decay = np.delete(ilist_decay, isel[0])
    return iout

def launch_psd(mask, imdata, bg_median,
               max_cluster_axis=100, min_surface=5,
               reference_threshold=0.4, maxcost=0.35, nsmooth=3):
    # extract coordinates from mask
    X = mask[0].astype(int); Y = mask[1].astype(int)
    n = X.size

    # Precompute imdata_mask to avoid repeated indexing
    imdata_mask = imdata[mask]

    # initialize tracking arrays
    counted = np.zeros(n, bool)
    clusters = []
    cluster_count = 0  # add counter for progress tracking

    # for each point
    for _ in range(n):
        # find next unprocessed point
        open_idx = np.where(~counted)[0]
        if open_idx.size == 0: break

        # get current point and calculate distances
        curr = open_idx[0]
        d2 = (X[curr] - X[open_idx])**2 + (Y[curr] - Y[open_idx])**2
        close = open_idx[d2 <= max_cluster_axis**2]

        # skip if too few points
        if close.size < min_surface:
            counted[curr] = True; continue
        
        # perform quick clustering
        qc = quick_cluster(X[close], Y[close], X[curr], Y[curr])
        iclust = close[qc]

        # validate cluster size
        if iclust.size < min_surface:
            counted[curr] = True; continue
        
        # calculate intensity-based costs using precomputed imdata_mask
        cost = np.maximum((imdata_mask[iclust] - imdata_mask[curr])**2 / bg_median**2, 0)
        filt = np.array([np.where(iclust == curr)[0][0]], int)
        maxpath = np.full(iclust.size, np.nan)

        # analyze paths through cluster
        for ci in range(iclust.size):
            if iclust[ci] == curr: continue
            # calculate threshold and find dark points
            vals = imdata_mask[iclust[filt]]
            thr = (bg_median - imdata_mask[curr]) * reference_threshold + imdata_mask[curr]
            idark = np.where(vals <= thr)[0]

            # select best reference point
            if idark.size == 0: continue

            # select best reference point
            if idark.size > 1:
                cand = filt[idark]
                d2c = (X[iclust[cand]] - X[iclust[ci]])**2 + (Y[iclust[cand]] - Y[iclust[ci]])**2
                best = np.argmin(d2c)
                idark = idark[best]

            # process path geometry
            idark = int(idark)
            tgt = iclust[filt[idark]]
            x1, y1 = X[iclust[ci]], Y[iclust[ci]]
            x2, y2 = X[tgt],        Y[tgt]
            x0, y0 = X[iclust],      Y[iclust]

            # calculate distances and validate path
            dd = np.abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.hypot(y2-y1, x2-x1)
            d1 = np.hypot(x0-x1, y0-y1)
            d2_ = np.hypot(x0-x2, y0-y2)
            d12 = np.hypot(x2-x1, y2-y1)
            path = np.where((dd <= np.sqrt(2)) & (d1 <= d12) & (d2_ <= d12))[0]

            # process valid paths
            if path.size:
                cp = cost[path]
                if nsmooth < cp.size:
                    sm = np.convolve(cp, np.ones(nsmooth)/nsmooth, mode='same')*nsmooth
                else:
                    sm = np.full(cp.size, cp.sum())
                maxpath[ci] = sm.max()

            # add point if path cost is acceptable
            if maxpath[ci] < maxcost:
                filt = np.append(filt, ci)

        # mark processed points
        counted[iclust] = True

        # skip small clusters
        if filt.size < min_surface: continue

        # calculate cluster properties
        sel = iclust[filt]
        xs, ys = X[sel], Y[sel]
        xm, ym = xs.mean(), ys.mean()
        dlist = np.hypot(xs-xm, ys-ym)
        axis = dlist.max(); surf = filt.size

        # store cluster information
        clusters.append({
            'surface': surf,
            'long_axis': axis,
            'short_axis': surf/(np.pi*axis),
            'xmean': xm,
            'ymean': ym,
            'points': list(zip(xs, ys))
        })
        # increment counter and print progress every 25 clusters
        cluster_count += 1
        if cluster_count % 50 == 0:
            print(f"{cluster_count} clusters identified")

    # print final count
    if cluster_count > 0:
        print(f"Final count: {cluster_count} clusters identified")
    else:
        print("No clusters identified")

    return clusters


def plot_histograms(results, scales, names):
    '''
    Creates histograms of particle measurements with error bars and average points.
    
    Args:
        results: List of ProcessingResults objects
        scales: List of pixel-to-physical scales
        names: List of image names
    '''
    # process each result set
    for r, s, n in zip(results, scales, names):
        # calculate physical measurements
        diams = np.array([2*np.sqrt(c['long_axis']*c['short_axis'])/s for c in r.cluster_data])
        surf = np.array([c['surface']/(s**2) for c in r.cluster_data])
        
        # create figure
        plt.figure(figsize=(12, 5))
        
        # plot diameter distribution
        plt.subplot(1, 2, 1)
        # calculate histogram with weights
        weights = np.ones_like(diams)/len(diams)
        ypdf, xpdfleft, patches = plt.hist(diams, bins=20, weights=weights, 
                                         color=(147/255, 36/255, 30/255), 
                                         edgecolor='black', lw=2, rwidth=0.8,
                                         zorder=1)  # Set histogram bars to lower zorder
        
        # calculate error bars
        xpdf = xpdfleft[0:-1] + np.diff(xpdfleft)/2.0
        poisson_pos = np.sqrt(ypdf)/len(diams)
        poisson_neg = poisson_pos
        
        # plot error bars
        plt.errorbar(xpdf, ypdf, yerr=[poisson_neg, poisson_pos], 
                    marker=".", markersize=0, linestyle="", 
                    color=(147/510, 36/510, 30/510), elinewidth=2, 
                    capsize=0, alpha=0.8, zorder=3)  # Set error bars to higher zorder
        
        # calculate and plot average point
        avg = np.average(diams, weights=weights)
        ypos = np.max(ypdf)*0.05
        plt.errorbar(avg, ypos, xerr=[[np.std(diams)], [np.std(diams)]], 
                    marker="o", markersize=8, linestyle="", 
                    color=(147/255, 36/255, 30/255), elinewidth=2,
                    ecolor=(147/255, 36/255, 30/255, 0.3), 
                    markeredgewidth=1.5, markeredgecolor="k",
                    capsize=3, capthick=2, zorder=4)  # Set average point to highest zorder
        
        plt.title(f"{n}\nDiameter Distribution (N={len(diams)})")
        plt.xlabel("Diameter (mm)")
        plt.ylabel("Fraction of Particles")
        
        # plot surface area distribution
        plt.subplot(1, 2, 2)
        # calculate histogram with weights
        weights = np.ones_like(surf)/len(surf)
        ypdf, xpdfleft, patches = plt.hist(surf, bins=20, weights=weights,
                                         color=(147/255, 36/255, 30/255),
                                         edgecolor='black', lw=2, rwidth=0.8,
                                         zorder=1)  # Set histogram bars to lower zorder
        
        # calculate error bars
        xpdf = xpdfleft[0:-1] + np.diff(xpdfleft)/2.0
        poisson_pos = np.sqrt(ypdf)/len(surf)
        poisson_neg = poisson_pos
        
        # plot error bars
        plt.errorbar(xpdf, ypdf, yerr=[poisson_neg, poisson_pos],
                    marker=".", markersize=0, linestyle="",
                    color=(147/510, 36/510, 30/510), elinewidth=2,
                    capsize=0, alpha=0.8, zorder=15)  # Set error bars to higher zorder
        
        # calculate and plot average point
        avg = np.average(surf, weights=weights)
        ypos = np.max(ypdf)*0.05
        plt.errorbar(avg, ypos, xerr=[[np.std(surf)], [np.std(surf)]],
                    marker="o", markersize=8, linestyle="",
                    color=(147/255, 36/255, 30/255), elinewidth=2,
                    ecolor=(147/255, 36/255, 30/255, 0.3),
                    markeredgewidth=1.5, markeredgecolor="k",
                    capsize=3, capthick=2, zorder=16)  # Set average point to highest zorder
        
        plt.title(f"{n}\nSurface Area Distribution")
        plt.xlabel("Surface Area (mm²)")
        plt.ylabel("Fraction of Particles")
        
        # adjust and display plot
        plt.tight_layout()

def watershed_alg(img, mask=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    #Otsu's binarization for mask
    if mask is None:
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        thresh = mask

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_t = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_t, 0.5 * dist_t.max(),255,0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0

    color_img = img.copy()
    cv2.watershed(color_img, markers)
    particle_mask = np.where(markers > 1)
    markers[markers == -1] = 0

    return particle_mask, markers

def detect_particles_mser(img, analysis_region):
    '''
    Detects particles using MSER algorithm.
    
    Args:
        img (PIL.Image): Input image
        analysis_region (tuple): (x1, y1, x2, y2) coordinates defining region
        
    Returns:
        list: List of detected regions with their properties
    '''
    # Extract region coordinates
    x1, y1, x2, y2 = analysis_region
    
    # Crop image to region of interest
    cropped = img.crop((x1, y1, x2, y2))
    
    # Convert to numpy array and convert to grayscale
    img_array = np.array(cropped)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Invert the image since MSER detects dark regions on light background
    img_gray = 255 - img_gray
    
    # Apply Gaussian blur to reduce noise
    img_blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
    
    # Initialize MSER detector with more sensitive parameters
    mser = cv2.MSER_create(
        delta=2,            # Smaller delta for more sensitivity to intensity variations
        min_area=3,        # Smaller minimum area to catch smaller particles
        max_area=1000,      # Smaller maximum area to prevent lumping
        max_variation=0.12, # Lower variation threshold for more precise boundaries
        min_diversity=0.2  # Lower diversity to allow more regions
    )
    
    # Detect regions on blurred image
    regions, _ = mser.detectRegions(img_blurred)
    
    # Filter and process regions
    clusters = []
    processed_regions = []
    
    # Get image dimensions for edge checking
    height, width = img_blurred.shape
    
    # Sort regions by area (largest first)
    regions = sorted(regions, key=lambda x: cv2.contourArea(cv2.convexHull(x.reshape(-1, 1, 2))), reverse=True)
    
    for region in regions:
        # Get region properties
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        x, y, w, h = cv2.boundingRect(hull)
        
        # Skip regions that touch the image edges
        if x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
            continue
        
        # Create a mask for the region
        mask = np.zeros(img_blurred.shape, dtype=np.uint8)
        cv2.drawContours(mask, [hull], 0, 255, -1)
        
        # Erode the mask with 4x4 kernel
        kernel = np.ones((4,4), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        
        # Find contours of eroded region
        eroded_contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not eroded_contours:
            continue
            
        # Use the largest contour after erosion
        eroded_hull = cv2.convexHull(max(eroded_contours, key=cv2.contourArea))
        
        # Calculate region properties
        area = cv2.contourArea(eroded_hull)
        if area < 5:  # Smaller minimum area threshold
            continue
            
        # Calculate center and axes
        M = cv2.moments(eroded_hull)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w//2, y + h//2
            
        # Calculate major and minor axes
        if len(eroded_hull) >= 5:  # Need at least 5 points for ellipse fitting
            ellipse = cv2.fitEllipse(eroded_hull)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
        else:
            major_axis = max(w, h)
            minor_axis = min(w, h)
        
        # Check for overlap with existing regions
        is_overlapping = False
        for proc_region in processed_regions:
            # Calculate intersection over union (IoU)
            x1, y1, w1, h1 = cv2.boundingRect(proc_region)
            x2, y2, w2, h2 = cv2.boundingRect(eroded_hull)
            
            # Calculate intersection rectangle
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right < x_left or y_bottom < y_top:
                continue
                
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            box1_area = w1 * h1
            box2_area = w2 * h2
            union_area = box1_area + box2_area - intersection_area
            
            iou = intersection_area / union_area if union_area > 0 else 0
            
            # If significant overlap, skip this region
            if iou > 0.01:  # Keep strict overlap threshold
                is_overlapping = True
                break
        
        if not is_overlapping:
            # Store region information
            clusters.append({
                'surface': area,
                'long_axis': major_axis,
                'short_axis': minor_axis,
                'xmean': cx,
                'ymean': cy,
                'points': eroded_hull.reshape(-1, 2).tolist()
            })
            processed_regions.append(eroded_hull)
    
    return clusters

def detect_coin(image_pil, click_x, click_y):
    print('detecting coin...')
    img_array = np.array(image_pil)
    height, width = img_array.shape[:2]
    
    # Define the region of interest (ROI) around the click point
    roi_size = 220
    
    # Calculate ROI boundaries, ensuring they stay within image bounds
    x1 = max(0, int(click_x - roi_size))
    y1 = max(0, int(click_y - roi_size))
    x2 = min(width, int(click_x + roi_size))
    y2 = min(height, int(click_y + roi_size))
    
    # Extract ROI
    roi = img_array[y1:y2, x1:x2]
    
    # Process ROI
    img_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9,9), 2)
    edges = cv2.Canny(blurred, 30, 100)
    
    # Run Hough transform on ROI
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                              param1=100, param2=20, minRadius=30, maxRadius=200)
    
    if circles is not None:
        circles = np.round(circles[0,:]).astype("int")
        filtered_circles = []
        
        # Translate click coordinates to ROI coordinates
        roi_click_x = click_x - x1
        roi_click_y = click_y - y1
        
        for (x, y, r) in circles:
            # Check distance in ROI coordinates
            dist = np.hypot(x - roi_click_x, y - roi_click_y)
            if dist < 50:
                # Translate circle coordinates back to original image coordinates
                x_orig = x + x1
                y_orig = y + y1
                filtered_circles.append((x_orig, y_orig, r))
        
        if not filtered_circles:
            return None, None
            
        filtered_circles = sorted(filtered_circles, key=lambda c: c[2], reverse=True)
        x, y, r = filtered_circles[0]
        diameter_pixels = 2 * r
        return diameter_pixels, (x, y, r)
    
    return None, None

def cluster_overlap(c1, c2):
    """
    Check if two clusters overlap using an efficient two-step approach:
    1. Quick bounding box check
    2. Optimized point comparison if bounding boxes overlap
    """
    # Convert points to numpy arrays for faster computation
    points1 = np.array(c1['points'])
    points2 = np.array(c2['points'])
    
    # Get bounding boxes
    min_x1, min_y1 = points1.min(axis=0)
    max_x1, max_y1 = points1.max(axis=0)
    min_x2, min_y2 = points2.min(axis=0)
    max_x2, max_y2 = points2.max(axis=0)
    
    # Quick bounding box check
    if (max_x1 < min_x2 or min_x1 > max_x2 or
        max_y1 < min_y2 or min_y1 > max_y2):
        return False
    
    # If bounding boxes overlap, check points more carefully
    # Convert points to sets of tuples for faster lookup
    points1_set = set(map(tuple, points1))
    points2_set = set(map(tuple, points2))
    
    # Check for any common points
    return bool(points1_set & points2_set)


def plot_histograms_comparison(results_basic, results_enhanced, results_mser, scales, names):
    '''
    Creates comparison histograms of particle measurements from all three methods.
    
    Args:
        results_basic: List of ProcessingResults objects from basic original method
        results_enhanced: List of ProcessingResults objects from enhanced original method
        results_mser: List of ProcessingResults objects from MSER method
        scales: List of pixel-to-physical scales
        names: List of image names
    '''
    # Create a list to store all figures
    figures = []
    
    # process each result set
    for r_basic, r_enh, r_mser, s, n in zip(results_basic, results_enhanced, results_mser, scales, names):
        # calculate physical measurements for basic method
        diams_basic = np.array([2*np.sqrt(c['long_axis']*c['short_axis'])/s for c in r_basic.cluster_data])
        surf_basic = np.array([c['surface']/(s**2) for c in r_basic.cluster_data])
        
        # calculate physical measurements for enhanced method
        diams_enh = np.array([2*np.sqrt(c['long_axis']*c['short_axis'])/s for c in r_enh.cluster_data])
        surf_enh = np.array([c['surface']/(s**2) for c in r_enh.cluster_data])
        
        # calculate physical measurements for MSER method
        diams_mser = np.array([2*np.sqrt(c['long_axis']*c['short_axis'])/s for c in r_mser.cluster_data])
        surf_mser = np.array([c['surface']/(s**2) for c in r_mser.cluster_data])
        
        # Filter out outliers (particles larger than 95th percentile)
        def filter_outliers(data):
            percentile_95 = np.percentile(data, 95)
            return data[data <= percentile_95]
        
        diams_basic = filter_outliers(diams_basic)
        surf_basic = filter_outliers(surf_basic)
        diams_enh = filter_outliers(diams_enh)
        surf_enh = filter_outliers(surf_enh)
        diams_mser = filter_outliers(diams_mser)
        surf_mser = filter_outliers(surf_mser)
        
        # Calculate global min/max for aligned axes
        diam_min = min(diams_basic.min(), diams_enh.min(), diams_mser.min())
        diam_max = max(diams_basic.max(), diams_enh.max(), diams_mser.max())
        surf_min = min(surf_basic.min(), surf_enh.min(), surf_mser.min())
        surf_max = max(surf_basic.max(), surf_enh.max(), surf_mser.max())
        
        # create figure
        fig, ax = plt.subplots(3, 2, figsize=(18, 18), sharex=True, sharey=True)
        fig.suptitle(f"Analysis Results for {n}", fontsize=16, y=0.95)
        
        # Enable zoom and pan functionality
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        # Add toolbar for zoom and pan
        root = tk.Tk()
        root.title(f"Particle Detection Results - {n}")
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Link all axes for synchronized zooming and panning
        for i in range(3):
            for j in range(2):
                if i == 0 and j == 0:
                    continue  # Skip the first subplot as it will be the reference
                ax[i,j].sharex(ax[0,0])
                ax[i,j].sharey(ax[0,0])
        
        # plot diameter distribution comparison
        ax = plt.subplot(3, 2, 1)
        weights_basic = np.ones_like(diams_basic)/len(diams_basic)
        ypdf_basic, xpdfleft_basic, _ = ax.hist(diams_basic, bins=20, weights=weights_basic, 
                                               color=(147/255, 36/255, 30/255), 
                                               edgecolor='black', lw=2, rwidth=0.8,
                                               zorder=1, label='Basic Method',
                                               range=(diam_min, diam_max))
        
        xpdf_basic = xpdfleft_basic[0:-1] + np.diff(xpdfleft_basic)/2.0
        
        # Calculate and plot average point
        avg_basic = np.average(diams_basic, weights=weights_basic)
        ypos_basic = np.max(ypdf_basic)*0.05
        ax.plot(avg_basic, ypos_basic, 'o', markersize=8,
                color=(147/255, 36/255, 30/255),
                markeredgewidth=1.5, markeredgecolor="k",
                zorder=4)
        
        ax.set_title(f"Diameter Distribution (Basic Method) N = {len(diams_basic)}")
        ax.set_xlabel("Diameter (mm)")
        ax.set_ylabel("Fraction of Particles")
        
        # plot surface area distribution comparison
        ax = plt.subplot(3, 2, 2)
        weights_basic = np.ones_like(surf_basic)/len(surf_basic)
        ypdf_basic, xpdfleft_basic, _ = ax.hist(surf_basic, bins=20, weights=weights_basic,
                                               color=(147/255, 36/255, 30/255),
                                               edgecolor='black', lw=2, rwidth=0.8,
                                               zorder=1, range=(surf_min, surf_max))
        
        xpdf_basic = xpdfleft_basic[0:-1] + np.diff(xpdfleft_basic)/2.0
        
        # Calculate and plot average point
        avg_basic = np.average(surf_basic, weights=weights_basic)
        ypos_basic = np.max(ypdf_basic)*0.05
        ax.plot(avg_basic, ypos_basic, 'o', markersize=8,
                color=(147/255, 36/255, 30/255),
                markeredgewidth=1.5, markeredgecolor="k",
                zorder=16)
        
        ax.set_title(f"Surface Area Distribution (Basic Method) N = {len(surf_basic)}")
        ax.set_xlabel("Surface Area (mm²)")
        ax.set_ylabel("Fraction of Particles")
        
        # plot enhanced method diameter distribution
        ax = plt.subplot(3, 2, 3)
        weights_enh = np.ones_like(diams_enh)/len(diams_enh)
        ypdf_enh, xpdfleft_enh, _ = ax.hist(diams_enh, bins=20, weights=weights_enh, 
                                           color=(30/255, 144/255, 255/255), 
                                           edgecolor='black', lw=2, rwidth=0.8,
                                           zorder=1, label='Enhanced Method',
                                           range=(diam_min, diam_max))
        
        xpdf_enh = xpdfleft_enh[0:-1] + np.diff(xpdfleft_enh)/2.0
        
        # Calculate and plot average point
        avg_enh = np.average(diams_enh, weights=weights_enh)
        ypos_enh = np.max(ypdf_enh)*0.05
        ax.plot(avg_enh, ypos_enh, 'o', markersize=8,
                color=(30/255, 144/255, 255/255),
                markeredgewidth=1.5, markeredgecolor="k",
                zorder=4)
        
        ax.set_title(f"Diameter Distribution (Enhanced Method) N = {len(diams_enh)}")
        ax.set_xlabel("Diameter (mm)")
        ax.set_ylabel("Fraction of Particles")
        
        # plot enhanced method surface area distribution
        ax = plt.subplot(3, 2, 4)
        weights_enh = np.ones_like(surf_enh)/len(surf_enh)
        ypdf_enh, xpdfleft_enh, _ = ax.hist(surf_enh, bins=20, weights=weights_enh,
                                           color=(30/255, 144/255, 255/255),
                                           edgecolor='black', lw=2, rwidth=0.8,
                                           zorder=1, range=(surf_min, surf_max))
        
        xpdf_enh = xpdfleft_enh[0:-1] + np.diff(xpdfleft_enh)/2.0
        
        # Calculate and plot average point
        avg_enh = np.average(surf_enh, weights=weights_enh)
        ypos_enh = np.max(ypdf_enh)*0.05
        ax.plot(avg_enh, ypos_enh, 'o', markersize=8,
                color=(30/255, 144/255, 255/255),
                markeredgewidth=1.5, markeredgecolor="k",
                zorder=16)
        
        ax.set_title(f"Surface Area Distribution (Enhanced Method) N = {len(surf_enh)}")
        ax.set_xlabel("Surface Area (mm²)")
        ax.set_ylabel("Fraction of Particles")
        
        # plot MSER diameter distribution
        ax = plt.subplot(3, 2, 5)
        weights_mser = np.ones_like(diams_mser)/len(diams_mser)
        ypdf_mser, xpdfleft_mser, _ = ax.hist(diams_mser, bins=20, weights=weights_mser, 
                                             color=(50/255, 205/255, 50/255), 
                                             edgecolor='black', lw=2, rwidth=0.8,
                                             zorder=1, label='MSER Method',
                                             range=(diam_min, diam_max))
        
        xpdf_mser = xpdfleft_mser[0:-1] + np.diff(xpdfleft_mser)/2.0
        
        # Calculate and plot average point
        avg_mser = np.average(diams_mser, weights=weights_mser)
        ypos_mser = np.max(ypdf_mser)*0.05
        ax.plot(avg_mser, ypos_mser, 'o', markersize=8,
                color=(50/255, 205/255, 50/255),
                markeredgewidth=1.5, markeredgecolor="k",
                zorder=4)
        
        ax.set_title(f"Diameter Distribution (MSER Method) N = {len(diams_mser)}")
        ax.set_xlabel("Diameter (mm)")
        ax.set_ylabel("Fraction of Particles")
        
        # plot MSER surface area distribution
        ax = plt.subplot(3, 2, 6)
        weights_mser = np.ones_like(surf_mser)/len(surf_mser)
        ypdf_mser, xpdfleft_mser, _ = ax.hist(surf_mser, bins=20, weights=weights_mser,
                                             color=(50/255, 205/255, 50/255),
                                             edgecolor='black', lw=2, rwidth=0.8,
                                             zorder=1, range=(surf_min, surf_max))
        
        xpdf_mser = xpdfleft_mser[0:-1] + np.diff(xpdfleft_mser)/2.0
        
        # Calculate and plot average point
        avg_mser = np.average(surf_mser, weights=weights_mser)
        ypos_mser = np.max(ypdf_mser)*0.05
        ax.plot(avg_mser, ypos_mser, 'o', markersize=8,
                color=(50/255, 205/255, 50/255),
                markeredgewidth=1.5, markeredgecolor="k",
                zorder=16)
        
        ax.set_title(f"Surface Area Distribution (MSER Method) N = {len(surf_mser)}")
        ax.set_xlabel("Surface Area (mm²)")
        ax.set_ylabel("Fraction of Particles")
        
        # adjust and display plot
        plt.tight_layout()
        
        # Store the figure and root window
        figures.append((fig, root))
    
    return figures

### Section 1: Load and display images

# enable interactive plotting mode
plt.ion()

# Configuration flags
SKIP_COIN = False  # flag for skipping coin detection

# define image paths for processing
#images foundin 
image_paths = ['Encore_#6_Espresso.jpg',
'Encore_#18_Drip.jpg',
'Encore_#32_FrenchPress.jpg',
#'Help/Decent_Example_Picture.png',
    #'Help/Better_Example_Picture.png',
    #'Help/Bad_Example_Picture.png',
]

# define fixed analysis regions for each image
analysis_regions = [
    #(50, 360, 2200, 2200), # region for Encore_#6_Espresso.jpg  
    #(50, 360, 2200, 2200), # region for Encore_#18_Drip.jpg
    #(50, 360, 2200, 2200), # region for Encore_#32_FrenchPress.jpg
    (50, 360, 1000, 1000), # smaller region for Encore_#6_Espresso.jpg  
    (50, 360, 1000, 1000), # smaller region for Encore_#18_Drip.jpg
    (50, 360, 1000, 1000), # smaller region for Encore_#32_FrenchPress.jpg
]
# load original images
original_images = [Image.open(path) for path in image_paths]
#plot_images(original_images, [Image.open(path) for path in image_paths], [path.split('/')[-1] for path in image_paths])

### Section 2: Preprocessing placeholder

# apply preprocessing to all images
processed_images = [preprocess_image(img) for img in original_images]
plot_images(original_images, processed_images, 
           [path.split('/')[-1] for path in image_paths],
           [f"{path.split('/')[-1]} with preprocessing" for path in image_paths])


### Section 3: Reference object (coin) detection scales
#
 
# Initialize measurement lists
if SKIP_COIN:
    # Hardcoded values for each image
    print('skipping coin detection for testing speed...')
    pixel_lengths = [500] * len(image_paths)  # known pixel measurements
    physical_lengths = [23.81] * len(image_paths)  # corresponding physical measurements
    pixel_scales = [pl / phl for pl, phl in zip(pixel_lengths, physical_lengths)]
    
else:
    pixel_lengths = []
    physical_lengths = []
    pixel_scales = []

    # Prompt user to continue
    input("\nPress Enter to proceed with coin detection...")
    plt.close('all')  # Close all preprocessed image displays

    coin_diameters={'us_penny':19.05,'us_nickel':21.21,'us_dime':17.91,'us_quarter':24.26,'us_half_dollar':30.61,'canadian_penny':19.05,'canadian_nickel':21.2,'canadian_dime':18.03,'canadian_quarter':23.88,'canadian_loonie':26.5}

    # Process each image individually
    for img, path in zip(original_images, image_paths):
        print(f"\nProcessing image: {path.split('/')[-1]}")
        print("Click on the center of the reference coin")
        
        # Get coin detection
        coin_detected = False
        while not coin_detected:
            # Show image and get click
            plt.figure(figsize=(8,6))
            plt.imshow(img)
            plt.title("Click on the center of the coin")
            plt.axis('off')
            
            # Get click data
            with SuppressTkinterErrors():
                click_data = plt.ginput(1, timeout=30)
                plt.close(plt.gcf())
            
            # Check if we got valid click data
            if click_data:  # If clicked (ginput returns a list of tuples)
                click_x, click_y = click_data[0]  # Unpack the first (and only) tuple
                print(f"Click detected at ({click_x}, {click_y})")
                
                # Detect coin and get parameters
                pixel_diameter, circle_params = detect_coin(img, click_x, click_y)
                
                # Show detection result
                with SuppressTkinterErrors():
                    plt.figure(figsize=(8,6))
                    plt.imshow(img)
                    if circle_params is not None:
                        x, y, r = circle_params
                        circle = plt.Circle((x,y), r, color='red', fill=False, linewidth=0.5)
                        plt.gca().add_patch(circle)
                        plt.title(f"Detected Coin (Diameter: {pixel_diameter} pixels)")
                    else:
                        plt.title("No Coin Detected")
                        plt.close(plt.gcf())
                        print("No coin detected. Please try again.")
                        continue
                    
                    plt.axis('off')
                    plt.show(block=False)
                    plt.pause(0.1)
                
                # Get confirmation before closing the window
                confirm = input("Is the coin properly highlighted? (y/n): ").strip().lower()
                
                # Close the current figure without using close('all')
                with SuppressTkinterErrors():
                    plt.close(plt.gcf())
                
                if confirm == 'y':
                    coin_detected = True
                else:
                    print("Reprompting for click...")
                    continue
        
        if not coin_detected:
            print(f"Skipping image {path.split('/')[-1]} due to no valid coin detection")
            continue
        
        # Get coin type
        while True:
            print("Available coin types:", ', '.join(coin_diameters.keys()))
            coin_type = input("Enter the type of coin in the image (e.g., us_quarter): ").strip().lower()
            if coin_type in coin_diameters:
                break
            print("Invalid coin type. Please try again.")
        
        plt.close('all')
        
        # Calculate and store measurements
        physical_diameter = coin_diameters[coin_type]
        pixel_lengths.append(pixel_diameter)
        physical_lengths.append(physical_diameter)
        pixel_scales.append(pixel_diameter/physical_diameter)
        print(f"Physical diameter ({coin_type}): {physical_diameter} mm")
        print(f"Pixel scale: {pixel_diameter/physical_diameter} pixels/mm")


### Section 4: Thresholding/Segmenting pipeline

# Process images using all three methods
print('Running all particle detection methods...')
all_results_original_basic = []
all_results_original_enhanced = []
all_results_mser = []
vis_figures = []  # List for visualization figures

for img, analysis_region, img_path in zip(processed_images, analysis_regions, image_paths):
    # print current file being processed
    print(f'\nProcessing {img_path.split("/")[-1]}...')
    
    # Run basic original method (no preprocessing, no watershed)
    print('\n' + '_'*50)
    print('Starting Basic Original Method')
    print('_'*50)
    start_time = time.time()
    cropped, mask, imdata, bg_median = threshold_image(img, analysis_region)
    clusters_original_basic = launch_psd(mask, imdata, bg_median)
    end_time = time.time()
    print(f"Basic original method completed in {end_time - start_time:.2f} seconds")
    
    # Store basic original method results
    res_original_basic = ProcessingResults()
    res_original_basic.mask_threshold = mask
    res_original_basic.cluster_data = clusters_original_basic
    res_original_basic.nclusters = len(clusters_original_basic)
    res_original_basic.clusters_surface = [c['surface'] for c in clusters_original_basic]
    res_original_basic.clusters_long_axis = [c['long_axis'] for c in clusters_original_basic]
    res_original_basic.clusters_short_axis = [c['short_axis'] for c in clusters_original_basic]
    all_results_original_basic.append(res_original_basic)
    
    # Run enhanced original method (with preprocessing and watershed)
    print('\n' + '_'*50)
    print('Starting Enhanced Original Method')
    print('_'*50)
    start_time = time.time()
    cropped, mask, imdata, bg_median = threshold_image(img, analysis_region)
    
    # Create binary mask and find connected components
    binary_mask = np.zeros_like(imdata, dtype=np.uint8)
    binary_mask[mask] = 255
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    clump_labels = [i for i, stat in enumerate(stats) if stat[cv2.CC_STAT_AREA] > 150 and i != 0]
    clump_mask = np.isin(labels, clump_labels).astype(np.uint8) * 255
    
    # Apply watershed segmentation
    print('Applying watershed segmentation...')
    cropped_cv = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
    particle_mask, markers = watershed_alg(cropped_cv, clump_mask)
    
    # Merge results: keep fine particles + split clumps
    final_mask = binary_mask.copy()
    final_mask[clump_mask > 0] = 0  # remove clump blobs
    final_mask[particle_mask] = 255
    particle_mask_final = np.where(final_mask > 0)
    
    # First pass particle detection
    print('Running first pass particle detection...')
    clusters_original_enhanced = launch_psd(particle_mask_final, imdata, bg_median)
    
    # Second pass for smaller particles
    print("Running second pass for smaller particle detection...")
    cluster_mask = np.zeros_like(imdata, dtype=np.uint8)
    for c in clusters_original_enhanced:
        for x, y in c['points']:
            x = int(np.clip(x, 0, cluster_mask.shape[1] - 1))
            y = int(np.clip(y, 0, cluster_mask.shape[0] - 1))
            cluster_mask[y, x] = 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(cluster_mask, kernel, iterations=1)
    
    bg_val = int(np.median(imdata[cluster_mask == 0]))
    residual_imdata = imdata.copy()
    residual_imdata[dilated_mask > 0] = bg_val
    
    residual_thresh = np.where(residual_imdata < bg_val * 0.588)  # match threshold percent
    
    residual_clusters = launch_psd(residual_thresh, residual_imdata, bg_median)
    
    non_overlap_res = []
    total_residual = len(residual_clusters)
    print(f"\nProcessing {total_residual} residual clusters for overlap checking...")
    
    for i, rc in enumerate(residual_clusters, 1):
        overlaps = False
        for c1 in clusters_original_enhanced:
            if cluster_overlap(rc, c1):
                overlaps = True
                break
        if not overlaps:
            non_overlap_res.append(rc)
        
        # Print progress every 20% or for every 200 clusters
        if i % max(1, total_residual // 5) == 0 or i % 200 == 0:
            progress = (i / total_residual) * 100
            print(f"Progress: {progress:.1f}% ({i}/{total_residual} clusters processed)")
    
    print(f"Found {len(non_overlap_res)} non-overlapping residual clusters")
    
    # Combine first and second pass results
    clusters_original_enhanced += non_overlap_res
    end_time = time.time()
    print(f"Enhanced original method completed in {end_time - start_time:.2f} seconds")
    
    # Store enhanced original method results
    res_original_enhanced = ProcessingResults()
    res_original_enhanced.mask_threshold = mask
    res_original_enhanced.cluster_data = clusters_original_enhanced
    res_original_enhanced.nclusters = len(clusters_original_enhanced)
    res_original_enhanced.clusters_surface = [c['surface'] for c in clusters_original_enhanced]
    res_original_enhanced.clusters_long_axis = [c['long_axis'] for c in clusters_original_enhanced]
    res_original_enhanced.clusters_short_axis = [c['short_axis'] for c in clusters_original_enhanced]
    all_results_original_enhanced.append(res_original_enhanced)
    
    # Run MSER method
    print('\n' + '_'*50)
    print('Starting MSER Method')
    print('_'*50)
    start_time = time.time()
    clusters_mser = detect_particles_mser(img, analysis_region)
    end_time = time.time()
    print(f"MSER method completed in {end_time - start_time:.2f} seconds")
    
    # Print cluster counts for all methods
    print(f"\nCluster counts for {img_path.split('/')[-1]}:")
    print(f"Basic original method: {len(clusters_original_basic)} clusters")
    print(f"Enhanced original method: {len(clusters_original_enhanced)} clusters")
    print(f"MSER method: {len(clusters_mser)} clusters")
    
    # Store MSER results
    res_mser = ProcessingResults()
    res_mser.cluster_data = clusters_mser
    res_mser.nclusters = len(clusters_mser)
    res_mser.clusters_surface = [c['surface'] for c in clusters_mser]
    res_mser.clusters_long_axis = [c['long_axis'] for c in clusters_mser]
    res_mser.clusters_short_axis = [c['short_axis'] for c in clusters_mser]
    all_results_mser.append(res_mser)

    # create visualization figure
    fig_vis, ax_vis = plt.subplots(3, 3, figsize=(18, 18), sharex=True, sharey=True)
    fig_vis.suptitle(f"Visualization Results for {img_path.split('/')[-1]}", fontsize=16, y=0.95)
    
    # Enable zoom and pan functionality
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Create Tkinter window for visualization
    root_vis = tk.Tk()
    root_vis.title(f"Particle Detection Visualization - {img_path.split('/')[-1]}")
    canvas_vis = FigureCanvasTkAgg(fig_vis, master=root_vis)
    canvas_vis.draw()
    toolbar_vis = NavigationToolbar2Tk(canvas_vis, root_vis)
    toolbar_vis.update()
    canvas_vis.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    # Link all axes for synchronized zooming and panning
    for i in range(3):
        for j in range(3):
            if i == 0 and j == 0:
                continue  # Skip the first subplot as it will be the reference
            ax_vis[i,j].sharex(ax_vis[0,0])
            ax_vis[i,j].sharey(ax_vis[0,0])
    
    # Plot basic original method results
    # Original cropped region
    ax_vis[0,0].imshow(cropped)
    ax_vis[0,0].set_title("Analysis Region")
    
    # Threshold result
    overlay_basic = cropped.convert('RGB')
    overlay_basic = np.array(overlay_basic)
    arr_basic = overlay_basic.copy()
    arr_basic[mask] = [255, 0, 0]  # mark detected particles in red
    ax_vis[0,1].imshow(arr_basic)
    ax_vis[0,1].set_title("Threshold Result")
    
    # Basic original method particles
    ax_vis[0,2].imshow(overlay_basic)
    for c in clusters_original_basic:
        x, y = zip(*c['points'])
        x = np.array(x)
        y = np.array(y)
        try:
            hull = ConvexHull(np.column_stack((y, x)), qhull_options='QJ')
            poly = patches.Polygon(hull.points[hull.vertices], 
                                 closed=True,
                                 fill=False, 
                                 edgecolor='red', 
                                 linewidth=0.5)
            ax_vis[0,2].add_patch(poly)
        except Exception as e:
            print(f"Warning: Could not create hull for cluster with {len(x)} points: {str(e)}")
            continue
    ax_vis[0,2].set_title(f"Basic Original Method Particles: {len(clusters_original_basic)}")
    
    # Plot enhanced original method results
    # Original cropped region
    ax_vis[1,0].imshow(cropped)
    ax_vis[1,0].set_title("Analysis Region")
    
    # Watershed segmentation result
    ax_vis[1,1].imshow(markers, cmap='nipy_spectral')
    ax_vis[1,1].set_title("Watershed Segmentation")
    
    # Enhanced original method particles
    overlay_enhanced = cropped.convert('RGB')
    overlay_enhanced = np.array(overlay_enhanced)
    ax_vis[1,2].imshow(overlay_enhanced)
    for c in clusters_original_enhanced:
        x, y = zip(*c['points'])
        x = np.array(x)
        y = np.array(y)
        try:
            hull = ConvexHull(np.column_stack((y, x)), qhull_options='QJ')
            poly = patches.Polygon(hull.points[hull.vertices], 
                                 closed=True,
                                 fill=False, 
                                 edgecolor='red', 
                                 linewidth=0.5)
            ax_vis[1,2].add_patch(poly)
        except Exception as e:
            print(f"Warning: Could not create hull for cluster with {len(x)} points: {str(e)}")
            continue
    ax_vis[1,2].set_title(f"Enhanced Original Method Particles: {len(clusters_original_enhanced)}")
    
    # Plot MSER method results
    # Grayscale visualization
    img_array = np.array(cropped)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img_gray = 255 - img_gray
    img_blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
    
    ax_vis[2,0].imshow(img_blurred, cmap='gray')
    ax_vis[2,0].set_title("Grayscale (Inverted)")
    
    # MSER detected regions
    ax_vis[2,1].imshow(img_blurred, cmap='gray')
    for c in clusters_mser:
        points = np.array(c['points'])
        poly = patches.Polygon(points, 
                             closed=True,
                             fill=False, 
                             edgecolor='red', 
                             linewidth=0.5)
        ax_vis[2,1].add_patch(poly)
    ax_vis[2,1].set_title("MSER Detected Regions")
    
    # MSER particles on original image
    overlay_mser = cropped.convert('RGB')
    overlay_mser = np.array(overlay_mser)
    ax_vis[2,2].imshow(overlay_mser)
    for c in clusters_mser:
        points = np.array(c['points'])
        poly = patches.Polygon(points, 
                             closed=True,
                             fill=False, 
                             edgecolor='red', 
                             linewidth=0.5)
        ax_vis[2,2].add_patch(poly)
    ax_vis[2,2].set_title(f"MSER Particles: {len(clusters_mser)}")
    
    # adjust and display plot
    plt.tight_layout()
    
    # Store the visualization figure and root window
    vis_figures.append((fig_vis, root_vis))

# generate histogram plots comparing all three methods
hist_figures = plot_histograms_comparison(all_results_original_basic, all_results_original_enhanced, all_results_mser, 
                                        pixel_scales, 
                                        [p.split('/')[-1] for p in image_paths])

# finalize plotting
plt.ioff()  # disable interactive mode

# Force close all figures after data calculation
plt.close('all')

print("\nAll plots have been generated. Close the plot windows to continue...")

# Wait for all Tkinter windows to be closed by the user
for _, root in vis_figures + hist_figures:
    root.mainloop()

# Clean up
for fig, root in vis_figures + hist_figures:
    plt.close(fig)
    root.destroy()