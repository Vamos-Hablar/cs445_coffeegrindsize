'''
Particle Analysis and Measurement System

This script processes microscope images to detect and analyze particles.
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


def plot_images(images, titles, figsize=(15, 5)):
    '''
    Displays multiple images in a single row with titles.
    
    Args:
        images (list): List of images to display
        titles (list): Corresponding titles for each image
        figsize (tuple): Figure dimensions (width, height)
    '''
    # create new figure
    plt.figure(figsize=figsize)
    
    # plot each image with its title
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(img)
        plt.title(title)
        #plt.axis('off')
    
    # adjust layout and display
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    
def preprocess_image(img):
    '''
    Placeholder for image preprocessing steps.
    
    Args:
        img (PIL.Image): Input image
    Returns:
        PIL.Image: Processed image
    '''
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # denoised = cv2.fastNlMeansDenoisingColored(img_cv, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    res = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

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
        plt.show(block=False)
        plt.pause(0.1)


### Section 1: Load and display images

# enable interactive plotting mode
plt.ion()

# define image paths for processing
#images foundin 
image_paths = [
    # 'Help/Decent_Example_Picture.png',
    'Help/Better_Example_Picture.png',
    #'Help/Bad_Example_Picture.png',
]

# load original images
original_images = [Image.open(path) for path in image_paths]
plot_images(original_images, [path.split('/')[-1] for path in image_paths])

### Section 2: Preprocessing placeholder

# apply preprocessing to all images
processed_images = [preprocess_image(img) for img in original_images]
plot_images(processed_images, [f"{path.split('/')[-1]} with preprocessing" for path in image_paths])

# Prompt user to continue
input("\nPress Enter to proceed with coin detection...")
plt.close('all')  # Close all preprocessed image displays

### Section 3: Reference object (coin) detection scales
coin_diameters={'us_penny':19.05,'us_nickel':21.21,'us_dime':17.91,'us_quarter':24.26,'us_half_dollar':30.61,'canadian_penny':19.05,'canadian_nickel':21.2,'canadian_dime':18.03,'canadian_quarter':23.88,'canadian_loonie':26.5}
def detect_coin(image_pil,click_x,click_y):
    print('detecting coin...')
    img_array=np.array(image_pil)
    img_bgr=cv2.cvtColor(img_array,cv2.COLOR_RGB2BGR)
    gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
    blurred=cv2.GaussianBlur(gray,(9,9),2)
    edges=cv2.Canny(blurred,30,100)
    circles=cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,dp=1,minDist=100,param1=100,param2=20,minRadius=30,maxRadius=300)
    if circles is not None:
        circles=np.round(circles[0,:]).astype("int")
        filtered_circles=[]
        for(x,y,r)in circles:
            dist=np.hypot(x-click_x,y-click_y)
            if dist<50:
                filtered_circles.append((x,y,r))
        if not filtered_circles:
            return None,None
        filtered_circles=sorted(filtered_circles,key=lambda c:c[2],reverse=True)
        x,y,r=filtered_circles[0]
        diameter_pixels=2*r
        return diameter_pixels,(x,y,r)
    return None,None
pixel_lengths=[]
physical_lengths=[]
pixel_scales=[]

# Process each image individually
for img, path in zip(original_images, image_paths):
    print(f"\nProcessing image: {path.split('/')[-1]}")
    print("Click on the center of the reference coin")
    
    # Reset click data for each image
    click_data = [None, None, False]
    coin_detected = False
    
    while not coin_detected:
        plt.close('all')  # Close any existing figures
        fig, ax = plt.subplots(figsize=(8,6))
        ax.imshow(img)
        ax.set_title("Click on the center of the coin (or press Enter to skip)")
        ax.axis('off')
        
        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                click_data[0] = int(event.xdata)
                click_data[1] = int(event.ydata)
                click_data[2] = True
                plt.close('all')
        
        # Connect the onclick event to the figure
        fig.canvas.mpl_connect('button_press_event', onclick)
        
        # Show the figure and wait for the user to click
        plt.show(block=False)
        
        # Give user time to click (30 seconds timeout)
        timeout = 30
        start_time = time.time()
        while not click_data[2] and time.time() - start_time < timeout:
            plt.pause(0.1)
        
        # Ensure all figures are closed
        plt.close('all')
        
        # Check if we got valid click data
        if click_data[2]:  # If clicked
            click_x, click_y = click_data[0], click_data[1]
            print(f"Click detected at ({click_x}, {click_y})")
            
            # Detect coin and get parameters
            pixel_diameter, circle_params = detect_coin(img, click_x, click_y)
            
            # Show detection result
            plt.figure(figsize=(8,6))
            plt.imshow(img)
            if circle_params is not None:
                x, y, r = circle_params
                ##added a mask for the coin
                coin_mask = np.zeros_like(np.array(img)[:, :, 0], dtype=np.uint8)
                cv2.circle(coin_mask, (x, y), r + 10, 255, -1)

                circle = plt.Circle((x,y), r, color='red', fill=False, linewidth=0.5)
                plt.gca().add_patch(circle)
                plt.title(f"Detected Coin (Diameter: {pixel_diameter} pixels)")
            else:
                plt.title("No Coin Detected")
                plt.close()
                print("No coin detected. Please try again.")
                continue
            
            plt.axis('off')
            plt.show(block=False)
            plt.pause(0.1)
            
            confirm = input("Is the coin properly highlighted? (y/n): ").strip().lower()
            if confirm == 'y':
                coin_detected = True
            else:
                print("Reprompting for click...")
                continue
        else:
            print("No click detected within timeout. Skipping image.")
            break
    
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

# define fixed analysis regions for each image
analysis_regions = [
    #(250, 250, 3000, 2500),  # region for Decent_Example_Picture.png
    # (250, 250, 1000, 1000),  # region for Better_Example_Picture.png
    (1100, 500, 2500, 1200), # smaller region for Better_Example_Picture.png
    # (1000, 250, 3500, 3000),  # region for Bad_Example_Picture.png
    #(575, 300, 2000, 1000),  # smaller region for Bad_Example_Picture.png for speed purposes
    #(575, 300, 3000, 2000),  # region for Bad_Example_Picture.png
]
all_results = []


#####

### Section 4.5: Breaking up of overlapping particles that were identified (watershed algorithm, etc)
# may make more sense to have this directly within section 4 functions above depending on workflow

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


### Section 4: Thresholding/Segmenting pipeline

# process each image in the dataset
for img, analysis_region, img_path in zip(processed_images, analysis_regions, image_paths):
    # print current file being processed
    print(f'\nProcessing {img_path.split("/")[-1]}...')
    
    # apply thresholding and get masks
    print('thresholding...')
    cropped, mask, imdata, bg_median = threshold_image(img, analysis_region)

    binary_mask = np.zeros_like(imdata, dtype=np.uint8)
    binary_mask[mask] = 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    clump_labels = [i for i, stat in enumerate(stats) if stat[cv2.CC_STAT_AREA] > 150 and i != 0]
    clump_mask = np.isin(labels, clump_labels).astype(np.uint8) * 255

    #exclude coin for analysis
    x1, y1, x2, y2 = analysis_region
    cropped_mask = coin_mask[y1:y2, x1:x2]  # match the cropped region
    binary_mask[cropped_mask > 0] = 0
    clump_mask[cropped_mask > 0] = 0

    print('applying watershed...')
    # Convert PIL to OpenCV image (cropped is PIL.Image)
    cropped_cv = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
    particle_mask, markers = watershed_alg(cropped_cv, clump_mask)

    # Merge results: keep fine particles + split clumps
    final_mask = binary_mask.copy()
    final_mask[clump_mask > 0] = 0  #remove clump blobs
    final_mask[particle_mask] = 255
    particle_mask_final = np.where(final_mask > 0)

    #visualize watershed segmentation
    plt.imshow(markers, cmap='nipy_spectral')
    plt.title("Watershed Segmentation")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.1)
    
    # detect particle clusters
    print('identifying particles...')
    clusters = launch_psd(particle_mask_final, imdata, bg_median)

    #second-pass
    print("Running second pass for smaller particle detection...")
    cluster_mask = np.zeros_like(imdata, dtype=np.uint8)
    for c in clusters:
        for x, y in c['points']:
            x = int(np.clip(x, 0, cluster_mask.shape[1] - 1))
            y = int(np.clip(y, 0, cluster_mask.shape[0] - 1))
            cluster_mask[y, x] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(cluster_mask, kernel, iterations=1)

    bg_val = int(np.median(imdata[cluster_mask == 0]))
    residual_imdata = imdata.copy()
    residual_imdata[dilated_mask > 0] = bg_val

    residual_thresh = np.where(residual_imdata < bg_val * 0.588)  #match threshold percent

    residual_clusters = launch_psd(residual_thresh, residual_imdata, bg_median)

    #remove any overlapping clusters
    def cluster_overlap(c1, c2):
        for pt in c1['points']:
            if pt in c2['points']:
                return True
        return False

    non_overlap_res = []
    for rc in residual_clusters:
        overlaps = False
        for c1 in clusters:
            if cluster_overlap(rc, c1):
                overlaps = True
                break

        if not overlaps:
            non_overlap_res.append(rc)
    clusters += non_overlap_res

    # store analysis results
    res = ProcessingResults()
    res.mask_threshold = mask
    res.cluster_data = clusters
    res.nclusters = len(clusters)
    res.clusters_surface = [c['surface'] for c in clusters]
    res.clusters_long_axis = [c['long_axis'] for c in clusters]
    res.clusters_short_axis = [c['short_axis'] for c in clusters]
    all_results.append(res)

    # create visualization figure
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # plot original cropped region
    ax[0].imshow(cropped)
    ax[0].set_title("Analysis Region")
    
    # create and plot threshold overlay
    overlay = cropped.convert('RGB')
    arr = np.array(overlay)
    arr[particle_mask] = [255, 0, 0]  # mark detected particles in red
    ax[1].imshow(arr)
    ax[1].set_title("Threshold Result")
    
    # plot detected clusters
    ax[2].imshow(overlay)
    for c in clusters:
        # extract cluster points
        x, y = zip(*c['points'])
        x = np.array(x)
        y = np.array(y)
        
        # compute convex hull with QJ option to handle collinear points
        try:
            hull = ConvexHull(np.column_stack((y, x)), qhull_options='QJ')
            # create polygon visualization
            poly = patches.Polygon(hull.points[hull.vertices], 
                                 closed=True,
                                 fill=False, 
                                 edgecolor='red', 
                                 linewidth=0.5)
            ax[2].add_patch(poly)
        except Exception as e:
            print(f"Warning: Could not create hull for cluster with {len(x)} points: {str(e)}")
            continue
    ax[2].set_title(f"Particles Detected: {len(clusters)}")
    
    # adjust and display plot
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


### Section 5: Analysis - basic histograms shown for now. 

# generate histogram plots
plot_histograms(all_results, 
                pixel_scales, 
                [p.split('/')[-1] for p in image_paths])

# finalize plotting
plt.ioff()  # disable interactive mode
plt.show()  # show all figures