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
        plt.axis('off')
    
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
    return img  # currently no preprocessing

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
    '''
    Performs rapid clustering using breadth-first search.
    
    Args:
        xlist, ylist (np.array): Coordinates of points
        xstart, ystart: Starting point coordinates
    Returns:
        np.array: Indices of points in cluster
    '''
    # get total number of points
    N = xlist.size
    
    # store original coordinates
    X = xlist
    Y = ylist
    idx = np.arange(N)

    # track unprocessed points
    alive = np.ones(N, dtype=bool)

    # find and validate start point
    istart = np.where((X == xstart) & (Y == ystart))[0]
    if istart.size == 0:
        return np.array([], dtype=int)
    
    # mark start point as processed
    alive[istart] = False
    queue = [istart[0]]
    output = [istart[0]]

    # process queue until empty
    while queue:
        cur = queue.pop(0)

        # get candidates from remaining alive points
        candidates = idx[alive]
        dx = np.abs(X[candidates] - X[cur])
        dy = np.abs(Y[candidates] - Y[cur])

        # find adjacent neighbors (manhattan distance <= 1)
        neigh = candidates[(dx + dy) <= 1]
        if neigh.size == 0:
            continue

        # update tracking arrays
        alive[neigh] = False
        output.extend(neigh.tolist())
        queue.extend(neigh.tolist())

    return np.array(output, dtype=int)

def launch_psd(mask, imdata, bg_median,
               max_cluster_axis=100, min_surface=5,
               reference_threshold=0.4, maxcost=0.35, nsmooth=3):
    '''
    Performs Particle Size Distribution (PSD) analysis using connected component labeling
    and path-based clustering.

    Args:
        mask (tuple): Binary mask coordinates (x_coords, y_coords)
        imdata (np.ndarray): Original image data
        bg_median (float): Background median intensity
        max_cluster_axis (int): Maximum allowed cluster axis length
        min_surface (int): Minimum allowed cluster surface area
        reference_threshold (float): Threshold for reference point selection
        maxcost (float): Maximum allowed path cost
        nsmooth (int): Window size for path cost smoothing

    Returns:
        list: List of dictionaries containing cluster properties
    '''
    # extract coordinates from mask
    X = mask[0].astype(int); Y = mask[1].astype(int)
    n = X.size

    # initialize tracking arrays
    counted = np.zeros(n, bool)
    clusters = []
    cluster_count = 0  # add counter for progress tracking

    #for each point
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

        #validate cluster size
        if iclust.size < min_surface:
            counted[curr] = True; continue
        
        # calculate intensity-based costs
        cost = np.maximum((imdata[mask][iclust] - imdata[mask][curr])**2 / bg_median**2, 0)
        filt = np.array([np.where(iclust == curr)[0][0]], int)
        maxpath = np.full(iclust.size, np.nan)

        # analyze paths through cluster
        for ci in range(iclust.size):
            if iclust[ci] == curr: continue
            # calculate threshold and find dark points
            vals = imdata[mask][iclust[filt]]
            thr = (bg_median - imdata[mask][curr]) * reference_threshold + imdata[mask][curr]
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
        #print('append')

        # store cluster information
        clusters.append({
            'surface': surf,
            'long_axis': axis,
            'short_axis': surf/(np.pi*axis),
            'xmean': xm,
            'ymean': ym,
            'points': list(zip(xs, ys))
        })
                # increment counter and print progress
        cluster_count += 1
        if cluster_count % 10 == 0:
            print(f"{cluster_count} clusters identified")

    # print final count
    if cluster_count > 0:
        print(f"Final count: {cluster_count} clusters identified")
    else:
        print("No clusters identified")

    return clusters


def plot_histograms(results, scales, names):
    '''
    Creates histograms of particle measurements.
    
    Args:
        results: List of ProcessingResults objects
        scales: List of pixel-to-physical scales
        names: List of image names
    '''
    # process each result set
    for r, s, n in zip(results, scales, names):
        # calculate physical measurements
        diams = [2*c['long_axis']/s for c in r.cluster_data]
        surf = [c['surface']/(s**2) for c in r.cluster_data]
        
        # create figure
        plt.figure(figsize=(12, 5))
        
        # plot diameter distribution
        plt.subplot(1, 2, 1)
        plt.hist(diams, bins=20, edgecolor='black')
        plt.title(f"{n}\nDiameter Distribution (N={len(diams)})")
        plt.xlabel("Diameter (mm)")
        plt.ylabel("Count")
        
        # plot surface area distribution
        plt.subplot(1, 2, 2)
        plt.hist(surf, bins=20, edgecolor='black')
        plt.title(f"{n}\nSurface Area Distribution")
        plt.xlabel("Surface Area (mm²)")
        plt.ylabel("Count")
        
        # adjust and display plot
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)


### Section 1: Load and display images

# enable interactive plotting mode
plt.ion()

# define image paths for processing
image_paths = [
    #'C:/Users/Andre/Documents/Decent_Example_Picture.png',
    'C:/Users/Andre/Documents/Better_Example_Picture.png'
]

# load original images
original_images = [Image.open(path) for path in image_paths]
plot_images(original_images, [path.split('/')[-1] for path in image_paths])

### Section 2: Preprocessing placeholder

# apply preprocessing to all images
processed_images = [preprocess_image(img) for img in original_images]
plot_images(processed_images, [f"{path.split('/')[-1]} with preprocessing" for path in image_paths])


### Section 3: Reference object (coin) detection scales (hardcoded for now)

# define calibration scales
pixel_lengths = [500]  # known pixel measurements
physical_lengths = [23.81]  # corresponding physical measurements
pixel_scales = [pl / phl for pl, phl in zip(pixel_lengths, physical_lengths)]

# define fixed analysis region
analysis_region = (1100, 500, 2500, 1200)
all_results = []


### Section 4: Thresholding/Segmenting pipeline

# process each image in the dataset
for img in processed_images:
    # apply thresholding and get masks
    print('thhresholding...')
    cropped, mask, imdata, bg_median = threshold_image(img, analysis_region)
    
    # detect particle clusters
    print('identifying particles...')
    clusters = launch_psd(mask, imdata, bg_median)
    
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
    arr[mask] = [255, 0, 0]  # mark detected particles in red
    ax[1].imshow(arr)
    ax[1].set_title("Threshold Result")
    
    # plot detected clusters
    ax[2].imshow(overlay)
    for c in clusters:
        # extract cluster points
        x, y = zip(*c['points'])
        
        # compute convex hull
        hull = ConvexHull(np.column_stack((y, x)))
        
        # create polygon visualization
        poly = patches.Polygon(hull.points[hull.vertices], 
                             closed=True,
                             fill=False, 
                             edgecolor='red', 
                             linewidth=0.5)
        ax[2].add_patch(poly)
    ax[2].set_title(f"Particles Detected: {len(clusters)}")
    
    # adjust and display plot
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


### Section 4.5: Breaking up of overlapping particles that were identified (watershed algorithm, etc)
# may make more sense to have this directly within section 4 functions above depending on workflow


### Section 5: Analysis - basic histograms shown for now. 

# generate histogram plots
plot_histograms(all_results, 
                pixel_scales, 
                [p.split('/')[-1] for p in image_paths])

# finalize plotting
plt.ioff()  # disable interactive mode
plt.show()  # show all figures