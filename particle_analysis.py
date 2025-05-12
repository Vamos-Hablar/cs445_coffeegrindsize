from skimage.measure import regionprops, label
import numpy as np

def analyze_particle_shapes(segmented_image):
    labeled_image = label(segmented_image)
    props = regionprops(labeled_image)

    shape_data = []
    for prop in props:
        area = prop.area
        perimeter = prop.perimeter if prop.perimeter > 0 else 1
        circularity = 4 * np.pi * area / (perimeter ** 2)
        solidity = prop.solidity
        aspect_ratio = prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else 0

        shape_data.append({
            'area': area,
            'circularity': circularity,
            'solidity': solidity,
            'aspect_ratio': aspect_ratio
        })
    return shape_data

def compute_shape_score(shape_data):
    scores = []
    for data in shape_data:
        # Example: weighted sum of circularity and solidity
        score = 0.5 * data['circularity'] + 0.5 * data['solidity']
        scores.append(score)
    return scores

def count_fines(shape_data, area_threshold):
    fines = [data for data in shape_data if data['area'] < area_threshold]
    return len(fines)
