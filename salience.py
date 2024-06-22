"""Script for calculating the local vessel salience (LVS) and the low salience
recal (LSRecall)."""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import skeletonize
from skimage import draw
from scipy.spatial import KDTree
import scipy.ndimage as ndi
import cv2
from pyvane.graph.creation import create_graph
from pyvane.graph import adjustment as net_adjust
from pyvane.image import Image as Image_pv
from pyvane.util import graph_to_img

def get_graph(img_bin):
    """Create graph representing blood vessel topology."""

    img_skel = skeletonize(img_bin)
    img_skel_pv = Image_pv(img_skel)
    graph = create_graph(img_skel_pv)
    graph_simple = net_adjust.simplify(graph)
    graph_final = net_adjust.adjust_graph(graph_simple, length_threshold=5)

    return graph_final

def get_contour(img_bin):
    """Get the contours of the blood vessels."""

    contours, _ = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Remove additional dimension from each contour and invert x and y. 
    # contour[i] contains a pixel at position (row, col)
    contour = np.concatenate([contour[:,0,::-1] for contour in contours])

    return contour

def get_closest_points(graph, contour, k=50, show_warnings=False):
    """For each medial axis point, get two points in the contour that are closest 
    to the medial axis point. The two contour points are constrained to be on 
    opposite sides of the medial axis. Parameter `k` controls the number of contour 
    points that are analysed. Larger values lead to higher computational cost, 
    but if k is too low, the second contour point might not be found.
    
    The function returns a list of lists. Each element of the list corresponds to
    a medial axis (a graph edge), and each sublist contains the two indices of
    the closest contour points to each respective medial axis pixel. The indices
    are relative to the list `contour`.    
    """

    # kdtree of the contour
    kdtree = KDTree(contour)
    # Flatten list of skeleton pixels
    paths = [path[2] for path in graph.edges(data='path')]
    pixels = [p for path in paths for p in path]

    point_map = []
    for v1,v2,path in graph.edges(data='path'):
        point_map_path = []
        for point_idx, point in enumerate(path):
            
            # We could use kdtree.query(point, k=[k]) to get only the k-th nearest neighbor, but the order
            # of the points is random when the distance is identical, which causes some missing points
            dists, nei_indices = kdtree.query(point, k=k)
            nei1_idx = nei_indices[0]
            nei1_point = contour[nei1_idx]
    
            # Vector going from the skeleton point to the closest contour point
            nei1_vec = nei1_point - point
            
            dot_prod = 1
            idx = 1
            # Find another contour point having a negative dot product with `nei1_vec`
            while dot_prod>=0 and idx<k:
                nei2_idx = nei_indices[idx]
                nei2_point = contour[nei2_idx]
                nei2_vec = nei2_point - point
                dot_prod = np.dot(nei1_vec, nei2_vec)
                idx += 1
    
            if idx==k:
                if show_warnings:
                    print('Warning, parameter k is not large enough')
                point_map_path.append( (nei1_idx, None))
            else:
                point_map_path.append(( nei1_idx, nei2_idx))
        
        point_map.append(point_map_path)

    return point_map

def get_intensities(img, img_bin, graph, contour, closest_points, radius=5):
    """Get image intensities around each medial axis point. The function returns 
    a list of lists. Each element of the list corresponds to a medial axis (a 
    graph edge), and each sublist contains a set of dictionaries for each
    medial axis pixel. 
    
    The items of the dictionary are
    
    'int_vessel'    # Vessel intensities
    'pix_vessel'    # Vessel pixels
    'int_back'      # Background intensities
    'pix_back'      # Background pixels
    """

    paths = [path[2] for path in graph.edges(data='path')]

    section_data = []
    for path, closest_points_path in zip(paths, closest_points):
        # For each segment

        section_data_path = []     
        for p, (nei1_idx, nei2_idx) in zip(path, closest_points_path):
            # For each pixel on the medial axis of a segment
            
            n1 = contour[nei1_idx]
            # Vessel and background positions for the nearest point on the contour
            rrv_1, ccv_1 = draw.line(n1[0], n1[1], p[0], p[1])
            rrb_1, ccb_1 = draw.disk((n1[0], n1[1]), radius, shape=img.shape)
    
            if nei2_idx is None:
                # Only the nearest point is used
                rrv = rrv_1
                ccv = ccv_1
                rrb = rrb_1
                ccb = ccb_1
            else:
                # Get positions for the oposite contour point
                n2 = contour[nei2_idx]     
                rrv_2, ccv_2 = draw.line(p[0], p[1], n2[0], n2[1])
                # Remove skeleton point p, since it is already on rrv_1 and ccv_1
                rrv_2, ccv_2 = rrv_2[1:], ccv_2[1:]
                rrv = np.concatenate((rrv_1, rrv_2))
                ccv = np.concatenate((ccv_1, ccv_2))
    
                rrb_2, ccb_2 = draw.disk((n2[0], n2[1]), radius, shape=img.shape)
                rrb = np.concatenate((rrb_1, rrb_2))
                ccb = np.concatenate((ccb_1, ccb_2))
    
            int_vessel = img[rrv, ccv]   
            # Keep only positions on the background     
            mask = img_bin[rrb, ccb]==0
            rrb, ccb = rrb[mask], ccb[mask]
            int_background = img[rrb, ccb]
    
            section_data_path.append({
                'int_vessel':int_vessel,            # Vessel intensities
                'pix_vessel':list(zip(rrv, ccv)),   # Vessel pixels
                'int_back':int_background,          # Background intensities
                'pix_back':list(zip(rrb, ccb))      # Background pixels
                })

        section_data.append(section_data_path)

    return section_data

def get_statistics(section_data):
    """Calculate statistics about vessel and background intensities at each medial 
    axis pixel. The function returns a list of lists. Each element of the list 
    corresponds to a medial axis (a graph edge), and each sublist contains a set 
    of dictionaries for each medial axis pixel. 
    
    The items of the dictionary are

    'int_vessel_m'  # Avg vessel intensity
    'int_back_m'    # Avg background intensity
    'diff_p':       # Intensity difference
    'diff_norm_p':  # Normalized intensity difference (LVS index before smoothing)
    """

    section_stats = []   
    for idx, segment in enumerate(section_data):
        int_vessel_m_prv = 0
        int_back_m_prv = 0
        section_stats_path =[]     
        for pix in segment:
            backg = pix.get('int_back')
            intves = pix.get('int_vessel')
            
            # Average intensity values. If for some reason there is no vessel
            # or background pixels, the value calculated for the previous point
            # is used
            int_vessel_m = int_vessel_m_prv
            int_back_m = int_back_m_prv
            if len(intves) != 0:
                int_vessel_m = np.mean(intves)
                
            if len(backg) != 0:
                int_back_m =  np.mean(backg)

            # Intensity difference between vessel and background
            diff_p = int_vessel_m-int_back_m
            
            # Normalized difference
            diff_norm_p = (int_vessel_m-int_back_m)/max([int_vessel_m,int_back_m])

            section_stats_path.append({
                'int_vessel_m': int_vessel_m,   # Avg vessel intensity
                'int_back_m': int_back_m,       # Avg background intensity
                'diff_p': diff_p,               # Intensity difference
                'diff_norm_p': diff_norm_p      # Normalized intensity difference
            })

            int_vessel_m_prv = int_vessel_m
            int_back_m_prv = int_back_m
            
        section_stats.append(section_stats_path)

    return section_stats
  
def smooth_values(section_stats, n=3):
    """Smooth LVS values along each vessel segment.
    
    Creates a new dictionary item in `section_stats` with key 'diff_norm_p_mean'
    """

    total_r = []
    for idx_path, section_stats_path in enumerate(section_stats):
        for idx_pix, _ in enumerate(section_stats_path):
            idx_ini = max([0, idx_pix-n])
            
            section_stats_nei = section_stats_path[idx_ini:idx_pix+n+1]
            diff_nei = [item['diff_norm_p'] for item in section_stats_nei]
            diff_m = np.mean(diff_nei)
            total_r.append(section_stats[idx_path][idx_pix]['diff_norm_p'])
            section_stats[idx_path][idx_pix]['diff_norm_p_mean']=diff_m

    section_stats = normalize(section_stats, np.nanmean(total_r), np.nanmean(total_r))

    return section_stats  

def normalize(section_stats, total_mean, total_std):
    """Normalizes LVS values by the mean and deviation of the values for all pixels
    in an image. Not used in the experiments."""

    for idx_path, section_stats_path in enumerate(section_stats):
        for idx_pix, _ in enumerate(section_stats_path):  
            dict_data = section_stats[idx_path][idx_pix]      
            value = dict_data['diff_norm_p']
            dict_data['diff_norm_p_div_mean'] = value/total_mean
            dict_data['diff_norm_p_div_std'] = (value-total_mean)/total_std

    return section_stats 

def expand_values(graph, section_stats_s, img_gray, img_bin):
    """Expands LVS values calculated for medial axis pixels to all pixels in the
    vessel."""

    obj_listas = list(graph.edges(data=True))
    img_lvs_skel = np.zeros_like(img_gray, dtype=float) 
    img_skel = np.zeros_like(img_gray, dtype=float) 
    for item, stats_path in zip(obj_listas, section_stats_s):

        path = item[2]['path']
        for p,pdata in zip(path, stats_path):
            img_lvs_skel[p] = pdata['diff_norm_p_mean']
            img_skel[p] = 1

    _, (img_inds_r, img_inds_c) = ndi.distance_transform_edt(img_skel==0, return_indices=True)
    img_lvs = img_lvs_skel[img_inds_r, img_inds_c]
    img_lvs = img_lvs*(img_bin>0)

    return img_lvs
    
def lvs(img_gray, img_bin, radius, k=50):
    """Calculate the Local vessel salience (LVS) of an image.

    Args:
        img_gray: original image
        img_bin: binary image containing vessel annotations
        radius: radius of the background region to consider around contour pixels
        (parameter r_b of the paper)
        k: number of contour points to search around each central point. Larger
        values guarantee that contour points will be found, but might be slower.

    Returns:
        An image containing the LVS value for each pixel
    """

    if img_bin.max()==255:
        img_bin = img_bin//255
    
    graph = get_graph(img_bin)
    contour = get_contour(img_bin)
    closest_points = get_closest_points(graph, contour, k)
    section_data = get_intensities(img_gray, img_bin, graph, contour, closest_points, radius)

    section_stats = get_statistics(section_data)
    section_stats_s = smooth_values(section_stats, n=15)

    img_lvs = expand_values(graph, section_stats_s, img_gray, img_bin)

    return img_lvs  

def ls_recall(img_lvs, img_bin, pred, threshold):
    """Calculates the low-salience recall (LSRecall).

    Args:
        img_lvs: image containing LVS values
        img_bin: binary image containing vessel annotations
        pred: binary image containing predictions of an algorithm
        threshold: salience threshold

    Returns:
        The LSRecall value
    """

    # Low-salience pixels
    img_hard = (img_lvs <= threshold) & (img_bin > 0)
    if img_hard.sum() == 0:
        # No vessel pixels
        return 0
        
    # Recovered low-salience pixels
    pred_hard = pred[img_hard>0]
        
    # Recall
    recall = pred_hard.sum()/pred_hard.size

    return recall

#### Auxiliary functions for visualization ####

def plot_sections(img_bin, graph, contour, closest_points):

    img_graph = graph_to_img(graph, img_shape=img_bin.shape, node_color=(1, 1, 1), 
                               node_pixels_color=(0, 0, 0), edge_color=(1, 1, 1))
    img_graph = img_graph[:,:,0]

    paths = [path[2] for path in graph.edges(data='path')]

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    plt.figure()
    plt.subplot(111)
    plt.imshow(img_bin+img_graph, 'gray')
    for path, closest_points_path in zip(paths, closest_points):
        for point_idx in range(len(path)):
            p = path[point_idx]
            nei1_idx, nei2_idx = closest_points_path[point_idx]
            n1 = contour[nei1_idx]
            plt.plot([p[1], n1[1]], [p[0], n1[0]], '-o', ms=1, color=colors[point_idx%10])
            if nei2_idx is not None:
                n2 = contour[nei2_idx]
                plt.plot([p[1], n2[1]], [p[0], n2[0]], '-o', ms=1, color=colors[point_idx%10])

def plot_sampling_regions(section_data, img_bin, graph, n=20):

    colors1 = ['C0', 'C1', 'C2', 'C3']
    colors2 = ['C4', 'C5', 'C6', 'C8']

    img_graph = graph_to_img(graph, img_shape=img_bin.shape, node_color=(1, 1, 1), 
                               node_pixels_color=(0, 0, 0), edge_color=(1, 1, 1))
    img_graph = img_graph[:,:,0]

    plt.figure(figsize=(10,10))
    plt.subplot(111)
    plt.imshow(img_bin+img_graph, 'gray')

    for _ in range(n):
        seg_idx = np.random.randint(0, len(section_data))
        section_data_path = section_data[seg_idx]
        pix_idx = np.random.randint(0, len(section_data_path))
        pix_data = section_data_path[pix_idx]

        pv = pix_data['pix_vessel']
        pb = pix_data['pix_back']
        y, x = zip(*pv)
        idx_c = np.random.randint(0, 4)
        plt.scatter(x, y, s=2, color=colors1[idx_c])
        y, x = zip(*pb)
        plt.scatter(x, y, s=2, color=colors2[idx_c])
