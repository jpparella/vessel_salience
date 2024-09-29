"""Script for augmenting the salience of blood vessels
"""

import random
import numpy as np
import scipy.ndimage as ndi
import skimage.morphology
import skimage.draw
import cv2
from .pyvane.graph.creation import create_graph as create_graph_pv
from .pyvane.graph import adjustment as net_adjust
from .pyvane.image import Image


def extract_full_length(graph):
    """Extract edge segments length."""

    graph_edges = list(graph.edges(data=True))

    segment_length = []
    for graph_edge in graph_edges:
        edge_path = graph_edge[2]['path']
        length = get_path_length([edge_path[0], edge_path[-1]])
        segment_length.append(length)
    min_v =  min(segment_length)
    max_v =  max(segment_length)

    return segment_length, min_v, max_v
        
def dist_p(points, p):
    """Distance between each point in `points` and a given point `p`."""

    return np.sqrt(np.sum((points - p)**2, axis=1))

def get_attenuation_coefficient(points, p_min_1_idx, p_min_2_idx):
    """Calculates the distance between all points in points[:p_min_1_idx] to
    points[p_min_1_idx] and all points in points[p_min_2_idx+1:] to 
    points[p_min_2_idx]. The distance for points points[p_min_1_idx:p_min_2_idx+1]
    is set to zero.

    Returns an array with the values concatenated. This represents an attenuation 
    coefficient between points points[0] and points[-1].
    """

    path_length = get_path_length_cum(points)
    first_seg = path_length[p_min_1_idx] - path_length[:p_min_1_idx]
    first_seg /= first_seg[0]
    second_seg = path_length[p_min_2_idx+1:] - path_length[p_min_2_idx]
    second_seg /= second_seg[-1]

    dist_vec = np.concatenate((first_seg, 
                               np.zeros(p_min_2_idx-p_min_1_idx+1), 
                               second_seg))

    return dist_vec

def get_path_length(path):
    """Get the arc-length of a path."""

    dpath = np.diff(path, axis=0)
    dlengths = np.sqrt(np.sum(dpath**2, axis=1))
    path_length = np.sum(dlengths)

    return path_length
    
def get_path_length_cum(path):
    """Get cumulative distance of each point in `path` to the firs point in the
    list."""

    dpath = np.diff(path, axis = 0)
    dlengths = np.sqrt(np.sum(dpath**2, axis=1))
    path_length = np.cumsum(dlengths)
    path_length = np.array([0]+path_length.tolist())

    return path_length

def point_from_dist(points, index_pc, dist):
    """Return the indices of the two points that are a distance `dist`
    from point `index_pc` in the list of points `points`."""

    res = get_path_length_cum(points)
    distpc = res[index_pc]
    distp1 = distpc - dist
    distp2 = distpc + dist
    indexp1 = np.argmin(np.abs(res-distp1))
    indexp2 = np.argmin(np.abs(res-distp2))

    return indexp1, indexp2
    
def get_segments(graph, img_origin, img_label):
    """Create image in which each vessel segment has an associated id. The ids
    are the same edges indices as in `graph`."""

    graph_edges = list(graph.edges(data=True))
    img_graph = np.zeros_like(img_origin, dtype=int) - 1
    for idx, item in enumerate(graph_edges):
        path = item[2]['path']
        for p in path:
            img_graph[p] = idx

    _, (img_inds_r, img_inds_c) = ndi.distance_transform_edt(img_graph==-1, 
                                                             return_indices=True)
    img_exp = img_graph[img_inds_r, img_inds_c]
    img_exp = img_exp*img_label

    return img_exp

def get_valid_pixels(graph_edges, cut):
    """Get segment pixels that are long enough to augment."""

    valid_pixels = []
    for edge_idx, graph_edge in enumerate(graph_edges):
        valid_seg_pixels = []

        edge_coords = graph_edge[2]['path']
        path_length = get_path_length_cum(edge_coords)
        for idx, dist in enumerate(path_length):
            if dist > cut and (path_length[-1]-dist) > cut:
                valid_seg_pixels.append(edge_coords[idx])

        if len(valid_seg_pixels)!=0:
            valid_pixels.append([edge_idx, valid_seg_pixels])

    return valid_pixels

def create_graph(label, adjust):
    """Create graph from a binary image."""

    label = np.clip(label, 0, 1)
    img_skel = skimage.morphology.skeletonize(label, method='lee')
    # Convert back to PyVaNe
    data_skel = Image(img_skel)
    graph_vessel = create_graph_pv(data_skel)
    if adjust:
        graph_vessel = net_adjust.adjust_graph(graph_vessel, 0)

    return graph_vessel

def get_crop(img, point, rqi_len):
    """Crop image around point."""

    nr, nc = img.shape
    pr, pc = int(point[0]), int(point[1])

    ri = max([0, pr - rqi_len])
    re = min([nr, pr + rqi_len])
    ci = max([0, pc - rqi_len])
    ce = min([nc, pc + rqi_len])
    img_crop = img[ri:re,ci:ce]

    return img_crop

def crop_alter(img, img2, point, rqi_len):  
    """Replace region of `img` around `point` by `img2`."""

    nr, nc = img.shape
    pr, pc = int(point[0]), int(point[1])

    ri = max([0, pr - rqi_len])
    re = min([nr, pr + rqi_len])
    ci = max([0, pc - rqi_len])
    ce = min([nc, pc + rqi_len])

    img[ri:re,ci:ce] = img2

    return img

def neighbors(point, shape):
    """8-neighbors of a pixel."""

    shifts = ((-1, -1), (-1,  0), (-1, +1), (0,  -1), 
              (0,  +1), (+1, -1), (+1, 0),  (+1, +1))

    neis = []
    for s in shifts:
        r = point[0]+s[0]
        c = point[1]+s[1]
        if 0<=r<shape[0] and 0<=c<shape[1]:
            neis.append((r, c))

    return neis

def expand_line(img_skel, img_skel_aug, img_origin, img_label, img_seg, 
                back_threshold):
    """Expand skeleton augmented image to the whole segment.

    Args:
        img_skel: Skeleton image
        img_skel_aug: Skeleton image with attenuation coefficient
        img_origin: Original vessel image
        img_label: Vessel label image
        img_seg: Label image containing only the segment to be augmented
        back_threshold: Threshold for accepting background region for
        replacement.
    """

    lbl_int = np.max(img_label)

    img_origin = img_origin.astype(float)
    img_skel = img_skel == 128

    ret = ndi.distance_transform_edt(np.logical_not(img_skel),
                                     return_indices=True)
    _, (img_inds_r, img_inds_c) = ret

    back_int = np.mean(img_origin[img_label!=lbl_int])
    img_origin_norm = img_origin - back_int

    # Expand skeleton attenuation coefficients to whole image
    img_exp = img_skel_aug[img_inds_r, img_inds_c]
    # Set attenuation to 1 outside segment
    img_exp[img_seg==0] = 1.

    # Apply attenuation
    img_aug = img_origin_norm * img_exp
    
    # Binary image where background is True
    img_label_inv = img_label!=lbl_int
    
    # Image with skeleton pixels that will have minimum value (discontinuity)
    img_sat = (img_exp==0) & (img_seg>0)

    img_aug = img_aug + back_int
    img_aug = np.round(img_aug).astype(np.uint8)
               
    # Crop on discontinuity region
    inds = np.nonzero(img_sat)    
    min_r, min_c = min(inds[0]), min(inds[1])
    max_r, max_c = max(inds[0]), max(inds[1])
    img_sat_crop = img_sat[min_r:max_r+1, min_c:max_c+1]

    # Coordinates of discontinuity region
    sat_coords = np.where(img_sat == 1)
    # Coordinates with respect to the central point
    p_center =  [(img_sat_crop.shape[0]-1)//2, (img_sat_crop.shape[1]-1)//2]
    sat_coords_norm = np.where(img_sat_crop == 1)
    sat_coords_norm = (sat_coords_norm[0] - p_center[0], sat_coords_norm[1] - p_center[1]) 

    # Use convolution to find valid regions for candidate background for discontinuity
    conv_output = ndi.convolve(img_label_inv.astype(int), img_sat_crop[::-1,::-1].astype(int), mode = 'constant')
    conv_output_valid = conv_output == img_sat_crop.sum()
    coords_valid_back = np.where(conv_output_valid == 1)
    coords_valid_back = list(zip(*coords_valid_back))

    # Dilation for obtaining discontinuity border
    img_sat_dil = ndi.binary_dilation(img_sat, iterations=1)
    img_sat_dil = img_sat_dil.astype(np.uint8)

    contour, _ = cv2.findContours(img_sat_dil,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    contour = np.array([p[0] for p in contour[0]])

    # Pixels that are the external border of the saturation region
    sat_ext_cont = []
    for p in contour:   
        p_inv = (p[1], p[0])    
        if img_label[p_inv] == 0:
            sat_ext_cont.append(p_inv)
    sat_ext_cont = np.array(sat_ext_cont)

    mean_diffs = []
    while True:
        # Search background regions to put on discontinuity
        img_only_back = np.zeros_like(img_aug)
        sat_coords_on_back = None
        if len(coords_valid_back) == 0:
            print('Unable to find a valid background for RQI')            
            break

        # Center of candidate region
        idx_coord = random.randint(0, len(coords_valid_back)-1)
        pc_back = coords_valid_back[idx_coord]            
        coords_valid_back.remove(pc_back)

        sat_coords_on_back = (sat_coords_norm[0] + pc_back[0], sat_coords_norm[1] + pc_back[1])  
        
        # Get candidate background region
        img_only_back[sat_coords] = img_aug[sat_coords_on_back] 

        # Difference between the outside border of the discontinuity region and the
        # inside border of the candidate background region
        int_diff = []
        for coord in sat_ext_cont:
            vizi = neighbors(coord, img_sat.shape)
            for pt in vizi:
                if img_sat[pt[0]][pt[1]] == 1:
                    value = int(img_only_back[pt[0]][pt[1]]) - int(img_aug[coord[0]][coord[1]])
                    int_diff.append(abs(value))

        mean_diff = np.mean(int_diff)
        mean_diffs.append(mean_diff)
        # Check if difference in intensities is smaller than the threshold
        if mean_diff > back_threshold:     
            continue
        else:
            # Put background region into the discontinuity region         
            img_aug[sat_coords] = img_aug[sat_coords_on_back] 
            break
    
    debug = (img_exp, img_sat, conv_output_valid, img_only_back, sat_coords, 
            sat_coords_on_back, sat_ext_cont)

    return img_aug, debug

def create_image(img_origin, img_label, rqi_len_interv, min_len_interv, 
                 n_rqi_interv, back_threshold, rng_seed=None, 
                 highlight_center=False):
    """Augment image using the method proposed in the paper.

    Args:
        img_origin: original blood vessel image
        img_label: binary image containing vessel annotations
        rqi_len_interv: range of possible lengths for augmentation (parameter l in the paper)
        min_len_interv: range of possible lengths for the discontinuity region (parameter l_d in the paper)
        n_rqi_interv: number of segments to augment
        back_threshold: similarity threshold for searching a valid background for the discontinuity
        rng_seed: seed of the random number generator for reproducible results
        highlight_center: if true, the central point of the augmentation is highlighted for easier visualization

    Returns:
        img_aug: the augmented image
        debug_full: a list of relevant variables used for the main steps of the method
        graph: the graph created from the annotation image
    """

    if img_label.max()==255:
        img_label = img_label//255

    graph = create_graph(img_label, True)

    if rng_seed is not None:
        random.seed(rng_seed)

    n_rqi = random.randint(n_rqi_interv[0],n_rqi_interv[1])
    graph_edges = list(graph.edges(data=True))
    img_segs = get_segments(graph, img_origin, img_label)

    img_aug = img_origin.copy()

    debug_full = []
    edges_drawn = []
    for i_rqi in range(n_rqi):

        rqi_len = random.randint(rqi_len_interv[0], rqi_len_interv[1])
        max_min_len = min([min_len_interv[1], rqi_len-5])
        min_len = random.randint(min_len_interv[0], max_min_len)

        # Get segment and central point to augment
        valid_edges_pixels = get_valid_pixels(graph_edges, rqi_len//2)

        valid_edge_index = random.randint(0, len(valid_edges_pixels)-1)
        if valid_edge_index in edges_drawn:
            continue

        edges_drawn.append(valid_edge_index)
        valid_edge_path = valid_edges_pixels[valid_edge_index][1]
        pc_index_v = random.randint(0, len(valid_edge_path)-1)

        # Central point of an augmented segment
        pc = valid_edge_path[pc_index_v]
        edge_index = valid_edges_pixels[valid_edge_index][0]
        edge_path = graph_edges[edge_index][2]['path']
        # Image containing label of segment to be processed
        img_seg = img_segs==edge_index

        # Initial and final points of augmented region
        pc_idx = edge_path.index(pc)
        p1_idx , p2_idx = point_from_dist(edge_path, pc_idx, rqi_len//2)
        p1 = edge_path[p1_idx]
        p2 = edge_path[p2_idx]

        skel_aug_points = edge_path[p1_idx:p2_idx+1]

        img_skel = np.zeros(img_label.shape, dtype=np.uint8)
        for point in skel_aug_points:
            img_skel[point] = 128

        img_aug_crop = get_crop(img_aug, pc, rqi_len).copy()
        img_label_crop = get_crop(img_label, pc, rqi_len)
        img_seg_crop = get_crop(img_seg, pc, rqi_len)

        # Initial and final points of minimum intensity region
        p_min_1_idx, p_min_2_idx = point_from_dist(edge_path, pc_idx,
                                                   min_len//2)
        p_min_1 = edge_path[p_min_1_idx]
        p_min_2 = edge_path[p_min_2_idx]

        attenuation_coeff = get_attenuation_coefficient(skel_aug_points,
                                            p_min_1_idx-p1_idx,
                                            p_min_2_idx-p1_idx)

        # For debugging
        freq_vaso = ([])
        for cord in edge_path:        
            freq_vaso.append(img_aug[cord[0]][cord[1]])
        vessel_int_plot = freq_vaso
        #----
        
        img_skel_aug = np.zeros(img_skel.shape, dtype=float)
        for idx, (r, c) in enumerate(skel_aug_points):
            img_skel_aug[r, c] = attenuation_coeff[idx]
            
        img_skel_crop = get_crop(img_skel, pc, rqi_len)
        img_skel_aug_crop = get_crop(img_skel_aug, pc, rqi_len)
        
        img_aug_crop_new, debug_expand = expand_line(img_skel_crop, img_skel_aug_crop, 
                                                img_aug_crop, 
                                                img_label_crop, img_seg_crop,
                                                back_threshold)
        
        img_aug = crop_alter(img_aug, img_aug_crop_new, pc, rqi_len)
    
        # For debugging
        if highlight_center:
            coords = neighbors(pc, img_aug.shape)
            for cord in coords:            
                img_aug[cord[0]][cord[1]] = 255
        
        freq_vaso = ([])
        for cord in edge_path:        
            freq_vaso.append(img_aug[cord[0]][cord[1]])
        vessel_int_new = freq_vaso
    
        debug = (img_aug_crop, img_label_crop, img_skel_crop, img_skel_aug_crop, 
                 img_seg_crop, 
                 rqi_len, attenuation_coeff, vessel_int_plot, vessel_int_new, 
                 edge_path, pc_idx, p1_idx, p2_idx, p_min_1_idx, p_min_2_idx, 
                 debug_expand,img_aug_crop_new)
        debug_full.append(debug)
        #------

    return img_aug, debug_full, graph
 