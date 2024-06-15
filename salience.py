"""Script for calculating the local vessel salience (LVS) and the low salience
recal (LSRecall)."""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import skeletonize
from scipy.spatial import KDTree
import scipy.ndimage as ndi
import cv2
import shapely
from skimage import draw
from pyvane.graph.creation import create_graph
from pyvane.graph import adjustment as net_adjust
from pyvane.image import Image as Image_pv
from pyvane.util import graph_to_img


def load_data(bin_path, img_path):
    """Load some test data."""

    img_bin = np.array(Image.open(bin_path),dtype=np.uint8)
    img_gray = plt.imread(img_path)
    # print(np.unique(img_bin))
    img_skel = skeletonize(img_bin)
    img_skel_pv = Image_pv(img_skel)
    graph = create_graph(img_skel_pv)
    graph_simple = net_adjust.simplify(graph)
    graph_final = net_adjust.adjust_graph(graph_simple, length_threshold=5)

    img_graph = graph_to_img(graph_final, img_shape=img_bin.shape, node_color=(1, 1, 1), node_pixels_color=(0, 0, 0),
                            edge_color=(1, 1, 1))

    img_graph = img_graph[:,:,0]

    contours, _ = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Remove additional dimension from each contour and invert x and y. 
    # contour[i] contains a pixel at position (row, col)
    contour = np.concatenate([contour[:,0,::-1] for contour in contours])

    return img_bin,img_gray, img_skel, graph_final, img_graph, contour

def load_data_iou(bin_path, img_path):
    """Load some test data."""

    img_bin = np.array(Image.open(bin_path),dtype=np.uint8)
    img_gray = plt.imread(img_path)
    # print(np.unique(img_bin))
    img_skel = skeletonize(img_bin)
    img_skel_pv = Image_pv(img_skel)
    graph = create_graph(img_skel_pv)
    graph_simple = net_adjust.simplify(graph)
    graph_final = net_adjust.adjust_graph(graph_simple, length_threshold=5)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Remove additional dimension from each contour and invert x and y. 
    # contour[i] contains a pixel at position (row, col)
    contour = np.concatenate([contour[:,0,::-1] for contour in contours])

    return img_bin,img_gray, graph_final, contour

def get_closest_points(graph, contour, k=20,showWarnings = False):
    """For each skeleton point, get two points in the contour that are closest to the skeleton point.
    The two contour points are constrained to be on opposite sides of the skeleton.
    Parameter `k` controls the number of contour points that are analysed. Larger values lead to
    higher computational cost, but if k is too low, the second contour point might not be found."""

    # kdtree of the contour
    kdtree = KDTree(contour)
    # Flatten list of skeleton pixels
    paths = [path[2] for path in graph.edges(data='path')]
    pixels = [p for path in paths for p in path]

    point_map = {}
    for point_idx, point in enumerate(pixels):

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

        '''if idx==k:
            print('Warning, parameter k is not large enough')

        point_map[point_idx] = (nei1_idx, nei2_idx)'''

        if idx==k:
            if (showWarnings):
                print('Warning, parameter k is not large enough')
            point_map[point_idx] = (nei1_idx, None)
        else:
            point_map[point_idx] = (nei1_idx, nei2_idx)

    return point_map

def get_closest_points2(graph, contour, k=20,showWarnings= False):
    """For each skeleton point, get two points in the contour that are closest to the skeleton point.
    The two contour points are constrained to be on opposite sides of the skeleton.
    Parameter `k` controls the number of contour points that are analysed. Larger values lead to
    higher computational cost, but if k is too low, the second contour point might not be found."""

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
    
            '''if idx==k:
                print('Warning, parameter k is not large enough')
    
            point_map[point_idx] = (nei1_idx, nei2_idx)'''
    
            if idx==k:
                if (showWarnings):
                    print('Warning, parameter k is not large enough')
                point_map_path.append( (nei1_idx, None))
            else:
                point_map_path.append(( nei1_idx, nei2_idx))
        
        point_map.append(point_map_path)
    return point_map

def get_intensities(img,img_bin, graph, contour, point_map, radius=5):
    """Get image intensities for each point in `point_map`"""

    paths = [path[2] for path in graph.edges(data='path')]
    pixels = [p for path in paths for p in path]

    int_dict = {}
    for point_idx, (nei1_idx, nei2_idx) in point_map.items():
        p = pixels[point_idx]
        n1 = contour[nei1_idx]
        # Vessel and background positions for the nearest point
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

        int_dict[point_idx] = {'int_vessel':int_vessel, 'pix_vessel':list(zip(rrv, ccv)),
                               'int_back':int_background, 'pix_back':list(zip(rrb, ccb))}

    return int_dict

def get_intensities2(img,img_bin, graph, contour, point_map, radius=5):
    """Get image intensities for each point in `point_map`"""

    paths = [path[2] for path in graph.edges(data='path')]
    pixels = [p for path in paths for p in path]

    int_dict = []
    #print('point_map')
    #print(point_map)
    for path,point_map_path in zip(paths,point_map):
        int_dict_path =[]     
        #print('path')
        #print(path)
        #print('point_map_path')
        #print(point_map_path)
        for p, (nei1_idx, nei2_idx) in zip(path,point_map_path):
            
            n1 = contour[nei1_idx]
            # Vessel and background positions for the nearest point
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
    
            int_dict_path.append({'int_vessel':int_vessel, 'pix_vessel':list(zip(rrv, ccv)),
                                   'int_back':int_background, 'pix_back':list(zip(rrb, ccb))})
        #print(int_dict_path)
        int_dict.append(int_dict_path)
    return int_dict

def plot_sections(graph, img_bin, img_graph, contour, point_map):

    paths = [path[2] for path in graph.edges(data='path')]
    pixels = [p for path in paths for p in path]

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    plt.figure()
    plt.subplot(111)
    plt.imshow(img_bin+img_graph, 'gray')
    for point_idx, (nei1_idx, nei2_idx) in point_map.items():
        p = pixels[point_idx]
        n1 = contour[nei1_idx]
        plt.plot([p[1], n1[1]], [p[0], n1[0]], '-o', ms=1, color=colors[point_idx%10])
        if nei2_idx is not None:
            n2 = contour[nei2_idx]
            plt.plot([p[1], n2[1]], [p[0], n2[0]], '-o', ms=1, color=colors[point_idx%10])

def plot_sampling_regions(int_dict, img_bin, img_graph, n=20):

    colors1 = ['C0', 'C1', 'C2', 'C3']
    colors2 = ['C4', 'C5', 'C6', 'C8']

    indices = np.random.randint(0, len(int_dict), size=n)

    plt.figure()
    plt.subplot(111)
    plt.imshow(img_bin+img_graph, 'gray')

    for idx in indices:
        data = int_dict[idx]
        # print(data[0]['pix_vessel'])
        pv = data[0]['pix_vessel']
        pb = data[0]['pix_back']
        y, x = zip(*pv)
        idx_c = np.random.randint(0, 4)
        plt.scatter(x, y, s=2, color=colors1[idx_c])
        y, x = zip(*pb)
        plt.scatter(x, y, s=2, color=colors2[idx_c])

def interpret(int_dict):

    ret_dict = {}
    dct_c = int_dict.copy()
    diff = []
    diff_norm = []
    
    while len(dct_c) > 0:
        item = dct_c.popitem()
        int_vessel_m = np.mean(item[1].get('int_vessel'))
        int_back_m =  np.mean(item[1].get('int_back'))
        diff_p = int_vessel_m-int_back_m
        diff_norm_p = (int_vessel_m-int_back_m)/max([int_vessel_m,int_back_m])
        ret_dict[item[0]] = {'int_vessel_m':int_vessel_m, 'int_back_m':int_back_m,
                               'diff_p':diff_p, 'diff_norm_p':diff_norm_p}
        diff.append(diff_p)
        diff_norm.append(diff_norm_p)
    #print(diff)
    #print(diff_norm)
    diff_mean = np.mean(diff)
    diff_min = np.min(diff)
    diff_norm_mean = np.mean(diff_norm)
    diff_norm_min = np.min(diff_norm)

    return ret_dict,diff_mean,diff_min,diff_norm_mean,diff_norm_min

def interpret2(int_dict):

    ret_dict = []
    dct_c = int_dict.copy()
    diff = []
    diff_norm = []
    
    for idx,segment in enumerate(int_dict):
        int_vessel_m_prv = 0
        int_back_m_prv = 0
        int_dict_path =[]     
        #print(segment)
        for pix in segment:
            backg = pix.get('int_back')
            intves = pix.get('int_vessel')
            #print(pix.get('int_vessel'))
            int_vessel_m = int_vessel_m_prv
            int_back_m = int_back_m_prv
            
            if len(intves) != 0:                
                int_vessel_m = np.mean(intves)
                
            if len(backg) != 0:                
                int_back_m =  np.mean(backg)

            
            diff_p = int_vessel_m-int_back_m
            
            diff_norm_p = (int_vessel_m-int_back_m)/max([int_vessel_m,int_back_m])
            # print(int_vessel_m,int_back_m)
            int_dict_path.append( {'int_vessel_m':int_vessel_m, 'int_back_m':int_back_m,
                                   'diff_p':diff_p, 'diff_norm_p':diff_norm_p})
            diff.append(diff_p)
            diff_norm.append(diff_norm_p)

            int_vessel_m_prv= int_vessel_m
            int_back_m_prv = int_back_m
            
        ret_dict.append(int_dict_path)
    # if diff is None:
    #     print(diff)
    diff_mean = np.mean(diff)
    diff_min = np.min(diff)

    diff_norm_mean = np.mean(diff_norm)
    diff_norm_min = np.min(diff_norm)

    return ret_dict,diff_mean,diff_min,diff_norm_mean,diff_norm_min
  
def image_process(img_bin,img_ori,graph,ret_dict,limiar):

    dct_c = ret_dict.copy()
    teste = np.zeros(img_bin.shape)

    paths = [path[2] for path in graph.edges(data='path')]
    pixels = [p for path in paths for p in path]

    

    for ind,diff in dct_c.items():
        pixel = pixels[ind]
        if (diff['diff_norm_p']<= limiar):
            teste[pixel] = 1

    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.imshow(teste+img_ori/255, 'gray')
    plt.subplot(122)
    plt.imshow(img_ori, 'gray',vmax = 30)
    
    # while len(dct_c) > 0:
    #     item = dct_c.popitem()
        
    #     diff_p = item[1].get('diff_norm_p')#diff_norm_p#diff_p
    #     pixel = pixels[item[0]]
    #     #print("diff - ",diff_p," idx - ", item[0])
    #     if (diff_p <= limiar):
    #         plt.scatter(pixel[1], pixel[0], s=2, color='C0')

def image_process2(img_bin,img_ori,graph,ret_dict,limiar,plot = False):

    dct_c = ret_dict.copy()
    teste = np.zeros(img_bin.shape)

    paths = [path[2] for path in graph.edges(data='path')]
    pixels = [p for path in paths for p in path]

    

    for idx_path, int_dict_path in enumerate(ret_dict):
        for idx_pix, data_px in enumerate(int_dict_path):
            pixel = paths[idx_path][idx_pix]
            if (data_px['diff_norm_p_mean']<= limiar):
                teste[pixel] = 1
    if (plot):
        plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.imshow((teste*255)+img_ori, 'gray')
        plt.subplot(122)
        plt.imshow(img_ori, 'gray')#,vmax = 30

    return (teste*255)+img_ori
    
    # while len(dct_c) > 0:
    #     item = dct_c.popitem()
        
    #     diff_p = item[1].get('diff_norm_p')#diff_norm_p#diff_p
    #     pixel = pixels[item[0]]
    #     #print("diff - ",diff_p," idx - ", item[0])
    #     if (diff_p <= limiar):
    #         plt.scatter(pixel[1], pixel[0], s=2, color='C0')

def get_segments(graph,ret_dict, img_origin, img_label):
    """Cria imagem na qual cada segmento de vaso possui um id diferente. Útil para isolar
    apenas o vaso que está sendo processado. Depois disso é calculado o valor de dificuldade para cada pixel do vaso em questão"""

    valid = 0
    obj_listas = list(graph.edges(data=True))
    img_graph = np.zeros_like(img_origin, dtype=float) 
    img_skel = np.zeros_like(img_origin, dtype=float) 
    for item,int_dict_path in zip(obj_listas,ret_dict):

        path = item[2]['path']
        for p,pdata in zip(path,int_dict_path):
            img_graph[p] = pdata['diff_norm_p_mean']
            img_skel[p] = 1
            if valid<10:
                #print(img_graph[p])
                valid = valid +1
        valid = 0
    
    # print(np.min(img_graph),np.max(img_graph))
    _, (img_inds_r, img_inds_c) = ndi.distance_transform_edt(img_skel==0, return_indices=True)
    img_exp = img_graph[img_inds_r, img_inds_c]
    img_exp = img_exp*(img_label>0)
    return img_exp

def difficult_avaliate(img_lv,img_lbl,threshold):
    img_hard = (img_lv <= threshold) & (img_lbl > 0)
    num_px_hard = img_hard.sum()

    return num_px_hard
     
def iou_lcr(img_lv,img_lbl,pred,threshold):

    #print(np.min(img_lv),np.max(img_lv))
    img_hard = (img_lv <= threshold) & (img_lbl > 0)
    if (img_hard.sum() == 0):
        return 0
        
    pred_hard = pred[img_hard>0]
    # print(img_hard)
    # print(pred_hard)
        
    iou = pred_hard.sum()/pred_hard.numel()

    return iou

def iou_lcr_n(img_lv,img_lbl,pred,threshold):

    ious = []
    pxs_hard = []
    for thr in threshold:        
        img_hard = (img_lv <= thr) & (img_lbl > 0)
        pred_hard = pred[img_hard>0]
        iou = pred_hard.sum()/pred_hard.numel()
        num_px_hard = img_hard.sum()
        ious.append(iou)
        pxs_hard.append(num_px_hard)

    return ious,pxs_hard

def process_means(int_dict,n=3):

    total_r=[];
    
    for idx_path, int_dict_path in enumerate(int_dict):
        for idx_pix, _ in enumerate(int_dict_path):
            idx_ini = max([0,idx_pix-n])
            
            int_dict_nei = int_dict_path[idx_ini:idx_pix+n+1]            
            diff_nei = [item['diff_norm_p'] for item in int_dict_nei]
            diff_m = np.mean(diff_nei)
            total_r.append(int_dict[idx_path][idx_pix]['diff_norm_p'])
            int_dict[idx_path][idx_pix]['diff_norm_p_mean']=diff_m
    # teste = np.mean(total_r)
    # print(teste)
    # teste = np.std(total_r) 
    # print(teste)
    int_dict = process_means_2(int_dict,np.nanmean(total_r),np.nanmean(total_r))
    return int_dict;    

def process_means_2(int_dict,total_mean,total_std):    

    for idx_path, int_dict_path in enumerate(int_dict):
        for idx_pix, _ in enumerate(int_dict_path):        
            int_dict[idx_path][idx_pix]['diff_norm_p_div_mean']=int_dict[idx_path][idx_pix]['diff_norm_p']/total_mean
            int_dict[idx_path][idx_pix]['diff_norm_p_div_std']=(int_dict[idx_path][idx_pix]['diff_norm_p']-total_mean)/total_std

    return int_dict;    

def prepare_hist(int_dict):

    hist_values_norm_p = []
    hist_values_div_mean = []
    for idx_path, int_dict_path in enumerate(int_dict):        
        for idx_pix, _ in enumerate(int_dict_path):     
            hist_values_norm_p.append(int_dict[idx_path][idx_pix]['diff_norm_p'])
            hist_values_div_mean.append(int_dict[idx_path][idx_pix]['diff_norm_p_div_std'])
            
    return hist_values_norm_p,hist_values_div_mean;    
    
def full_process_unique(bin_path, img_path,k,radius,limiar):

    img_bin,img_gray, img_skel, graph, img_graph, contour = load_data(bin_path,img_path)
    # k = 50          # Number of contour points to analyse for each skeleton pixel
    # radius = 4      # Radius of the background region to get for each skeleton pixel
    point_map_n = get_closest_points2(graph, contour, k)
    int_dict_n = get_intensities2(img_gray,img_bin, graph, contour, point_map_n, radius)

    ret_dict,diff_mean,diff_min,diff_norm_mean,diff_norm_min = interpret2(int_dict_n)
    ret_dict_n = process_means(ret_dict,n=15)
    image_process2(img_bin,img_gray,graph,ret_dict_n,limiar)
    img_lv = get_segments(graph,ret_dict_n, img_gray, img_bin)

    return img_lv
            
def full_process_iou_unique(bin_path, img_path,k,radius,limiar,pred):

    img_bin,img_gray,  graph,  contour = load_data_iou(bin_path,img_path)
    # k = 50          # Number of contour points to analyse for each skeleton pixel
    # radius = 4      # Radius of the background region to get for each skeleton pixel
    point_map_n = get_closest_points2(graph, contour, k)
    int_dict_n = get_intensities2(img_gray,img_bin, graph, contour, point_map_n, radius)

    ret_dict,diff_mean,diff_min,diff_norm_mean,diff_norm_min = interpret2(int_dict_n)
    ret_dict_n = process_means(ret_dict,n=15)
    # print(int_dict_n)
    
    # image_process2(img_bin,img_gray,graph,ret_dict_n,limiar)

    img_lv = get_segments(graph,ret_dict_n, img_gray, img_bin)
    num_px_hard = difficult_avaliate(img_lv,img_bin,limiar)
    if (num_px_hard == 0):
        iou = 0
    else:
        iou = iou_lcr(img_lv,img_bin,pred,limiar)

    return iou,img_lv,num_px_hard

def full_process_iou(bin_path, img_path,k,radius,limiar,pred):
    
    img_bin,img_gray,  graph,  contour = load_data_iou(bin_path,img_path)
    # k = 50          # Number of contour points to analyse for each skeleton pixel
    # radius = 4      # Radius of the background region to get for each skeleton pixel
    point_map_n = get_closest_points2(graph, contour, k)
    int_dict_n = get_intensities2(img_gray,img_bin, graph, contour, point_map_n, radius)

    ret_dict,diff_mean,diff_min,diff_norm_mean,diff_norm_min = interpret2(int_dict_n)
    ret_dict_n = process_means(ret_dict,n=15)
    # print(int_dict_n)
    

    img_lv = get_segments(graph,ret_dict_n, img_gray, img_bin)

    ious,pxs_hard = iou_lcr_n(img_lv,img_bin,pred,limiar)

    return ious,img_lv,pxs_hard
            