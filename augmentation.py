import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as pilimg
import scipy.ndimage as ndi
import skimage.morphology
import skimage.draw
import cv2
from pyvane.graph.creation import create_graph as create_graph_pv
from pyvane.graph import adjustment as net_adjust
from pyvane.image import Image
from pyvane import util


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

def dist_without_center(points, idxpp1, idxpp2):
    """distance between 2 points to all other points, the region between 
    these two points is returned as 0."""

    path_length = get_path_length_cum(points)
    first_seg = path_length[idxpp1] - path_length[:idxpp1]
    first_seg /= first_seg[0]
    second_seg = path_length[idxpp2+1:] - path_length[idxpp2]
    second_seg /= second_seg[-1]

    dist_vec = np.concatenate((first_seg, np.zeros(idxpp2-idxpp1+1), second_seg))

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

    _, (img_inds_r, img_inds_c) = ndi.distance_transform_edt(img_graph==-1, return_indices=True)
    img_exp = img_graph[img_inds_r, img_inds_c]
    img_exp = img_exp*img_label//255

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
    """funcao de corte da imagem para facilitar processamento da RQI"""

    nr, nc = img.shape
    pr, pc = int(point[0]), int(point[1])

    ri = max([0, pr - rqi_len])
    re = min([nr, pr + rqi_len])
    ci = max([0, pc - rqi_len])
    ce = min([nc, pc + rqi_len])
    img_crop = img[ri:re,ci:ce]

    return img_crop

def crop_alter(img,img2,p1,p2,rqi_len):  
    """funcao de substituição da imagem original pela imagem modificada, é trocado a area cortada da imagem pela nova imagem em tamanho menor"""

    param1 = max([0, p1 -rqi_len])
    param2 = p1 +rqi_len
    param3 = max([0, p2 -rqi_len])
    param4 = p2 +rqi_len

    img[param1:param2,param3:param4] = img2

    return img

def calc_median(img,mask):
    return np.mean(img[mask>0])

def vizinhos(cord_center,shape):
    """retorna vizinhos de um pixel"""

    retorno = []
    
    #-1 -1  0 -1  +1 -1
    #-1 0   0  0  +1 0 
    #-1 +1  0 +1  +1 +1   
    if ((cord_center[0]-1 < shape[0]) and (cord_center[1]-1 < shape[1])):
        retorno.append([cord_center[0]-1,cord_center[1]-1])

    if ((cord_center[0] < shape[0]) and (cord_center[1]-1 < shape[1])):
        retorno.append([cord_center[0],cord_center[1]-1])

    if ((cord_center[0]+1 < shape[0]) and (cord_center[1]-1 < shape[1])):
        retorno.append([cord_center[0]+1,cord_center[1]-1])

    if ((cord_center[0]-1 < shape[0]) and (cord_center[1] < shape[1])):
        retorno.append([cord_center[0]-1,cord_center[1]])

    if ((cord_center[0]+1 < shape[0]) and (cord_center[1] < shape[1])):
        retorno.append([cord_center[0]+1,cord_center[1]])

    if ((cord_center[0]-1 < shape[0]) and (cord_center[1]+1 < shape[1])):
        retorno.append([cord_center[0]-1,cord_center[1]+1])

    if ((cord_center[0]-1 < shape[0]) and (cord_center[1]-1 < shape[1])):
        retorno.append([cord_center[0]-1,cord_center[1]-1])

    if ((cord_center[0]+1 < shape[0]) and (cord_center[1]+1 < shape[1])):
        retorno.append([cord_center[0]+1,cord_center[1]+1])
    
    return retorno

def expand_line(img_line,img_linedyn,img_origin,img_label,img_seg_crop,median_threshold,idx=0):    
    """
    Variáveis:
    img_line: esqueleto da RQI
    img_linedyn: esqueleto da RQI com queda de intensidade normalizada
        Exemplo: [1,0.9,...,0.1,0,0,0,0,0.1,...,0.9,0.1]
    img_origin: imagem dos vasos
    img_label: imagem com rótulos
    Todas as imagens possuem um crop    
    """

    #* Mudei várias coisas nesta função

    #* conversão para float para melhorar a precisão
    img_origin = img_origin.astype(float)  
    #cria imagem binária da imagem do esqueleto da RQI
    img_line = img_line == 128
    #Diltação da imagem de esqueleto da RQI 
    #img_bin = ndi.binary_dilation(img_line, iterations=width)
    #img_exp = np.zeros_like(img_line, dtype=float)        
    #img_bin = img_bin * img_label

    #vesValue = calc_median(img_origin,(img_label == 255))
    
    img_dist, (img_inds_r, img_inds_c) = ndi.distance_transform_edt(np.logical_not(img_line), return_indices=True)
    
    #img_dist = img_dist/np.max(img_dist)

    bckValue = calc_median(img_origin,(img_label != 255))

    img_origin_bck = img_origin - bckValue

    img_exp = img_linedyn[img_inds_r, img_inds_c]
    #* Coloca valor 1 fora da RQI para deixar a imagem inalterada
    img_exp[img_seg_crop==0] = 1.

    #zera os valores de fundo na imagem com queda de intensidade
    #img_ret é a imagem final, então a todo momento que for algo definitivo, vai ser usada ela
    img_ret = img_origin_bck * img_exp
    
    #retorna binário com fundo sendo true e vaso false
    img_label_inv = img_label != np.max(img_label)
    
    #pega as areas da RQI que serao afetadas(pode ser que pegue algo a mais
    #img_ret_loc é uma variavel de controle para saber a real área que vai ser afetada no vaso

    
    img_ret_loc = (img_linedyn[img_inds_r, img_inds_c]==0) & (img_seg_crop>0)

    # plt.imshow(img_seg_crop, cmap='gray')
    # # plt.subplot(1, 2, 2)
    # # plt.imshow(seg.detach().numpy(), cmap='gray')
    # plt.show()
    
    # plt.figure('img_ret_loc'+str(idx),figsize= [10,10])
    # plt.imshow(img_ret_loc,'gray')
    

    img_ret = img_ret + bckValue
    #* Arredonda para evitar saturação dos valores
    img_ret = np.round(img_ret).astype(np.uint8)

    
    #aqui é isolado apenas as areas corretas do vaso que vao ser alteradas(tanto no img_ret, quanto img_ret_loc)
    '''for row in range(img_bin.shape[0]):
        for col in range(img_bin.shape[1]):  
            if img_bin[row,col] == 255:
                img_ret[row,col] = img_ret[row,col]
            else:
                img_ret[row,col] = img_origin[row,col]    
    

    for row in range(img_bin.shape[0]):
        for col in range(img_bin.shape[1]):  
            if img_bin[row,col] == 255:
                img_ret_loc[row,col] = img_ret_loc[row,col]
            else:
                img_ret_loc[row,col] = 0'''
                
    #* Extrai apenas a região de saturação da imagem para que a convolução seja mais intuitiva
   
    inds = np.nonzero(img_ret_loc)
    
    # print(inds)
    # print(np.min(img_ret_loc))
    # print(np.max(img_ret_loc))

    
    min_r, min_c = min(inds[0]), min(inds[1])
    max_r, max_c = max(inds[0]), max(inds[1])
    img_ret_loc_iso = img_ret_loc[min_r:max_r+1, min_c:max_c+1]

    

    coordsLoc = np.where(img_ret_loc == 1)
    # Coordenadas da região de saturação em relação ao centro
    p_center =  [(img_ret_loc_iso.shape[0]-1)//2, (img_ret_loc_iso.shape[1]-1)//2]
    coordsLocNorm = np.where(img_ret_loc_iso == 1)
    coordsLocNorm = (coordsLocNorm[0] - p_center[0], coordsLocNorm[1] - p_center[1]) 

    #faz convolução para poder pegar áreas válidas para uso na área de saturação
    convoleOutput = ndi.convolve(img_label_inv.astype(int), img_ret_loc_iso[::-1,::-1].astype(int), mode = 'constant')
    outPutConv = convoleOutput == img_ret_loc_iso.sum()
    coordsConv = np.where(outPutConv == 1)
    # Pontos possíveis para pegar o fundo
    coordsConv = list(zip(*coordsConv))

    #dilata a area de saturação para poder pegar coordenadas das bordas do vaso
    img_ret_loc_d = ndi.binary_dilation(img_ret_loc, iterations=1)
    img_ret_loc_d = img_ret_loc_d.astype(np.uint8)

    #encontra contornos para poder utilizar as coordendas das bordas do vaso
    contours, hierarchy = cv2.findContours(img_ret_loc_d,
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
    contours = np.array([p[0] for p in contours[0]])

    contorno = []
    for p in contours:   
        p_inv = (p[1], p[0])    
        if img_label[p_inv] == 0:
            contorno.append(p_inv)
    contorno = np.array(contorno)

    while True:
        #começa a busca por um fundo válido para área de saturação
        img_only_back = np.zeros_like(img_ret)
        coordsLocNew = None
        if (len(coordsConv) == 0):
            #raise Exception('Unable to find a valid background for RQI')
            print('Unable to find a valid background for RQI')            
            break
        #faz um random nas coordenadas que ainda estão no vetor
        indexConv = random.randint(0,len(coordsConv)-1)
        #faz a localização do ponto sorteado anteriormente e remove as coordenadas do vetor
        p_new = coordsConv[indexConv]            
        coordsConv.remove(p_new)

        #localiza as coordenadas da area de saturação
        #depois movimenta as coordenadas para 0
        #depois movimenta as coordenadas para o ponto sorteado
        #isso faz com que seja possivel localizar o objeto de saturação na imagem 
        coordsLocNew = (coordsLocNorm[0] + p_new[0], coordsLocNorm[1] + p_new[1])  
        
        

        #aqui pega apenas o fundo para poder fazer as validações
        img_only_back[coordsLoc] = img_ret[coordsLocNew] 

        #faz diferença entre cada ponto da borda do vaso e seu respectivo vizinho dentro do vaso
        dif = []

        
        for cord in contorno:
            #print('cord')
            #print(cord)
            vizi = vizinhos(cord,img_ret_loc.shape)
            #print('vizi')
            #print(vizi)
            #print(img_ret_loc.shape)
            for pt in vizi:
                if ((img_ret_loc[pt[0]][pt[1]]) == 1):
                    value = int(img_only_back[pt[0]][pt[1]]) - int(img_ret[cord[0]][cord[1]])
                    dif.append(abs(value))

        #Faz a media entre todos os valores encontrados anteriormente
        media_new_center = np.mean(dif)

        #Valida se a média é válida
        if (media_new_center > median_threshold):     
            continue
        else:
        #--------linha responsável por trocar a área RQI pelo fundo                
            img_ret[coordsLoc] = img_ret[coordsLocNew] 
            break
    
    debug = img_exp, img_ret_loc, outPutConv, img_only_back, coordsLoc, coordsLocNew, contorno

    return img_ret, debug

def create_image(img_origin, img_label, rqi_len_interv, min_len_interv, 
                 n_rqi_interv, median_threshold, rng_seed=None, 
                 highlight_center=False):

    graph = create_graph(img_label,True)

    n_rqi = random.randint(n_rqi_interv[0],n_rqi_interv[1])
    graph_edges = list(graph.edges(data=True))
    img_segs = get_segments(graph, img_origin, img_label)

    if rng_seed:
        random.seed(rng_seed)

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
        
        #IMAGEM DO ESQUELETO DA RQI
        newCrop = np.zeros(img_label.shape, dtype=np.uint8)
        
        #IMAGEM DO ESQUELETO COM QUEDA DE INTENSIDADE
        newCropDyn = np.zeros(newCrop.shape, dtype=float)
       
        #CORTES DA IMAGEM DE ORIGEM E DE LABEL
        oriCrop = get_crop(img_aug, pc, rqi_len)   
        lblCrop = get_crop(img_label, pc, rqi_len)
        img_seg_crop = get_crop(img_seg, pc, rqi_len)
       
        points = edge_path[p1_idx:p2_idx+1]
       
        for point in points:
            newCrop[point] = 128

        #INDICE PP1, PP2. 
        idxpp1,idxpp2 = point_from_dist(edge_path,pc_idx,min_len//2)
        
        #PP1 = INICIO DA ÁREA MINIMA DA RQI(BACKGROUND)
        #PP2 = FIM DA ÁREA MINIMA DA RQI(BACKGROUND)    
        pp1 = edge_path[idxpp1]
        pp2 = edge_path[idxpp2]
    
        #Faz a queda de intensidade(distancia) entre p1(inicio da RQI) e pp1(inicio da area de saturação) , depois pp2(fim da area de saturacao) até p2(fim da RQI)
        values = dist_without_center(points, idxpp1-p1_idx,idxpp2-p1_idx)
    
        #plot dos valores de intensidade
        rqi_int_plot = values
        
        freq_vaso = ([])
        for cord in edge_path:        
            freq_vaso.append(img_aug[cord[0]][cord[1]])
        vessel_int_plot = freq_vaso
        
        for idx, (r, c) in enumerate(points):
            newCropDyn[r, c] = values[idx]
            
        #Corte das imagens de esqueleto para tratamento na função EXPAND_LINE
        newCrop = get_crop(newCrop, pc, rqi_len)
        newCropDyn = get_crop(newCropDyn, pc, rqi_len)
        
        newCrop_mod, debug_expand = expand_line(newCrop,newCropDyn,oriCrop,lblCrop,img_seg_crop,median_threshold,i_rqi)#dilatação
        
        img_aug = img_aug.copy()
        img_aug = crop_alter(img_aug,newCrop_mod,int(pc[0]),int(pc[1]),rqi_len)
    
        #marca a area proxima ao centro da RQI para localizar na imagem final
        if(highlight_center):     
            coords = vizinhos(pc,img_aug.shape)
            for cord in coords:            
                img_aug[cord[0]][cord[1]] = 255        
        
        freq_vaso = ([])
        for cord in edge_path:        
            freq_vaso.append(img_aug[cord[0]][cord[1]])
        vessel_int_new = freq_vaso
    
        debug = (oriCrop, lblCrop, newCrop, newCropDyn, img_seg_crop, rqi_int_plot, vessel_int_plot, 
                 vessel_int_new, edge_path, pc_idx, p1_idx, p2_idx, idxpp1, idxpp2, debug_expand,newCrop_mod)
        debug_full.append(debug)
        im = pilimg.fromarray(img_aug)
    
    return img_aug, debug_full, graph
 