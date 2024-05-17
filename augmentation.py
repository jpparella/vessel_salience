from pyvane.graph.creation import create_graph
from pyvane.graph import adjustment as net_adjust
from pyvane.image import Image
from pyvane import util
import skimage.morphology
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as pilimg
import skimage.draw
import random
import scipy.ndimage as ndi
import json

def extract_full_lenght(graph):
    obj_listas = list(graph.edges(data=True))

    segment_lenght = []    
    for lista in obj_listas:           
        obj_cords = lista[2]['path']
        tamanho = path_length([obj_cords[0],obj_cords[-1]])
        segment_lenght.append(tamanho) 
    minn =  min(segment_lenght)
    maxx =  max(segment_lenght)
    
    return segment_lenght,minn,maxx
        
def dist_p(points, p):
    '''Distance between each point in `points` and a given point `p`.'''

    return np.sqrt(np.sum((points - p)**2, axis=1))

def dist_without_center(points, idxpp1, idxpp2):
    '''distance between 2 points to all other points, where the area between these two points is returned as 0'''
    #* Várias mudanças nesta função, para considerar distância ao longo do segmento
    # e por questão de eficiência.
    path_length = path_length_cum(points)
    first_seg = path_length[idxpp1] - path_length[:idxpp1]
    first_seg /= first_seg[0]
    second_seg = path_length[idxpp2+1:] - path_length[idxpp2]
    second_seg /= second_seg[-1]

    dist_vec = np.concatenate((first_seg, np.zeros(idxpp2-idxpp1+1), second_seg))

    return dist_vec

def path_length(path):
    dpath = np.diff(path, axis=0)
    dlengths = np.sqrt(np.sum(dpath**2, axis=1))

    
    path_length = np.sum(dlengths)

    return path_length
    
def path_length_cum(paths):
    dpath = np.diff(paths, axis = 0)
   
    dlengths = np.sqrt(np.sum(dpath**2, axis=1))
    
    path_length = np.cumsum(dlengths)
    
    path_length = np.array([0]+path_length.tolist())
    
    return path_length

def pointFromDist(points, index_pc, leng):
    returns = ([])
    pc = points[index_pc]
    res = path_length_cum(points)
    distpc = res[index_pc]
    distp1 = distpc - leng
    distp2 = distpc + leng
    indexp1 = np.argmin(np.abs(res-distp1))
    indexp2 = np.argmin(np.abs(res-distp2))
    return indexp1,indexp2
    
def get_segments(graph, img_origin, img_label):
    '''Cria imagem na qual cada segmento de vaso possui um id diferente. Útil para isolar
    apenas o vaso que está sendo processado.'''

    obj_listas = list(graph.edges(data=True))
    img_graph = np.zeros_like(img_origin, dtype=int) - 1
    for idx, item in enumerate(obj_listas):
        path = item[2]['path']
        #idx_rand = np.random.randint(0, len(obj_listas))
        for p in path:
            img_graph[p] = idx

    _, (img_inds_r, img_inds_c) = ndi.distance_transform_edt(img_graph==-1, return_indices=True)
    img_exp = img_graph[img_inds_r, img_inds_c]
    img_exp = img_exp*img_label//255

    return img_exp

def create_image(rqi_len_param_interv, min_len_interv, n_rqi_interv,graph,img_label,img_origin,median_threshold,rnd_seed=None,highlight_center=False):
    

    n_rqi = random.randint(n_rqi_interv[0],n_rqi_interv[1])
    obj_listas = list(graph.edges(data=True))
    img_segs = get_segments(graph, img_origin, img_label)
    # print('img_segs')
    # print(img_segs)

    # plt.imshow(img_label, cmap='gray')
    # # plt.subplot(1, 2, 2)
    # # plt.imshow(seg.detach().numpy(), cmap='gray')
    # plt.show()

      
    if (rnd_seed):
        random.seed(rnd_seed)
    
    img_new = img_origin.copy()
    
    debug_full = []
    block_idx = []
    for i_rqi in range(n_rqi):
        
        rqi_len_param = random.randint(rqi_len_param_interv[0],rqi_len_param_interv[1])
        max_min_len = min([min_len_interv[1],rqi_len_param-5])
        
        min_len = random.randint(min_len_interv[0],max_min_len)
        
        list_valid_idx,idx_vector = onlyValidIndexed(obj_listas,rqi_len_param//2)  

        
        
        index1 = random.randint(0,len(list_valid_idx)-1)
        list_valid = list_valid_idx[index1][1]
        index2 = random.randint(0,len(list_valid)-1)
       
        if (index1 in block_idx):
            continue
        
        block_idx.append(index1)
        #if (obj_listas[list_valid_idx[index1][0]])
        
        pc = list_valid[index2]
        
        
        #PONTOS QUE PERTENCEM MAS NÃO SAO VALIDOS PARA SER PC
        idx_notvalid = list_valid_idx[index1][0]
    
        obj_cords = obj_listas[idx_notvalid][2]['path']
        img_seg = img_segs==idx_notvalid

        # plt.imshow(img_seg, cmap='gray')
        # # plt.subplot(1, 2, 2)
        # # plt.imshow(seg.detach().numpy(), cmap='gray')
        # plt.show()

        '''
        Variáveis:
        list_valid: lista de pontos válidos em todos os segmentos, segmentos sem pontos válidos
            não incluídos
        idx_vector: lista de pontos válidos em todos os segmentos, incluindo segmentos vazios
        obj_cords: lista de todos os pontos do segmento selecionado
        pc: ponto central na lista de pontos válidos do segmento selecionado (list_valid[index1])
        idxpc: ponto central em obj_cords
        img_seg: imagem contendo o segmento de vaso selecionado
        '''
    
        idxpc = obj_cords.index(pc)
        #INDICE P1 E P2
        idxp1,idxp2 = pointFromDist(obj_cords,idxpc,rqi_len_param//2)
        
        #P1 = PONTO INICIAL DA RQI
        #P2 = PONTO FINAL DA RQI
        p1 = obj_cords[idxp1]   
        p2 = obj_cords[idxp2]
        
        #IMAGEM DO ESQUELETO DA RQI
        newCrop = np.zeros(img_label.shape,dtype=np.uint8)
        
        #IMAGEM QUE VAI SER RETORNADA COM AS ALTERAÇÕES
        
        
        #IMAGEM DO ESQUELETO COM QUEDA DE INTENSIDADE
        newCropDyn = np.zeros(newCrop.shape, dtype=float)
       
        #CORTES DA IMAGEM DE ORIGEM E DE LABEL
        oriCrop = crop(img_new,int(pc[0]),int(pc[1]),rqi_len_param)   
        lblCrop = crop(img_label,int(pc[0]),int(pc[1]),rqi_len_param)
        img_seg_crop = crop(img_seg,int(pc[0]),int(pc[1]),rqi_len_param)

        # plt.imshow(img_seg_crop, cmap='gray')
        # # plt.subplot(1, 2, 2)
        # # plt.imshow(seg.detach().numpy(), cmap='gray')
        # plt.show()
        # plt.imshow(img_label, cmap='gray')
        # # plt.subplot(1, 2, 2)
        # # plt.imshow(seg.detach().numpy(), cmap='gray')
        # plt.show()
        
        #* Usar idxp2+1 ao invés de idxp2 para pegar também o último ponto.
        points = obj_cords[idxp1:idxp2+1]
       
        for point in points:        
            newCrop[point] = 128
            
        #INDICE PP1, PP2. 
        idxpp1,idxpp2 = pointFromDist(obj_cords,idxpc,min_len//2)
        
        #PP1 = INICIO DA ÁREA MINIMA DA RQI(BACKGROUND)
        #PP2 = FIM DA ÁREA MINIMA DA RQI(BACKGROUND)    
        pp1 = obj_cords[idxpp1]
        pp2 = obj_cords[idxpp2]
    
        #Faz a queda de intensidade(distancia) entre p1(inicio da RQI) e pp1(inicio da area de saturação) , depois pp2(fim da area de saturacao) até p2(fim da RQI)
        values = dist_without_center(points, idxpp1-idxp1,idxpp2-idxp1)
    
        #plot dos valores de intensidade
        rqi_int_plot = values
        
        freq_vaso = ([])
        for cord in obj_cords:        
            freq_vaso.append(img_new[cord[0]][cord[1]])
        vessel_int_plot = freq_vaso
        
        for idx, (r, c) in enumerate(points):
            newCropDyn[r, c] = values[idx]
            
        #Corte das imagens de esqueleto para tratamento na função EXPAND_LINE
        newCrop = crop(newCrop,int(pc[0]),int(pc[1]),rqi_len_param)
        newCropDyn = crop(newCropDyn,int(pc[0]),int(pc[1]),rqi_len_param)
        
        newCrop_mod, debug_expand = expand_line(newCrop,newCropDyn,oriCrop,lblCrop,img_seg_crop,median_threshold,i_rqi)#dilatação
        
        img_new = img_new.copy()
        img_new = cropAlter(img_new,newCrop_mod,int(pc[0]),int(pc[1]),rqi_len_param)
    
        #marca a area proxima ao centro da RQI para localizar na imagem final
        if(highlight_center):     
            coords = vizinhos(pc,img_new.shape)
            for cord in coords:            
                img_new[cord[0]][cord[1]] = 255        
        
        freq_vaso = ([])
        for cord in obj_cords:        
            freq_vaso.append(img_new[cord[0]][cord[1]])
        vessel_int_new = freq_vaso
    
        debug = (oriCrop, lblCrop, newCrop, newCropDyn, img_seg_crop, rqi_int_plot, vessel_int_plot, 
                 vessel_int_new, obj_cords, idxpc, idxp1, idxp2, idxpp1, idxpp2, debug_expand,newCrop_mod)
        debug_full.append(debug)
        im = pilimg.fromarray(img_new)
        
        
       
    
    return img_new, debug_full
        
    

def image_augmentation(graph,img_label,img_origin, rqi_len_param,min_len,median_threshold,rnd_seed=None,highlight_center=False):

    #* Em toda esta função mudei o uso de rqi_len para rqi_len_param. rqi_len não é mais usado
    #rqi_len = rqi_len_param - min_len#colocar o rqi_len_param como tamanho total, e subtrair o min_len
    
    obj_listas = list(graph.edges(data=True))
    img_segs = get_segments(graph, img_origin, img_label)

    list_valid,idx_vector = onlyValid(obj_listas,rqi_len_param//2)       
    if (rnd_seed):
        random.seed(rnd_seed)
    
    index1 = random.randint(0,len(list_valid)-1)

    index2 = random.randint(0,len(list_valid[index1])-1)
   
    
    pc = list_valid[index1][index2]
    
    
    #PONTOS QUE PERTENCEM MAS NÃO SAO VALIDOS PARA SER PC
    idx_notvalid = idx_vector.index(list_valid[index1])

    obj_cords = obj_listas[idx_notvalid][2]['path']
    img_seg = img_segs==idx_notvalid
    
    '''
    Variáveis:
    list_valid: lista de pontos válidos em todos os segmentos, segmentos sem pontos válidos
        não incluídos
    idx_vector: lista de pontos válidos em todos os segmentos, incluindo segmentos vazios
    obj_cords: lista de todos os pontos do segmento selecionado
    pc: ponto central na lista de pontos válidos do segmento selecionado (list_valid[index1])
    idxpc: ponto central em obj_cords
    img_seg: imagem contendo o segmento de vaso selecionado
    '''

    idxpc = obj_cords.index(pc)
    #INDICE P1 E P2
    idxp1,idxp2 = pointFromDist(obj_cords,idxpc,rqi_len_param//2)
    
    #P1 = PONTO INICIAL DA RQI
    #P2 = PONTO FINAL DA RQI
    p1 = obj_cords[idxp1]   
    p2 = obj_cords[idxp2]
    
    #IMAGEM DO ESQUELETO DA RQI
    newCrop = np.zeros(img_label.shape,dtype=np.uint8)
    
    #IMAGEM QUE VAI SER RETORNADA COM AS ALTERAÇÕES
    img_new = np.zeros(img_label.shape,dtype=np.uint8)
    
    #IMAGEM DO ESQUELETO COM QUEDA DE INTENSIDADE
    newCropDyn = np.zeros(newCrop.shape, dtype=float)
   
    #CORTES DA IMAGEM DE ORIGEM E DE LABEL
    oriCrop = crop(img_origin,int(pc[0]),int(pc[1]),rqi_len_param)   
    lblCrop = crop(img_label,int(pc[0]),int(pc[1]),rqi_len_param)
    img_seg_crop = crop(img_seg,int(pc[0]),int(pc[1]),rqi_len_param)
    
    #* Usar idxp2+1 ao invés de idxp2 para pegar também o último ponto.
    points = obj_cords[idxp1:idxp2+1]
   
    for point in points:        
        newCrop[point] = 128
        
    #INDICE PP1, PP2. 
    idxpp1,idxpp2 = pointFromDist(obj_cords,idxpc,min_len//2)
    
    #PP1 = INICIO DA ÁREA MINIMA DA RQI(BACKGROUND)
    #PP2 = FIM DA ÁREA MINIMA DA RQI(BACKGROUND)    
    pp1 = obj_cords[idxpp1]
    pp2 = obj_cords[idxpp2]

    #Faz a queda de intensidade(distancia) entre p1(inicio da RQI) e pp1(inicio da area de saturação) , depois pp2(fim da area de saturacao) até p2(fim da RQI)
    values = dist_without_center(points, idxpp1-idxp1,idxpp2-idxp1)

    #plot dos valores de intensidade
    rqi_int_plot = values
    
    freq_vaso = ([])
    for cord in obj_cords:        
        freq_vaso.append(img_origin[cord[0]][cord[1]])
    vessel_int_plot = freq_vaso
    
    for idx, (r, c) in enumerate(points):
        newCropDyn[r, c] = values[idx]
        
    #Corte das imagens de esqueleto para tratamento na função EXPAND_LINE
    newCrop = crop(newCrop,int(pc[0]),int(pc[1]),rqi_len_param)
    newCropDyn = crop(newCropDyn,int(pc[0]),int(pc[1]),rqi_len_param)
    
    newCrop_mod, debug_expand = expand_line(newCrop,newCropDyn,oriCrop,lblCrop,img_seg_crop,median_threshold)#dilatação
    
    img_new = img_origin.copy()
    img_new = cropAlter(img_new,newCrop_mod,int(pc[0]),int(pc[1]),rqi_len_param)

    #marca a area proxima ao centro da RQI para localizar na imagem final
    if(highlight_center):     
        coords = vizinhos(pc,img_new.shape)
        for cord in coords:            
            img_new[cord[0]][cord[1]] = 255        
    
    freq_vaso = ([])
    for cord in obj_cords:        
        freq_vaso.append(img_new[cord[0]][cord[1]])
    vessel_int_new = freq_vaso

    debug = (oriCrop, lblCrop, newCrop, newCropDyn, img_seg_crop, rqi_int_plot, vessel_int_plot, 
             vessel_int_new, obj_cords, idxpc, idxp1, idxp2, idxpp1, idxpp2, debug_expand)

    return img_new, debug
    
#retorna apenas coordenadas válidas para o centro do RQI    
def onlyValid(obj_listas,cut):
    validPixels = ([])
    aux = ([])
    idx_ = ([])
    for lista in obj_listas:           
        obj_cords = lista[2]['path']
        #* Distância no segmentedo ao invés da distância entre pontos
        path_length = path_length_cum(obj_cords)
        for idx,dist in enumerate(path_length):
            #if ((path_length([px,obj_cords[0]]) >= cut and path_length([px,obj_cords[-1]]) >= cut) ):
            if dist > cut and (path_length[-1]-dist) > cut:
                aux.append(obj_cords[idx])
        if not(aux == []):
            validPixels.append(aux)
        idx_.append(aux)
        aux = ([])
    return validPixels,idx_

def onlyValidIndexed(obj_listas,cut):
    validPixels = ([])
    aux = ([])
    idx_ = ([])
    index_idx = 0
    for lista in obj_listas: 
        
        obj_cords = lista[2]['path']
        #* Distância no segmentedo ao invés da distância entre pontos
        path_length = path_length_cum(obj_cords)
        for idx,dist in enumerate(path_length):
            #if ((path_length([px,obj_cords[0]]) >= cut and path_length([px,obj_cords[-1]]) >= cut) ):
            if dist > cut and (path_length[-1]-dist) > cut:
                aux.append(obj_cords[idx])
        
        idx_.append(aux)
        if not(aux == []):
            validPixels.append([index_idx,aux])
        index_idx=index_idx+1         
        aux = ([])
    return validPixels,idx_
    
def createGraph(img,adjust):
    img_bin = np.clip(img, 0, 1)    
    img_skel = skimage.morphology.skeletonize(img_bin, method='lee')
    # Convert back to PyVaNe
    data_skel = Image(img_skel)
    proto_graph = create_graph(data_skel)
    if (adjust):
        graph_vessel = net_adjust.adjust_graph(proto_graph, 0)
        return graph_vessel
    return proto_graph

#funcao de corte da imagem para facilitar processamento da RQI
def crop(img,p1,p2,rqi_len):
    #* Uso do max para evitar que o crop seja negativo
    param1 = max([0, p1 -rqi_len])
    param2 = p1 +rqi_len
    
    param3 = max([0, p2 -rqi_len])
    param4 = p2 +rqi_len
    
    imgRetorno = img[param1:param2,param3:param4]
    return imgRetorno

#funcao de substituição da imagem original pela imagem modificada, é trocado a area cortada da imagem pela nova imagem em tamanho menor
def cropAlter(img,img2,p1,p2,rqi_len):    
    param1 = max([0, p1 -rqi_len])
    param2 = p1 +rqi_len
    
    param3 = max([0, p2 -rqi_len])
    param4 = p2 +rqi_len

    img[param1:param2,param3:param4] = img2
    return img

def calcMedian(img,mask):
    return np.mean(img[mask>0])

#retorna vizinhos de um pixel
def vizinhos(cord_center,shape):
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
    '''
    Variáveis:
    img_line: esqueleto da RQI
    img_linedyn: esqueleto da RQI com queda de intensidade normalizada
        Exemplo: [1,0.9,...,0.1,0,0,0,0,0.1,...,0.9,0.1]
    img_origin: imagem dos vasos
    img_label: imagem com rótulos
    Todas as imagens possuem um crop    
    '''

    #* Mudei várias coisas nesta função

    #* conversão para float para melhorar a precisão
    img_origin = img_origin.astype(float)  
    #cria imagem binária da imagem do esqueleto da RQI
    img_line = img_line == 128
    #Diltação da imagem de esqueleto da RQI 
    #img_bin = ndi.binary_dilation(img_line, iterations=width)
    #img_exp = np.zeros_like(img_line, dtype=float)        
    #img_bin = img_bin * img_label

    #vesValue = calcMedian(img_origin,(img_label == 255))
    
    img_dist, (img_inds_r, img_inds_c) = ndi.distance_transform_edt(np.logical_not(img_line), return_indices=True)
    
    #img_dist = img_dist/np.max(img_dist)

    bckValue = calcMedian(img_origin,(img_label != 255))

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