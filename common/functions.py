import cv2
import numpy as np

def grayscale(img):
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray

def binary(img,th=128):
    """generate binary image 
    
    Args:
    img : image array
    th : threshold
    """
    
    img[img<th] = 0
    img[img>=th] = 255
    
    return img

def ootu_binary(img):
    """generate binary image 
    
    """
    sb_list = []
    img_flat = img.flatten() # 画像を1次元配列に変換
    n = len(img_flat) # データ数
    
    for th in range(0,256):
        data_class0 = img_flat[img_flat<th]
        data_class1 = img_flat[img_flat>=th]
        
        w0 = len(data_class0)/n # クラス0のデータが占める割合
        w1 = len(data_class1)/n # クラス1のデータが占める割合
        
        # class0の分散
        if len(data_class0)!=0:
            m0 = np.mean(data_class0)
            v0 = np.var(data_class0)
        else:
            m0=0
            v0=0
            
        # class1の分散
        if len(data_class1)!=0:
            m1 = np.mean(data_class1)
            v1 = np.var(data_class1)
        else:
            m1=0
            v1=0        
            
        sb = w0*w1*(m0-m1)**2
        sb_list.append(sb)
        
    best_th = np.argmax(sb_list)
    print("best threshold =%d"%(best_th))
    return binary(img,best_th)

def rgb2hsv(img):
    """transform RGB image to HSV image
    
    """
    img = img/255
    hsv = np.zeros_like(img, dtype=np.float32)
    
    max_v = np.max(img,axis=2)
    min_v = np.min(img,axis=2)
    min_arg = np.argmin(img,axis=2)
    
    # Hの計算 ...は次元を省略して書く記法
    hsv[...,0][np.where(max_v==min_v)]=0
    ## if min == B
    ind = np.where(min_arg == 0)
    hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
    ## if min == R
    ind = np.where(min_arg == 2)
    hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180
    ## if min == G
    ind = np.where(min_arg == 1)
    hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300
    
    # Sの計算
    hsv[...,1] = max_v-min_v
    
    # Vの計算
    hsv[...,2] = max_v
    
    return hsv
    
def hsv2rgb(img):
    """transform HSV image to RGB image

    """
    
    max_v = np.max(img,axis=2)
    min_v = np.min(img,axis=2)
    rgb = np.zeros_like(img)
    
    C = img[...,1]
    Hdash = img[...,0]/60
    X = C * (1 - np.abs( Hdash % 2 - 1))
    Z = np.zeros_like(img[...,0])
    
    vals = [[Z,X,C], [Z,C,X], [X,C,Z], [C,X,Z], [C,Z,X], [X,Z,C]]

    for i in range(6):
        ind = np.where((i <= Hdash) & (Hdash < (i+1)))
        rgb[..., 0][ind] = (img[...,2] - C)[ind] + vals[i][0][ind]
        rgb[..., 1][ind] = (img[...,2] - C)[ind] + vals[i][1][ind]
        rgb[..., 2][ind] = (img[...,2] - C)[ind] + vals[i][2][ind]
    
    rgb[np.where(max_v == min_v)] = 0
    rgb = np.clip(rgb, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)

    return rgb

def substract_color(img):
    """substruct image to four color(32,96,160,224)
    
    """
    
    out = img.copy()

    out = out // 64 * 64 + 32 # 整数の割り算

    return out

def average_pooling(img):
    out = np.zeros_like(img)
    grid_size = 8
    img_width,img_height,layer = img.shape
    for height in range(0,img_height,grid_size):
        for width in range(0,img_width,grid_size):
            for l in range(layer):
                out[height:height+grid_size,width:width+grid_size,l] = np.mean(img[height:height+grid_size,width:width+grid_size,l])
                
    return out.astype(np.uint8)

def max_pooling(img):
    out = np.zeros_like(img)
    grid_size = 8
    img_width,img_height,layer = img.shape
    for height in range(0,img_height,grid_size):
        for width in range(0,img_width,grid_size):
            for l in range(layer):
                out[height:height+grid_size,width:width+grid_size,l] = np.max(img[height:height+grid_size,width:width+grid_size,l])
                
    return out.astype(np.uint8)

def gaussian_filter(img, K_size=3, sigma=1.3):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape

    ## Zero padding
    pad = K_size // 2
    out = np.zeros([H + pad * 2, W + pad * 2, C], dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

    ## prepare Kernel
    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp( - (x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    #K /= (sigma * np.sqrt(2 * np.pi))
    K /= (2 * np.pi * sigma * sigma)
    K /= K.sum()

    tmp = out.copy()

    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(K * tmp[y : y + K_size, x : x + K_size, c])

    out = np.clip(out, 0, 255)
    out = out[pad : pad + H, pad : pad + W]
    out = out.astype(np.uint8)
    out = out[..., 0]

    return out

def median_filter(img):
    """filtering 3x3 median filter
    
    """
    return  cv2.medianBlur(img_noise,3)

def smoothing_filter(img):
    """filtering 3x3 smoothing filter
    
    """

    K = np.ones((3,3),np.float32)/9
    return cv2.filter2D(img,-1,K)

def motion_filter(img):
    """filtering 3x3 motion filter

    """

    K = np.zeros((3,3),np.float32)
    K[0][0] = 1/3
    K[1][1] = 1/3
    K[2][2] = 1/3
    return cv2.filter2D(img,-1,K)

def differential_filter(img,v="x"):
    """filtering 3x3 defferential filter

    Args:
    img : image array
    v : when v equals x, calculate x direction's differential filter. 
        when v equal y, calculate y direction's differential filter.
        otherwise return -1 .
    """    
    
    K_x = np.array([[0,0,0],[-1,1,0],[0,0,0]])
    K_y = np.array([[0,-1,0],[0,1,0],[0,0,0]])
    if v=="x":
        return cv2.filter2D(img,-1,K_x)
    if v=="y":
        return cv2.filter2D(img,-1,K_y)
    else :
        return -1

def sobel_filter(img,v="x"):
    """filtering 3x3 sobel filter

    Args:
    img : image array
    v : when v equals x, calculate x direction's sobel filter. 
        when v equal y, calculate y direction's sobel filter.
        otherwise return -1 .
    """

    K_x = np.array([[1.,0.,-1.],[2.,0.,-2.],[1.,0.,-1.]])
    K_y = np.array([[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]])
    if v=="x":
        return cv2.filter2D(img,-1,K_x)
    if v=="y":
        return cv2.filter2D(img,-1,K_y)
    else :
        return -1

def prewitt_filter(img,v="x"):
    """filtering 3x3 prewitt filter

    Args:
    img : image array
    v : when v equals x, calculate x direction's prewitt filter. 
        when v equal y, calculate y direction's prewitt filter.
        otherwise return -1 .
    """

    K_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    K_y = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])

    if v=="x":
        return cv2.filter2D(img,-1,K_x)
    if v=="y":
        return cv2.filter2D(img,-1,K_y)
    else :
        return -1

def laplacian_filter(img):
    """filtering 3x3 laplacian filter

    Args:
    img : image array
    """

    K = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    return cv2.filter2D(img,-1,K_x)

def  emboss_filter(img):
    """filtering 3x3 emboss filter

    Args:
    img : image array
    """

    K = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
    return cv2.filter2D(img,-1,K_x)

def scale_transform(img,output_range):
    """
    gray scale transformation
    
    Args:
    x : pixel value
    input range : input pixel's range [min,max]
    output range : output pixel's range [min,max]
    
    Returns:
    result of gray scale transformation
    """
    
    c,d = img.min(),img.max()
    a,b = output_range[0],output_range[1]
    h_img,w_img,l= img.shape
    result = img.copy()

    result = (b-a)/(d-c)*(img-c) + a
    result[img<c] = a
    result[img>d] = b
    return result.astype(np.uint8)

def hist_ctrl(img,m0,s0):
    result = img.copy()
    m = np.mean(img)
    s = np.std(img)
    result = s0/s *(img-m)+m0
    result[result<0] = 0
    result[result>255] = 255 
    
    return result.astype(np.uint8)

def hist_equalize(img):
    h_img,w_img,l = img.shape
    S = h_img*w_img*l
    result = img.copy()
    z_max = img.max()
    sum_h = 0

def gamma_adj(img,c=1,g=2.2):
    result = img.copy()/255
    result= (result/c)**(1/g)
    result*=255
    return result.astype(np.uint8)

def nn_interpolate(img,ax=1,ay=1):
    """
    Nereset Neighbor interpolation
    
    Args:
    img : image
    ax,ay: expantion rate
    """
    h,w,l = img.shape
    h_a = int(ay*h)
    w_a = int(ax*w)
    
    y = np.arange(h_a).repeat(w_a).reshape(h_a,-1) # 0~h_aをw_a回繰り返したベクトルと(h_a,w_a)行列に変換したもの
    x = np.tile(np.arange(w_a),(h_a,1)) # 先の行列のx方向バージョン
    y = np.round(y/ay).astype(np.int8)
    x = np.round(x/ax).astype(np.int8)
    out = img[y,x]
    
    return out

def bi_linear(img,ax=1,ay=1):
    h,w,l = img.shape
    h_a = int(ay*h)
    w_a = int(ax*w)
    # get position of resized image
    y = np.arange(h_a).repeat(w_a).reshape(h_a, -1)
    x = np.tile(np.arange(w_a), (h_a, 1))

    # get position of original position
    y = (y / ay)
    x = (x / ax)

    ix = np.floor(x).astype(np.int32)
    iy = np.floor(y).astype(np.int32)

    ix = np.minimum(ix, w-2)
    iy = np.minimum(iy, h-2)

    # get distance 
    dx = x - ix
    dy = y - iy

    dx = np.repeat(np.expand_dims(dx, axis=-1), 3, axis=-1)
    dy = np.repeat(np.expand_dims(dy, axis=-1), 3, axis=-1)
    
    # interpolation
    out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out

def shift_img(img,tx,ty):
    h,w,l = img.shape
    
    M = np.array([[1,0,tx],[0,1,ty]],dtype=np.float32)
    out = cv2.warpAffine(img,M,(w,h))
    return out

def scale_img(img,n,m):
    h,w,l = img.shape
    
    M = np.array([[n,0,1],[0,m,1]],dtype=np.float32)
    out = cv2.warpAffine(img,M,(int(w*n),int(h*m)))
    return out

def rotate_img_by_origin(img,A):
    # 画像の原点を中心に時計回りにA回転, 角度は度で指定
    h,w,l = img.shape
    
    affine = cv2.getRotationMatrix2D((0,0), A, 1.0)
    return cv2.warpAffine(img, affine, (w, h))

def rotate_img_by_center(img,A):
    # 画像の中央を中心に時計回りにA回転, 角度は度で指定
    h,w,l = img.shape
    
    affine = cv2.getRotationMatrix2D((w/2.0, h/2.0), A, 1.0)
    return cv2.warpAffine(img, affine, (w, h))

def skew_img(img,sx,sy):
    h,w,l = img.shape
    
    M = np.array([[1,sx/h,0],[sy/w,1,0]],dtype=np.float32)
    out = cv2.warpAffine(img,M,(w+sx,h+sy))
    return out

def low_pass_filter(img,r):
    """generate low pass filter
    Args:
    img : image array
    r : radius

    """
    size = img_gray.shape
    mask = np.zeros(size)
    length = size[0]
    centery = size[0]/2
    for x in range(0,length):
        for y in range(0,length):
            if (x- centery)**2 +(y- centery)**2 <r**2:
                mask[x,y]=1
    return mask

def high_pass_filter(img,r):
    """generate high pass filter
    Args:
    img : image array
    r : radius

    """
    size = img_gray.shape
    mask = np.zeros(size)
    length = size[0]
    centery = size[0]/2
    for x in range(0,length):
        for y in range(0,length):
            if (x- centery)**2 +(y- centery)**2 >r**2:
                mask[x,y]=1
    return mask

def get_edge_angle(fx,fy):
    """エッジ強度と勾配を計算する関数
    """

    # np.power : 行列のn乗を計算
    # np.sqrt : 各要素の平方根を計算
    edge = np.sqrt(np.power(fx.astype(np.float32),2)+np.power(fy.astype(np.float32),2))
    edge = np.clip(edge, 0, 255)
    fx = np.maximum(fx, 1e-5)
    angle = np.arctan(fy/fx)
    return edge,angle

def angle_quantization(angle):
    angle = 180*angle/np.pi
    angle[angle < -22.5] = 180 + angle[angle < -22.5]
    _angle = np.zeros_like(angle, dtype=np.uint8)
    _angle[np.where(angle <= 22.5)] = 0
    _angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
    _angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
    _angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135

    return _angle

def non_maximum_suppression1(angle, edge):
    H, W = angle.shape
    _edge = edge.copy()

    for y in range(H):
        for x in range(W):
            if angle[y, x] == 0:
                dx1, dy1, dx2, dy2 = -1, 0, 1, 0
            elif angle[y, x] == 45:
                dx1, dy1, dx2, dy2 = -1, 1, 1, -1
            elif angle[y, x] == 90:
                dx1, dy1, dx2, dy2 = 0, -1, 0, 1
            elif angle[y, x] == 135:
                dx1, dy1, dx2, dy2 = -1, -1, 1, 1
            if x == 0:
                dx1 = max(dx1, 0)
                dx2 = max(dx2, 0)
            if x == W-1:
                dx1 = min(dx1, 0)
                dx2 = min(dx2, 0)
            if y == 0:
                dy1 = max(dy1, 0)
                dy2 = max(dy2, 0)
            if y == H-1:
                dy1 = min(dy1, 0)
                dy2 = min(dy2, 0)
            if max(max(edge[y, x], edge[y + dy1, x + dx1]), edge[y + dy2, x + dx2]) != edge[y, x]:
                _edge[y, x] = 0

    return _edge

def non_maximum_suppression2(hough):
    rmax,_ = hough.shape

    for y in range(rmax):
        for x in range(180):
            # x,yが行列からはみ出さないように調整
            x1 = max(x-1,0)
            x2 = min(x+2,180)
            y1 = max(y-1,0)
            y2 = min(y+2,rmax-1) 
            # 注目画素と8近傍のvoteの最大値が注目画素でかつ0でないとき
            if np.max(hough[y1:y2, x1:x2]) == hough[y,x] and hough[y, x] != 0:
                pass
            else :
                hough[y,x]=0

    # 上位20voteを取得
    ind_x = np.argsort(hough.ravel())[::-1][:20]
    ind_y = ind_x.copy()
    thetas = ind_x % 180
    rhos = ind_y // 180
    _hough = np.zeros_like(hough, dtype=np.int)
    _hough[rhos, thetas] = 255
    return _hough

def hysterisis(edge, HT=100, LT=30):
    H, W = edge.shape

    # Histeresis threshold
    edge[edge >= HT] = 255
    edge[edge <= LT] = 0

    _edge = np.zeros((H + 2, W + 2), dtype=np.float32)
    _edge[1 : H + 1, 1 : W + 1] = edge

    ## 8 - Nearest neighbor
    nn = np.array(((1., 1., 1.), (1., 0., 1.), (1., 1., 1.)), dtype=np.float32)

    for y in range(1, H+2):
            for x in range(1, W+2):
                if _edge[y, x] < LT or _edge[y, x] > HT:
                    continue
                if np.max(_edge[y-1:y+2, x-1:x+2] * nn) >= HT:
                    _edge[y, x] = 255
                else:
                    _edge[y, x] = 0

    edge = _edge[1:H+1, 1:W+1]
                                
    return edge

def hough_transform(edge):
    H,W = edge.shape
    rmax = np.sqrt(H**2 + W**2).astype(np.int)

    hough = np.zeros((rmax*2,180),dtype=np.int)
    ind = np.where(edge == 255) # edgeのインデックス

    # すべてのedgeについて
    for y,x in zip(ind[0],ind[1]):
        # 0から180度について
        for theta in range(0,180,1):
            t = np.pi / 180 * theta
            rho = int(x * np.cos(t) + y * np.sin(t))
            hough[rho + rmax, theta] += 1
    out = hough.astype(np.uint8)
    return out

def inverse_hough(hough,img):
    H,W,_ = img.shape
    rmax,_ = hough.shape

    out = img.copy()

    # 上位20voteを取得
    ind_x = np.argsort(hough.ravel())[::-1][:20]
    ind_y = ind_x.copy()
    thetas = ind_x % 180
    rhos = ind_y // 180 -rmax/2

    for theta, rho in zip(thetas,rhos):
        t = np.pi/180 *theta

        for x in range(W):
            if np.sin(t)!=0:
                y = - (np.cos(t) / np.sin(t)) * x + (rho) / np.sin(t)
                y = int(y)
                if y >= H or y < 0:
                    continue
                out[y, x] = [0, 0, 255]

        for y in range(H):
            if np.cos(t) != 0:
                x = - (np.sin(t) / np.cos(t)) * y + (rho) / np.cos(t)
                x = int(x)
                if x >= W or x < 0:
                    continue
                out[y, x] = [0, 0, 255]
    
    out = out.astype(np.uint8)
    return out

def morphology_expand(img):
    H,W = img.shape

    out = np.copy(img)
    for y in range(1,H-1):
        for x in range(1,W-1):
            if img[y,x]==0:
                tmp_l = [img[y-1,x],img[y,x-1],img[y,x+1],img[y+1,x]]
                if max(tmp_l) == 255:
                    out[y,x] = 255

    return out

def morphology_shrinkage(img):
    H,W = img.shape

    out = np.copy(img)
    for y in range(1,H-1):
        for x in range(1,W-1):
            if img[y,x]==255:
                tmp_l = [img[y-1,x],img[y,x-1],img[y,x+1],img[y+1,x]]
                if min(tmp_l) == 0:
                    out[y,x] = 0

    return out    