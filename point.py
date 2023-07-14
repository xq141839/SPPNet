import numpy as np
import cv2
from scipy.ndimage import gaussian_filter



def RandomExtractor(npy_path, pos_num = 1, neg_num = 1):
    # np.random.seed(42)
    point_coord = []
    point_class = []

    npy = np.load(npy_path)
    mask = npy[:,:,0]
    classes = npy[:,:,1]
    num_cell = list(set(mask.flatten()))[1:]

    if pos_num == -1:
        pos_num = len(num_cell)
        neg_num = len(num_cell)
    
    if pos_num == 0:
        cell_num = len(num_cell)
        pos_num = np.random.randint(cell_num+1,size=1)[0]
        neg_num = pos_num

    neg_coord = np.argwhere(mask == 0)
    neg_ids = np.random.randint(len(neg_coord+1),size=neg_num)
    for idx in neg_ids:
        row, col = neg_coord[idx][0], neg_coord[idx][1]
        point_coord.append([row, col])
        point_class.append(classes[row][col])

    pos_ids = np.random.choice(num_cell, pos_num, replace=False)
    for idx in pos_ids:
        pos_coord = np.argwhere(mask == idx)
        coord_idx = np.random.randint(len(pos_coord+1),size=1)[0]
        row, col = pos_coord[coord_idx][0], pos_coord[coord_idx][1]
        point_coord.append([col, row])
        point_class.append(classes[row][col])

    return point_coord, point_class


def CentreExtractor(npy_path, pos_num = 1, neg_num = 1):
    np.random.seed(42)
    point_coord = []
    point_class = []

    npy = np.load(npy_path)
    mask = npy[:,:,0]
    classes = npy[:,:,1]
    num_cell = list(set(mask.flatten()))[1:]

    if pos_num == -1:
        pos_num = len(num_cell)
        neg_num = len(num_cell)
    
    if pos_num == 0:
        cell_num = len(num_cell)
        pos_num = np.random.randint(cell_num+1,size=1)[0]
        neg_num = pos_num

    neg_coord = np.argwhere(mask == 0)
    neg_ids = np.random.randint(len(neg_coord+1),size=neg_num)
    for idx in neg_ids:
        row, col = neg_coord[idx][0], neg_coord[idx][1]
        point_coord.append([col, row])
        point_class.append(classes[row][col])

    pos_ids = np.random.choice(num_cell, pos_num, replace=False)
    for idx in pos_ids:
        src = np.zeros(mask.shape, np.uint8)
        src[mask == idx] = 255
        dist = cv2.distanceTransform(src, cv2.DIST_L1, 3)
        pos_coord = np.argwhere(dist == dist.max())
        row, col = pos_coord[0][0], pos_coord[0][1]
        point_coord.append([col, row])
        point_class.append(classes[row][col])
        

    return point_coord, point_class

def CNPS(npy_path, pos_num = 1, neg_num = 1, is_center = False):
    np.random.seed(42)
    point_coord = []
    point_class = []

    npy = np.load(npy_path)
    mask = npy[:,:,0]
    classes = npy[:,:,1]
    num_cell = list(set(mask.flatten()))[1:]

    if pos_num == -1:
        pos_num = len(num_cell)
        neg_num = len(num_cell)
    
    if pos_num == 0:
        cell_num = len(num_cell)
        pos_num = np.random.randint(cell_num+1,size=1)[0]
        neg_num = pos_num

    neg_coord = np.argwhere(mask == 0)
    neg_ids = np.random.randint(len(neg_coord+1),size=neg_num)
    for idx in neg_ids:
        row, col = neg_coord[idx][0], neg_coord[idx][1]
        point_coord.append([col, row])
        point_class.append(classes[row][col])

    pos_ids = np.random.choice(num_cell, pos_num, replace=True)
    for idx in pos_ids:
        src = np.zeros(mask.shape, np.uint8)
        src[mask == idx] = 255
        dist = cv2.distanceTransform(src, cv2.DIST_L1, 3)
        pos_coord = np.argwhere(dist == dist.max())
        row, col = pos_coord[0][0], pos_coord[0][1]
        
        if is_center:
            point_coord.append([col, row])
            point_class.append(classes[row][col])
        else:
            target = np.zeros(mask.shape, dtype=np.uint8)
            target[row][col] = 255
            target = 100.0 * (target[:,:] > 0)
            target = gaussian_filter(target, sigma=(1, 1), mode='nearest', radius=2)
            #numpy 1.21.5, scipy 1.7,1
            coords = np.argwhere(target != 0)

            normalised = (target - target.min()) / (target.max() - target.min())
            normalised *= 128
            normalised[normalised !=0] += 128

            coords_randm = np.random.randint(len(coords+1),size=len(coords)) 
            for coord_id in coords_randm:
                x, y = coords[coord_id]
                # print(row, col)
                if src[x][y] != 0:
                    point_coord.append([y, x])
                    point_class.append(classes[x][y])
                    break

    # return normalised, point_coord, point_class  
    # return (col, row), point_coord, point_class
    return point_coord, point_class

    



if __name__ == '__main__':
    npy_path = 'monuseg/data/npy/TCGA-2Z-A9J9-01A-01-TS1.npy'
    mask_path = 'monuseg/data/masks/TCGA-2Z-A9J9-01A-01-TS1.png'
    point_coord, point_class = CentreExtractor(npy_path)

    print(point_coord)
    print(point_class)

    mask = cv2.imread(mask_path, 0)
    print(mask.shape)

    for coord in point_coord:
        cv2.circle(mask, coord, 1, 128, 1)

    cv2.imwrite('point_visual.png', mask)


    tmp, point_coord, point_class = CNPS(npy_path)

    print(point_coord)
    print(point_class)

    mask = cv2.imread(mask_path, 0)
    # print(mask.shape)
    # print(tmp.astype(np.uint8))
    mask += tmp.astype(np.uint8)
    # print(set(tmp.flatten()))
    # for coord in point_coord:
    #     cv2.circle(mask, coord, 1, 128, 2)
    
    # cv2.circle(mask, tmp, 1, 128, 1)

    cv2.imwrite('point_visual2.png', mask)






