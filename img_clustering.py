#  
# image clustering using sklearn
# david magnuson
# 

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

def loadImg(filename):
    # read img as rgb matrix
    data = plt.imread(filename)
    img3d = data[:, :, :3]
    # convert 3D matrix into a 2D matrix and return
    x, y, z = img3d.shape
    img2d = img3d.reshape(x * y, z)
    return img2d, img3d

def useKmean(n_clusters, data):
    return KMeans(n_clusters=n_clusters).fit(data)

def useDbscan(eps, minPts, data):
    return DBSCAN(eps=eps, min_samples=minPts).fit(data)

# MAIN
def main():
    # load image data
    imgFileName, imgFileType = 'earth', 'webp'

    img2d, img3d = loadImg(f'{imgFileName}.{imgFileType}')
    x, y, z = img3d.shape

    # use k-means to cluster the image into n_clusters
    for i in range(2, 5):
        kmeans = useKmean(i, img2d)
        kmeans_labels = kmeans.labels_
        kmeans_centers = kmeans.cluster_centers_
        
        # convert 2D matrix back to 3D matrix with cluster centers as new colors
        img3d_kmeans = kmeans_centers[kmeans_labels].reshape(x, y, 3)

        # display and save updated image
        img3d_kmeans = img3d_kmeans.astype(np.uint8)
        pic = plt.imshow(img3d_kmeans)
        pic.axes.get_xaxis().set_visible(False)
        pic.axes.get_yaxis().set_visible(False)
        plt.title(f'{imgFileName} {str(i)} clusters')
        plt.show()
        plt.savefig(f'{imgFileName}_{str(i)}clusters.jpg', dpi=400)
    
if __name__ == '__main__':
    main()