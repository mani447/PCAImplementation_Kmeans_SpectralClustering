import numpy as np
from numpy import matlib
import pandas as pd
import math
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import cv2


def LP_Matrix_Evaluate(similarity_matrix):
    diagonal_matrix = np.zeros(similarity_matrix.shape)
    sum_vec = np.sum(similarity_matrix, axis=1)
    np.fill_diagonal(diagonal_matrix, sum_vec)
    return diagonal_matrix - similarity_matrix
    
def Similarity_Val(point1, point2, sigma):
    similarity = math.exp((-1 / (2 * (sigma ** 2))) * (np.linalg.norm(point1 - point2, ord=2) ** 2))
    return similarity
    
def Similarity_Matrix_Evaluate(data_array, sigma):
    [no_of_points, _] = data_array.shape
    similarity_matrix = np.zeros((no_of_points, no_of_points))
    for i in range(0, no_of_points):
        for j in range(i, no_of_points):
            similarity_matrix[i, j] = Similarity_Val(data_array[i], data_array[j], sigma)
            similarity_matrix[j, i] = Similarity_Val(data_array[i], data_array[j], sigma)
    return similarity_matrix
    
    
def Similarity_Matrix_Image_Evaluate(data_array, sigma):
    [no_of_points, _] = data_array.shape
    cross_mul_matrix = data_array@np.transpose(data_array)
    square_matrix = np.power(data_array, 2)
    col_square_matrix = matlib.repmat(square_matrix, 1, no_of_points)
    row_square_matrix = matlib.repmat(np.transpose(square_matrix), no_of_points, 1)
    similarity_matrix = col_square_matrix+row_square_matrix - 2 * cross_mul_matrix
    similarity_matrix = np.exp((-1/(2*(sigma**2)))*similarity_matrix)
    return similarity_matrix


    
class spectral_clustering:
    def __init__(self):
        self.eigen_vectors = None
        self.eigen_values = None
    
    def get_clusters(self, k):
        eigen_mat = self.eigen_vectors[:, 0:k]
        kmeans = KMeans(n_clusters=k)
        clusters = kmeans.fit_predict(eigen_mat)
        return clusters
        
    def fit(self, data_array, sigma):
        similarity_mat = Similarity_Matrix_Evaluate(data_array, sigma)
        laplacian_mat = LP_Matrix_Evaluate(similarity_mat)
        eigen_values, eigen_vectors = np.linalg.eigh(laplacian_mat)
        sort_index = np.argsort(eigen_values)
        self.eigen_vectors = eigen_vectors[:, sort_index]
        self.eigen_values = eigen_values[sort_index]

    def image_fit(self, data_array, sigma):
        similarity_mat = Similarity_Matrix_Image_Evaluate(data_array, sigma)
        laplacian_mat = LP_Matrix_Evaluate(similarity_mat)
        eigen_values, eigen_vectors = np.linalg.eigh(laplacian_mat)
        sort_index = np.argsort(eigen_values)
        self.eigen_vectors = eigen_vectors[:, sort_index]
        self.eigen_values = eigen_values[sort_index]
        

def circs():
    data = np.zeros((100,2))
    y = 0
    i = 0
    while(i<2*math.pi):
        data[y, 0] = math.cos(i)
        data[y, 1] = math.sin(i)
        y = y+1
        i = i+(math.pi/25)
        
    i = 0
    while(i<2*math.pi):
        data[y, 0] = 2*math.cos(i)
        data[y, 1] = 2*math.sin(i)
        y = y+1
        i = i+(math.pi/25)
    
    return data
    

data = circs()

sigmas = [0.01, 0.05, 0.1, 1, 5, 10, 50, 100]
for sigma in sigmas:
    Spectral_Object = spectral_clustering()
    Spectral_Object.fit(data, sigma)
    print("Eigen Vectors of the data are:")
    print(Spectral_Object.eigen_vectors)
    Clusters_Spectral = Spectral_Object.get_clusters(2)
    Kmeans_Object = KMeans(n_clusters=2)
    Clusters_Kmeans = Kmeans_Object.fit_predict(data)
    fig1, ax1 = plt.subplots()
    scatter1 = ax1.scatter(x=data[:, 0], y=data[:,1], c=Clusters_Spectral)
    legend1 = ax1.legend(*scatter1.legend_elements(),
                         loc="upper right", title="Spectral Clusters")
    ax1.add_artist(legend1)
    plt.savefig('Spectral_Clusters_a' + str(sigma) + '.jpg')
    plt.show()

    fig2, ax2 = plt.subplots()
    scatter2 = plt.scatter(x=data[:, 0], y=data[:, 1], c=Clusters_Kmeans)
    legend2 = ax2.legend(*scatter2.legend_elements(),
                         loc="upper right", title="KMeans Clusters")
    ax2.add_artist(legend2)
    plt.savefig('Kmeans_Clusters_a' + str(sigma) + '.jpg')
    plt.show()
    
 
sigmas = [0.701, 0.702, 0.703, 0.704, 0.705, 0.706, 0.707, 0.708, 0.709, 0.71]
image_data = cv2.imread(r"bw.jpg", cv2.IMREAD_GRAYSCALE)
[m, n] = image_data.shape
image_data = np.array(image_data, dtype=float)
image_flattened_data = np.ravel(image_data)
image_flattened_data = np.reshape(image_flattened_data, (image_flattened_data.shape[0], 1))
kmeans = KMeans(n_clusters=2)
Clusters_Kmeans = kmeans.fit_predict(image_flattened_data)
Image_Kmeans = np.uint8(255 * Clusters_Kmeans.reshape([m, n]))
plt.imsave("Kmeans_Clusters_b.jpg", Image_Kmeans)
cv2.imwrite(r"Kmeans_bw.jpg", Image_Kmeans)
for sigma in sigmas:
    Spectral_Object = spectral_clustering()
    Spectral_Object.image_fit(image_flattened_data, sigma)
    Clusters_Spectral = Spectral_Object.get_clusters(2)
    #Clusters_Spectral = np.array([1 if x == 0 else 0 for x in Clusters_Spectral])
    Image_Spectral = np.uint8(255 * Clusters_Spectral.reshape([m, n]))
    plt.imsave("Spectral_Clusters_b" + str(sigma) + ".jpg", Image_Spectral)
    cv2.imwrite(r"Spectral_bw" + str(sigma)+".jpg", Image_Spectral)