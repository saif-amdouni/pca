import numpy as np
import matplotlib.pyplot as plt

class pca:
    def __init__(self,n_dim):
        self.n_dim = n_dim
    @staticmethod
    def variance(vector):
        """claculates the variance of a given vector"""
        return np.sum(np.power(vector - np.mean(vector) , 2))/len(vector)
    @staticmethod 
    def covariance(vector1,vector2):
        """calculates the covariance between two vectors"""
        return np.sum((vector1 - np.mean(vector1)) * (vector2 - np.mean(vector2)) )/len(vector1)
    @staticmethod
    def substract_mean(data):
        """substacts the mean of data"""
        return data - np.mean(data,axis=0)
    @staticmethod
    def calcuate_cov_matrix(data):
        n_features = data.shape[1]
        cov_matrix = np.zeros(shape = (n_features,n_features))
        for i in range(n_features):
            for j in range(n_features):
                cov_matrix[i,j] = pca.covariance(data[:,i],data[:,j])
        return cov_matrix
    @staticmethod
    def eigenvalues_eigenvectors(cov):
        return np.linalg.eig(cov)
    @staticmethod
    def project_data_on_pc(data,pc):
        projected_data = []
        for c in pc:
            projected_data.append(np.dot(data,c))
        return np.array(projected_data).T
    
    def reduce_dim(self,data):
        data = pca.substract_mean(data)
        val,vect = pca.eigenvalues_eigenvectors(pca.calcuate_cov_matrix(data))
        ordered_vect = [x for x, _ in sorted(zip(vect.tolist(), val))]
        return pca.project_data_on_pc(data,ordered_vect[0:self.n_dim])
            
if __name__ == "__main__":
    # generate Data 
    slope = 1
    intercept = 0
    interval = 5
    n_points = 100
    sigma = 0.5

    x = np.random.uniform(0,interval,n_points)
    y = slope * x + intercept + np.random.normal(2, sigma, n_points)
    data = np.column_stack((x,y))

    PCA = pca(n_dim=2)
    transformed_data = PCA.reduce_dim(data)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    fig.gca().set(xlim=(-5, 5), ylim=(-5, 5))
    
    axes[0].scatter(data[:,0],data[:,1])
    axes[1].scatter(PCA.reduce_dim(data)[:,0],PCA.reduce_dim(data)[:,1])
    plt.show()