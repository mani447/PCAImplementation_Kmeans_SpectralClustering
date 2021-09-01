import numpy as np
import pandas as pd
import cvxopt

class standard_scaling:
    def __init__(self):
        self.std = None
        self.mean = None

    def transform_and_fit(self, data):
        self.std = np.std(data, axis=0)
        self.mean = np.mean(data, axis=0)
        transformed_data = np.subtract(data, self.mean)
        transformed_data = np.divide(transformed_data, self.std)
        return transformed_data

    def transform(self, data):
        transformed_data = np.subtract(data, self.mean)
        transformed_data = np.divide(transformed_data, self.std)
        return transformed_data
    
class PCA_Algorithm:
    def __init__(self):
        self.eigen_vectors = None
        self.eigen_values = None

    def fit(self, data):
        # Zeroing out the mean
        data = np.transpose(np.subtract(np.transpose(data), np.mean(data, axis=1)))
        # Calculating the Covariance Matrix
        covariance_matrix = data @ np.transpose(data)
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
        sort_index = np.argsort(eigen_values)[::-1]
        self.eigen_values = np.sort(eigen_values)[::-1]
        self.eigen_vectors = eigen_vectors[:, sort_index]

    def transform_and_fit(self, data, k):
        # Zeroing out the mean
        data = np.transpose(np.subtract(np.transpose(data), np.mean(data, axis=1)))
        # Calculating the Covariance Matrix
        covariance_matrix = data @ np.transpose(data)
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
        sort_index = np.argsort(eigen_values)[::-1]
        self.eigen_values = np.sort(eigen_values)[::-1]
        self.eigen_vectors = eigen_vectors[:, sort_index]
        transformed_data = np.transpose(self.eigen_vectors[:, 0:k]) @ data
        return transformed_data

    def transform(self, data, k):
        transformed_data = np.transpose(self.eigen_vectors[:, 0:k]) @ data
        return transformed_data
        

def SVM_Slack(input_data, output_data, c_constant):
    if input_data.shape[0] != output_data.shape[0]:
        raise ValueError("Input and Output data size Mismatch")
    dataset_size = input_data.shape[0]
    inp_size = input_data.shape[1]
    conditional_matrix = np.zeros((dataset_size * 2, inp_size + dataset_size))
    h = np.zeros((dataset_size * 2, 1))
    #print(dataset_size, inp_size, conditional_matrix.shape)
    for j in range(0, dataset_size):
        conditional_matrix[j, 0:inp_size] = -1 * input_data[j] * output_data[j]
        conditional_matrix[j, inp_size + j] = -1
        h[j] = -1
    for j in range(dataset_size, 2 * dataset_size):
        conditional_matrix[j, inp_size + j - dataset_size] = -1
    p = np.zeros((inp_size + dataset_size, inp_size + dataset_size))
    for j in range(0, inp_size - 1):
        p[j][j] = 1
    q = np.zeros((inp_size + dataset_size, 1))
    q[inp_size:] = c_constant
    cvxopt.solvers.options['show_progress'] = False
    p = cvxopt.matrix(p, tc='d')
    q = cvxopt.matrix(q, tc='d')
    g = cvxopt.matrix(conditional_matrix, tc='d')
    h = cvxopt.matrix(h, tc='d')
    sol = cvxopt.solvers.qp(p, q, g, h)
    return sol['x']

    
def accuracy(input_data, output_data, w_vec):
    estimated_output = np.matmul(w_vec, np.transpose(input_data))
    margin_vector = np.transpose(np.multiply(estimated_output, np.transpose(output_data)))
    acc = 100 - ((len(np.where(margin_vector <= 0)[0]) / len(input_data)) * 100)
    return acc
    
    
def homogenize_data(data):
    temp = np.ones((data.shape[0], data.shape[1] + 1))
    temp[:, :-1] = data
    return temp
    
    
def find_support_vectors(input_data, output_data, w_vec):
    estimated_output = np.matmul(w_vec, np.transpose(input_data))
    margin_vector = np.transpose(np.multiply(estimated_output, np.transpose(output_data)))
    support_vectors = list()
    for j in range(0, len(margin_vector)):
        if (margin_vector[j] > 0.999) & (margin_vector[j] < 1.001):
            support_vectors.append(input_data[j][:4])
    support_vectors = np.array(support_vectors)
    return support_vectors


def train_validate_test_split(df, train_percent=.6, validate_percent=.3, seed=None):
    np.random.seed(seed)
    #print(df.index)
    perm = np.array([i for i in range(0, df.shape[0])])
    #print(perm)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test
    
def load_data(file_path):
    with open(file_path, 'rb') as input_file:
        data_frame = pd.read_csv(input_file, sep=',', header=None)
    
    train, validate, test = train_validate_test_split(data_frame)
    train_input_data = np.array(train[train.columns[0:-1]])
    train_output_data = np.array(train[train.columns[-1]])
    valid_input_data = np.array(validate[validate.columns[0:-1]])
    valid_output_data = np.array(validate[validate.columns[-1]])
    test_input_data = np.array(test[test.columns[0:-1]])
    test_output_data = np.array(test[test.columns[-1]])
    return train_input_data, train_output_data, valid_input_data, valid_output_data, test_input_data, test_output_data
    
    
train_data, train_output, val_data, val_output, test_data, test_output= load_data(r"madelon.data")
hom_train_data = homogenize_data(train_data)
hom_val_data = homogenize_data(val_data)
hom_test_data = homogenize_data(test_data)
train_svm_error = []
test_svm_error = []
val_svm_error = []
[_, num_features] = hom_train_data.shape

for i in range(0, 4):
    c = np.power(10, i)
    weight_vector = SVM_Slack(hom_train_data, train_output, c)
    weight_vector_list = [x for x in weight_vector]
    weight_vector = np.array(weight_vector_list)
    train_svm_error.append(100 - accuracy(hom_train_data, train_output, weight_vector[0:num_features]))
    val_svm_error.append(100 - accuracy(hom_val_data, val_output, weight_vector[0:num_features]))
    test_svm_error.append(100 - accuracy(hom_test_data, test_output, weight_vector[0:num_features]))

print("SVM error on train data is:")   
print(train_svm_error)
print("SVM error on val data is:")
print(val_svm_error)
print("SVM error on test data is:")
print(test_svm_error)

train_error_matrix = np.zeros((6, 4))
val_error_matrix = np.zeros((6, 4))
test_error_matrix = np.zeros((6, 4))

scaler = standard_scaling()
train_data = scaler.transform_and_fit(train_data)
val_data = scaler.transform(val_data)
test_data = scaler.transform(test_data)

pca = PCA_Algorithm()
pca.fit(np.transpose(train_data))
print("Top 6 Eigen Values:")
print(pca.eigen_values[0:6])

for k in range(1, 7):
    kd_train_data = np.transpose(pca.transform(np.transpose(train_data), k))
    kd_train_data = homogenize_data(kd_train_data)
    kd_val_data = np.transpose(pca.transform(np.transpose(val_data), k))
    kd_val_data = homogenize_data(kd_val_data)
    kd_test_data = np.transpose(pca.transform(np.transpose(test_data), k))
    kd_test_data = homogenize_data(kd_test_data)
    num_features = kd_train_data.shape[1]
    for i in range(0, 4):
        c = np.power(10, i)
        weight_vector = SVM_Slack(kd_train_data, train_output, c)
        weight_vector_list = [x for x in weight_vector]
        weight_vector = np.array(weight_vector_list)
        train_error_matrix[k - 1, i] = 100 - accuracy(kd_train_data, train_output, weight_vector[0:num_features])
        val_error_matrix[k - 1, i] = 100 - accuracy(kd_val_data, val_output, weight_vector[0:num_features])
        test_error_matrix[k - 1, i] = 100 - accuracy(kd_test_data, test_output, weight_vector[0:num_features])
        
print("Train Error Matrix k vs c:")
print(train_error_matrix)
print("Validation Error Matrix k vs c:")
print(val_error_matrix)
print("Test Error Matrix k vs c:")
print(test_error_matrix)