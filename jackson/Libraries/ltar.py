from os import P_WAIT
from statsmodels.tsa.api import VAR
import scipy.fft as sfft
import pywt
import numpy as np
import pandas as pd

def diff(data_tensor, interval = 1):
    shape = data_tensor.shape

    if len(shape) != 3:
        raise ValueError(f"{len(train.shape)} is in invalid tensor order. Only 3rd order tensors are valid with this class")

    diff = np.zeros((shape[0] - interval, shape[1], shape[2]))
    for i in range(interval, shape[0] - interval + 1):
        diff[i - interval] = data_tensor[i] - data_tensor[i - interval]
    return diff

def invert_diff(diff_tensor, y0_tensor, interval = 1):
    shape = diff_tensor.shape
    result = diff_tensor.copy()

    if len(shape) != 3:
        raise ValueError(f"{len(train.shape)} is in invalid tensor order. Only 3rd order tensors are valid with this class")

    if len(y0_tensor.shape) != 3:
        raise ValueError(f"{len(y0_tensor.shape)} is in invalid tensor order. Only 3rd order tensors are valid with this class")

    for i in range(shape[0]):
        accum = 0
        for j in range(i // interval):
            accum += diff_tensor[i - (j+1)*interval]
        result[i] += y0_tensor[-interval + i % interval] + accum
    return result

class LTAR():
    def __init__(self, train):

        if len(train.shape) != 3:
            raise ValueError(f"{len(train.shape)} is in invalid tensor order. Only 3rd order tensors are valid with this class")
        
        self.matrix_shape = train.shape[1:3]
        self.train = train
        self.transformation = ""
        self.var_fits = []
        self.coefs = []
        self.c = []
        self.p = -1

    def __apply_trans(self, tensor, transformation, axis):            

        if transformation == "dwt":

            # Only allow even axis size
            if tensor.shape[axis] % 2 != 0:
                raise ValueError(f"{tensor.shape[axis]} is not a valid axis size for DWT. Only even sizes are allowed")

            cA,cD = pywt.dwt(tensor, "haar", axis=axis)
            result = np.append(cA, cD, axis=axis)
        elif transformation == "dct":
            result = sfft.dct(tensor, axis=axis)
        elif transformation == "dft":
            raise NotImplementedError()
        else:
            raise ValueError(f"{self.transformation} is not a valid transformation")

        return result

    def __apply_inverse_trans(self, trans_tensor, transformation, axis):

        if transformation == "dwt":
            cA,cD = np.split(trans_tensor, 2, axis=axis)
            result = pywt.idwt(cA,cD, "haar", axis=axis)
        elif transformation == "dct":
            result = sfft.idct(trans_tensor, axis=axis)
        elif transformation == "dft":
            raise NotImplementedError()
        else:
            raise ValueError(f"{self.transformation} is not a valid transformation")

        return result

    def __split_cols_into_model_sets(self, transformed_tensor):
        N = len(transformed_tensor)
        matrix_shape = transformed_tensor[0].shape
        model_sets = np.empty((matrix_shape[1], N, matrix_shape[0]))
        for i in range(matrix_shape[1]):
            for j in range(N):
                model_sets[i][j] = transformed_tensor[j][:,i]
        return model_sets

    def __mul_ten_and_mat(self, tensor, matrix):

        matrix_shape = matrix.shape
        tensor_shape = tensor.shape
        block_matrix = np.zeros((tensor_shape[0]*tensor_shape[1], tensor_shape[0]*tensor_shape[2]))

        # Makes the block vector
        trans_vector = self.__apply_trans(matrix, self.transformation, axis=1)
        block_vector = trans_vector.transpose().reshape(matrix_shape[0]*matrix_shape[1])

        # Makes the block matrix
        transform_tensor = self.__apply_trans(tensor, self.transformation, axis=0)
        l = tensor_shape[1]
        for i in range(tensor_shape[0]):
            c = i * l
            block_matrix[c:c+l,c:c+l] = transform_tensor[i]

        result_block_vector = np.matmul(block_matrix, block_vector)
        result_trans_matrix = result_block_vector.reshape((matrix_shape[1], matrix_shape[0])).transpose()
        result_matrix = self.__apply_inverse_trans(result_trans_matrix, self.transformation, axis=1)

        return result_matrix

    def fit(self, p, transformation = "dct"):

        if p < 1:
            raise ValueError(f"{p} is an invalid lag")
        self.p = p

        self.transformation = transformation
        l_train_tensor = self.__apply_trans(self.train, transformation, 2) # Applies the transformation across the rows

        train_model_sets = self.__split_cols_into_model_sets(l_train_tensor)

        # Fits all of the var models
        fits = []
        for i in range(self.matrix_shape[1]):
            train_df = pd.DataFrame(train_model_sets[i])
            model = VAR(train_df)
            fit = model.fit(p)
            fits.append(fit) 
        self.var_fits = fits

        # Groups all of the coef matrix to coef tensors
        coefs = np.empty((p, self.matrix_shape[1], self.matrix_shape[0], self.matrix_shape[0]))
        c = np.empty(self.matrix_shape)
        for i in range(self.matrix_shape[1]):
            curr_coefs = fits[i].coefs
            for j in range(p):
                coefs[j][i] = curr_coefs[j]

            # Adds onto c
            c[:,i] = fits[i].params[fits[i].params.index == "const"].iloc[0]
        
        # Performs an inverse tranform to all of them
        for i in range(p):
            coefs[i] = self.__apply_inverse_trans(coefs[i], transformation, 0)

        # Performs the inverse transformation to the const matrix
        c = self.__apply_inverse_trans(c, transformation, 1)

        self.coefs = coefs
        self.c = c

    def forecast(self, interval):
        
        if self.p < 1:
            raise RuntimeError("Model is not fitted!")

        p = self.p
        matrix_shape = self.matrix_shape
        forecast_tensor = np.zeros((interval+p, matrix_shape[0], matrix_shape[1]))
        forecast_tensor[:p] = self.train[-p:]
        for i in range(p,interval+p):
            total = np.zeros(matrix_shape)
            for j in range(p):
                total += self.__mul_ten_and_mat(self.coefs[j], forecast_tensor[i-j-1])
            forecast_tensor[i] = total + self.c
        return forecast_tensor[p:]
