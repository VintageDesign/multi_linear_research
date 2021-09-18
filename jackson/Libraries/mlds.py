from os import P_WAIT
from statsmodels.tsa.api import VAR
import scipy.fft as sfft
import pywt
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, './')
from lds import LDS

class LMLDS():
    def __init__(self, train):

        if len(train.shape) != 3:
            raise ValueError(f"{len(train.shape)} is in invalid tensor order. Only 3rd order tensors are valid with this class")
        
        self.matrix_shape = train.shape[1:3]
        self.train = train
        self.transformation = ""
        self.lds_fits = []
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

    def fit(self, transformation = "dct"):

        self.transformation = transformation
        l_train_tensor = self.__apply_trans(self.train, transformation, 2) # Applies the transformation across the rows

        train_model_sets = self.__split_cols_into_model_sets(l_train_tensor)

        # Fits all of the var models
        fits = []
        for i in range(self.matrix_shape[1]):
            train_df = train_model_sets[i]
            model = LDS(train_df)
            model.fit()
            fits.append(model) 
        self.lds_fits = fits

    def forecast(self, interval):
        matrix_shape = self.matrix_shape
        l_forecast_tensor = np.zeros((interval, matrix_shape[0], matrix_shape[1]))

        # predict for every lateral slice
        for i in range(self.matrix_shape[1]):
            lat_slice = self.lds_fits[i].forecast(interval)
            l_forecast_tensor[:,:,i] = lat_slice

        # inverse the transformation
        forecast_tensor = self.__apply_inverse_trans(l_forecast_tensor, self.transformation, 2)

        return forecast_tensor

    def single_step_forecast(self, interval, test_tensor):
        
        raise NotImplementedError()
