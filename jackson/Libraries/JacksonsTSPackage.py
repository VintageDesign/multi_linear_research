# Manipulation
import pandas as pd
import numpy as np
import scipy.fft as sfft
import pywt

# Plots
import matplotlib.pylab as plt
import seaborn as sns

# TS tests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.tsa.stattools as tsa
from statsmodels.tsa.api import VAR
from scipy import stats
from statsmodels.stats.stattools import durbin_watson

####################### DATA MANIPULATION #################################
def fill_missing_dates_with_interpolation(df):
    # Adds the missing dates
    df = df.resample('D').mean()

    # Interpolates the values
    for col in df.columns:
        df[col] = df[col].interpolate()

    return df


def make_view_ts(ts):
    # Makes a data frame just to be able to view
    tsView = ts.copy()
    tsView.insert(0, "Month", ts.index.month, True)
    tsView.insert(0, "Year", ts.index.year, True)
    return tsView


def calc_seasonal_diff(data, interval):
    """Data has to be a dataframe!"""
    shape = data.shape
    diff = np.zeros((shape[0] - interval, shape[1]))
    for i in range(shape[1]):
        for j in range(shape[0] - interval):
            diff[j][i] = data[i][j + interval] - data[i][j]

    return pd.DataFrame(diff)
    

def invert_diff_transformation(df_forecast, df_train, interval = 1):
    """Revert back the differencing to get the forecast to original scale."""
    df_forecast_invert = df_forecast.copy()
    for col in range(len(df_forecast.columns)): 

        forcast_col = df_forecast_invert.columns[col]
        train_col = df_train.columns[col]

        # Does the invert 
        for i in range(len(df_forecast)):

            accum = 0
            for j in range(int(i / interval)):
                accum += df_forecast[forcast_col][i - (j+1)*interval]

            df_forecast_invert[forcast_col][i] += df_train[train_col].iloc[-interval + i % interval] + accum

    return df_forecast_invert


def extract_train(data, N):   
    train = np.zeros((N, len(data[0])))
    for i in range(N):
        train[i] = data[i]
    return train


def extract_test(data, N_train, N_test):
    test = np.zeros((N_test, len(data[0])))
    for i in range(N_test):
        test[i] = data[N_train + i]
    return test


def extract_train_tensor(tensor, N):
    train = np.zeros((N, tensor[0].shape[0], tensor[0].shape[1]))
    for i in range(N):
        train[i] = tensor[i]
    return train


def extract_test_tensor(tensor, N_train, N_test):
    test = np.zeros((N_test, tensor[0].shape[0], tensor[0].shape[1]))
    for i in range(N_test):
        test[i] = tensor[i + N_train]
    return test


def tensor_to_vector(tensor):
    
    p = tensor[0].shape[0] * tensor[0].shape[1]
    N = len(tensor)
    shape = (N, p)
    v = np.zeros(shape)
    for n in range(N):
        v[n, :] = vec(tensor[n])

    return v

def vec(matrix):
    return matrix.flatten('F')


def calc_diff_per_vector(test, results):
    
    shape = test.shape
    errors = pd.DataFrame(index=test.index, columns=test.columns)

    for i in range(shape[0]):
        for j in range(shape[1]):
            errors.iloc[i][j] = test.iloc[i][j] - results.iloc[i][j]

    return errors


def apply_dct_to_tensor(tensor, type = 2):

    N = len(tensor)
    matrix_shape = tensor[0].shape
    dct_tensor = np.empty((N, matrix_shape[0], matrix_shape[1]))

    for i in range(N):
        for j in range(matrix_shape[0]):
            dct_tensor[i][j] = sfft.dct(tensor[i][j], type=type)

    return dct_tensor


def apply_inverse_dct_to_tensor(dct_tensor, type = 2):

    N = len(dct_tensor)
    matrix_shape = dct_tensor[0].shape
    tensor = np.empty((N, matrix_shape[0], matrix_shape[1]))

    for i in range(N):
        for j in range(matrix_shape[0]):
            tensor[i][j] = sfft.idct(dct_tensor[i][j], type=type)

    return tensor
    

# def apply_dft_to_tensor(tensor):

#     N = len(tensor)
#     matrix_shape = tensor[0].shape
#     dft_tensor = np.empty((N, matrix_shape[0], matrix_shape[1]))

#     for i in range(N):
#         for j in range(matrix_shape[0]):
#             dft_tensor[i][j] = sfft.fft(tensor[i][j])

#     return dft_tensor


# def apply_inverse_dft_to_tensor(dft_tensor):

#     N = len(dft_tensor)
#     matrix_shape = dft_tensor[0].shape
#     tensor = np.empty((N, matrix_shape[0], matrix_shape[1]))

#     for i in range(N):
#         for j in range(matrix_shape[0]):
#             tensor[i][j] = sfft.ifft(dft_tensor[i][j])

#     return tensor


def apply_dwt_to_tensor(tensor, wavelet = 'haar'):

    N = len(tensor)
    matrix_shape = tensor[0].shape
    dwt_tensor = np.zeros((N, matrix_shape[0], matrix_shape[1]))

    for i in range(N):
        for j in range(matrix_shape[0]):
            tmp = pywt.dwt(tensor[i][j], wavelet)
            dwt_tensor[i][j] = np.append(tmp[0], tmp[1])
    return dwt_tensor


def apply_inverse_dwt_to_tensor(transformed_tensor, wavelet = 'haar'):

    N = len(transformed_tensor)
    matrix_shape = transformed_tensor[0].shape
    tensor = np.empty((N, matrix_shape[0], matrix_shape[1]))
    half_len = int(matrix_shape[0] / 2)

    for i in range(N):
        for j in range(matrix_shape[0]):
            tmp = (transformed_tensor[i][j][:half_len], transformed_tensor[i][j][half_len:])
            tensor[i][j] = pywt.idwt(tmp[0], tmp[1], wavelet)
    return tensor


def split_cols_into_model_sets(transformed_tensor, N):
    matrix_shape = transformed_tensor[0].shape
    model_sets = np.empty((matrix_shape[1], N, matrix_shape[0]))
    for i in range(matrix_shape[1]):
        for j in range(N):
            model_sets[i][j] = transformed_tensor[j][:,i]
    return model_sets


def collect_result_cols_into_tensor(model_sets, N_test):
    shape = model_sets.shape
    tensor = np.empty((N_test, shape[2], shape[0]))
    for i in range(N_test):
        for j in range(shape[0]):
            tensor[i][:,j] = model_sets[j][i]
    return tensor

####################### PLOTS #################################
def implot_tensor(tensor, fig_dim, cmap='gray'):
    fig, axs = plt.subplots(fig_dim[0], fig_dim[1])
    for i in range(fig_dim[0]):
        for j in range(fig_dim[1]):
            axs[i, j].imshow(tensor[i*fig_dim[1] + j], cmap='gray')
            axs[i, j].axis('off')

def plot_ts(ts, seperate_figsize = (14, 7), stacked_figuresize = (14, 7), colormap="Dark2"):

    # Plots the seprate plots
    ax = ts.plot(colormap='Dark2', figsize=seperate_figsize, subplots = True, layout = (len(ts.columns), 1), sharex = False, sharey = False)
    plt.show()

    # Plots the stacked plot
    ax = ts.plot(colormap='Dark2', figsize=stacked_figuresize)
    plt.show()


def plot_seasonal_plots(ts, figsize=(14, 28)):
    fig, ax = plt.subplots(nrows = len(ts.columns), figsize=figsize)

    # Makes viewts
    tsView = make_view_ts(ts)

    palette = sns.color_palette("rocket", tsView["Year"].nunique())
    ax[0].set_title("Seasonal Plots")
    i = 0
    for col in ts.columns:
        ax0 = sns.lineplot(tsView["Month"], tsView[col], hue=tsView["Year"], palette = palette, ax = ax[i])
        ax0.get_legend().remove()
        i+=1


def draw_trend_and_seasonality_box_plots(ts):

    # Makes a datafame thats used for viewing
    tsView = make_view_ts(ts)

    for col in ts.columns:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 6))

        sns.boxplot(tsView['Year'], ts[col], ax=ax[0])
        ax[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
        sns.boxplot(tsView['Month'], ts[col], ax=ax[1])
        ax[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize = 20, loc='center', fontdict=dict(weight='bold'))


def plot_rolling_mean_and_std(df, mean_roll = 60, std_roll = 20, figsize = (14, 14)):

    # Plots the rolling mean
    df.rolling(mean_roll).mean().plot(subplots=True, layout=(len(df.columns), 1), figsize=figsize, title="Rolling Mean")
    plt.show()

    # plots the rolling std
    df.rolling(std_roll).std().plot(subplots=True, layout=(len(df.columns), 1), figsize=figsize, title="Rolling SD")
    plt.show()


def plot_forecast_vs_actual(test, fc, upper, lower, nobs, show_conf = True):
    fig, axes = plt.subplots(nrows=int(len(test.columns)/2), ncols=2, dpi=150, figsize=(10,10))
    for i, (col,ax) in enumerate(zip(test.columns, axes.flatten())):
        fc[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
        test[col][-nobs:].plot(legend=True, ax=ax)
        ax.set_title(col + ": Forecast vs Actuals")
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)
        if show_conf:
            ax.fill_between(fc.index, lower[col + "_lower"], upper[col + "_upper"], color = 'b', alpha = .2)

    plt.tight_layout()

####################### TESTS AND MODELING #################################
def grangers_causation_matrix(data, variables, maxlag=12, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = tsa.grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)


def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = tsa.adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.") 


def mts_adf(ts):
    # Does the ADF test
    for name, col in ts.iteritems():
        adfuller_test(col, name=col.name)
        print('\n')

def var_order(ts, max_p = 8):

    model = VAR(ts)

    # prints the header
    print(f"    Sequential Ratio Test", "\n", '-'*75)
    print(f"p", ' '*10, "AIC", ' '*10, "BIC", ' '*10, "HQIC", ' '*10, "M", ' '*8, "p-value")

    aic_min = float("inf")
    aic_p = -1
    bic_min = float("inf")
    bic_p = -1
    hqic_min = float("inf")
    hqic_p = -1
    prev_cov = np.linalg.det(model.fit(0).sigma_u)
    for i in range(1, max_p + 1):

        # fits the model
        result = model.fit(i)

        # Test for new min
        if aic_min > result.aic:
            aic_min = result.aic
            aic_p = i
            
        # Test for new min
        if bic_min > result.bic:
            bic_min = result.bic
            bic_p = i
            
        # Test for new min
        if hqic_min > result.hqic:
            hqic_min = result.hqic
            hqic_p = i
        
        curr_cov = np.linalg.det(result.sigma_u)

        # Calculates the test statistic
        T = len(ts.index)
        k = len(ts.columns)
        M = -1*(T - max_p - 1.5 - k*i) * np.log(curr_cov / prev_cov)

        p_value = 1 - stats.chi2.cdf(M, k*k)

        # Sets the previous cov to the current
        prev_cov = curr_cov

            # prints the results only for p > 0
        print('{0:<8}'.format(f'{i}'), '{0:<14}'.format("{:.6f}".format(result.aic)), '{0:<14}'.format("{:.6f}".format(result.bic)), '{0:<15}'.format("{:.6f}".format(result.hqic)), '{0:<13}'.format("{:.6f}".format(M)), '{0:<13}'.format("{:.6f}".format(p_value)))

    print("\n")
    print(f"    AIC  = {aic_p}")
    print(f"    BIC  = {bic_p}")
    print(f"    HQIC = {hqic_p}")

def dw_test(fit, ts):
    print("Durbin Watson Test")
    print('-'*20)

    out = durbin_watson(fit.resid)

    for col, val in zip(ts.columns, out):
        print(col, ':', round(val, 2))

def forecast(fit, train, test, nobs, calc_conf = True):
    
    # Gets the lag order
    lag_order = fit.k_ar

    # Input data
    input_data = train.values[-lag_order:]

    fc = fit.forecast_interval(y = input_data, steps = nobs)

    df = pd.DataFrame(fc[0], index=test.index[-nobs:], columns=test.columns)
    df.columns = test.columns + "_forecast"

    # Makes cd
    if calc_conf:
        df1 = pd.DataFrame(fc[2], index=test.index[-nobs:], columns=test.columns + '_upper')
        df2 = pd.DataFrame(fc[1], index=test.index[-nobs:], columns=test.columns + '_lower')

        # combines
        df = pd.concat([df, df1, df2], axis = 1)
        df = pd.DataFrame(df)
    

    return df

def forecast_by_step(fit, train, test, nobs, calc_conf = True):
    
    # Gets the lag order
    lag_order = fit.k_ar

    # Input data
    input_data = train.values[-lag_order:]

    df = pd.DataFrame(columns=test.columns)
    df.columns = test.columns + "_forecast"

    for i in range(nobs):
        # Forecases
        fc = fit.forecast_interval(y = input_data, steps = 1)

        tmp = pd.DataFrame(fc[0], index=[test.index[i]], columns=df.columns)

        df = df.append(tmp)

        # adds the result onto the data
        input_data = fc[0]

    return df

def forecast_accuracy(forecast, actual):
   print("Results")
   print('-'*70)
   print(' '*10, "ME", ' '*10, "MSE", ' '*10, "MAE", ' '*10, "MAPE")

   for i in range(0, len(forecast.columns)):
      
      currActualCol = actual.columns[i]
      currForecastCol = forecast.columns[i]

      me = np.mean(forecast[currForecastCol] - actual[currActualCol])
      mse = np.mean(np.square(forecast[currForecastCol] - actual[currActualCol]))
      mae = np.mean(np.abs(forecast[currForecastCol] - actual[currActualCol]))
      mape = np.mean(np.abs(forecast[currForecastCol] - actual[currActualCol])/np.abs(actual[currActualCol]))*100

      # prints the row
      print('{0:<7}'.format(currActualCol), '{0:<14}'.format('{:.3f}'.format(me)), '{0:<14}'.format('{:.3f}'.format(mse)), '{0:<14}'.format('{:.3f}'.format(mae)), '{0:<14}'.format('{:.3f}%'.format(mape)))


def calc_mape_per_vector(test, results):
    shape = test.shape
    mape = pd.DataFrame(index=test.index, columns=["MAPE"])

    for i in range(shape[0]):

        test_vec = test.iloc[i].to_numpy()
        result_vec = results.iloc[i].to_numpy()
        error = np.linalg.norm(test_vec - result_vec)/np.linalg.norm(test_vec)

        mape.iloc[i] = error
      
    return mape

def calc_mape_per_matrix(test_tensor, result_tensor):

    N = len(test_tensor)

    mape = pd.DataFrame(index=range(N), columns=["MAPE"])

    for i in range(N):

        test_matrix = test_tensor[i]
        result_matrx = result_tensor[i]
        error = np.linalg.norm(test_matrix - result_matrx)/np.linalg.norm(test_matrix)

        mape.iloc[i] = error

    return mape