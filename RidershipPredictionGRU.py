#Neural Net for 
#Original Source below, modified to GRU and adapted for data
#https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import GRU
from keras.layers import Dropout
from keras.layers import Activation
from keras import optimizers
from keras.models import load_model
from keras import regularizers

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

from numpy.random import seed
seed(1066)
from tensorflow import set_random_seed
set_random_seed(1204)

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# load dataset
dataset = read_csv('MOA_numeric.csv', header=0, index_col=0)
values = dataset.values
ncols = 92 #number of columns in dataset
# ensure all data is float
values = values.astype('float')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

# split into train and test sets
values = reframed.values

train = values[:1011, :]
test = values[1011:, :]

rmse_3 = []
mae_3 = []
mape_3 = []

#It varies with each run, even with set seed.

for x in range(1,2):
    train_X = concatenate((train[:, 1:ncols],train[:,(ncols+1):]), axis = 1)
    train_y = train[:, ncols]
    test_X = concatenate((test[:, 1:ncols],test[:, (ncols+1):]), axis= 1)
    test_y = test[:, ncols]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    # design network
    model = Sequential()
    # model.add(LSTM(i, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(GRU(32,kernel_initializer='lecun_uniform',recurrent_dropout = 0.2,return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2]))) #return_sequences=True,
    model.add(Dropout(0.2))
    model.add(GRU(32,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mae', optimizer='adam', metrics = ["mse",'mae']) #['mse','mae']
    #model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')
    # fit network
    history = model.fit(train_X, train_y, epochs=150, batch_size=114, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    
    #plot history
#    pyplot.figure(x)
#    pyplot.plot(history.history['loss'], label='train')
#    pyplot.plot(history.history['val_loss'], label='test')
#    pyplot.legend()
#    pyplot.show()
#    pyplot.savefig('GRU.png')

    
    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast

    inv_yhat = concatenate((yhat, test_X[:, 0:(ncols-2)]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    ################################################
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 0:(ncols-2)]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    mae = mean_absolute_error(inv_y, inv_yhat)
    mape = mean_absolute_percentage_error(inv_y, inv_yhat)
    rmse_3.append(rmse)
    mae_3.append(mae)
    mape_3.append(mape)
print('Test Median RMSE: %.3f' % np.median(rmse_3))
print('Test RMSE Last Run: %.3f' % rmse) #np.median(rmse_3)
print('Test MAE: %.3f' % np.median(mae_3))
print('Test MAPE: %.3f' % np.median(mape_3))

yhat2 = model.predict(train_X)
train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))
#invert scaling for forecast
# 
inv_yhat2 = concatenate((yhat2, train_X[:, 0:(ncols-2)]), axis=1)
inv_yhat2 = scaler.inverse_transform(inv_yhat2) 
inv_yhat2 = inv_yhat2[:,0]
# ################################################
#invert scaling for actual
train_y = train_y.reshape((len(train_y), 1))
inv_y2 = concatenate((train_y, train_X[:,0:(ncols-2)]), axis=1) # 
inv_y2 = scaler.inverse_transform(inv_y2)
inv_y2 = inv_y2[:,0]
######calculate RMSE
rmse2 = sqrt(mean_squared_error(inv_y2, inv_yhat2))
mae2 = mean_absolute_error(inv_y2, inv_yhat2)
mape2 = mean_absolute_percentage_error(inv_y2, inv_yhat2)
print('Train RMSE: %.3f' % (rmse2))

#copy forecasts externally, note this will save the last run only, check it!
# np.savetxt("GRUtest.csv", inv_yhat, delimiter=",")

#model.save("GRUmodel.h5")


#This section for copying predictions on training data (for ensembling, etc)
# =============================================================================
# yhat2 = model.predict(train_X)
# train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))
# # invert scaling for forecast
# 
# inv_yhat2 = concatenate((yhat2, train_X[:, 0:95]), axis=1)
# inv_yhat2 = scaler.inverse_transform(inv_yhat2) 
# inv_yhat2 = inv_yhat2[:,0]
# ################################################
# # invert scaling for actual
# train_y = train_y.reshape((len(train_y), 1))
# inv_y2 = concatenate((train_y, train_X[:, 0:95]), axis=1)
# inv_y2 = scaler.inverse_transform(inv_y2)
# inv_y2 = inv_y2[:,0]
# # calculate RMSE
# rmse2 = sqrt(mean_squared_error(inv_y2, inv_yhat2))
# mae2 = mean_absolute_error(inv_y2, inv_yhat2)
# mape2 = mean_absolute_percentage_error(inv_y2, inv_yhat2)
# np.savetxt("GRUtrain.csv", inv_yhat2, delimiter=",")
# =============================================================================
