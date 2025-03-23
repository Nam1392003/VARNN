import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from statsmodels.tsa.api import VAR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from optuna.integration import KerasPruningCallback
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import time
from bayes_opt import BayesianOptimization
import random
import torch
import torch.nn as nn
import torch.optim as optim
#Biến đổi cột Date Time
def tranformation(data_target):
    if data_target is None:
        return None
    else:
        # Kiểm tra xem cột đầu tiên có phải dạng datetime hay không
        if 'Date Time' in data_target.columns:
            data_target['Date Time'] = pd.to_datetime(data_target['Date Time'], format="%d.%m.%Y %H:%M:%S")
            data_target.set_index('Date Time', inplace=True, drop=True)
            daily_target = data_target.resample('1D').mean()
            daily_target.fillna(daily_target.rolling(window=10, min_periods=5).mean(), inplace=True)
        else:
            data_target.set_index(data_target.columns[0], inplace=True)
            daily_target=data_target.apply(lambda col: col.fillna(col.mean()))
        return daily_target

#them du lieu
def add_gaussian_noise(data, mean,stddev):
    # Gaussian noise generation
    noise = np.random.normal(mean, stddev, data.shape)

    # Adding noise to the original time series
    noisy_series = data + noise

    return noisy_series
def adf_test(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput
     
def kiemdinh(df,target):
    dfoutput= adf_test(df[target])
    if dfoutput["p-value"] < 0.05:
        str=f"=>{target} là chuỗi dừng"
        k = True
    else:
        str = f"=> {target} Không phải chuỗi dừng"
        k = False
    return str,k

def chuanhoachuoidung(data):
    for i in data.columns:
        while True:
            # Kiểm định chuỗi
            str_result, is_stationary = kiemdinh(data, i)
            if is_stationary:  # Chuỗi đã dừng
                break
            else:  # Chuỗi không dừng
                # Xử lý theo loại chuỗi không dừng
                if is_trend(data[i]):  # Chuỗi không dừng do xu hướng + nhiễu
                    trend_series = data[i]
                    data[i] = np.diff(trend_series, prepend=trend_series.iloc[0])  # Sai phân
                elif is_exponential_growth(data[i]):  # Chuỗi không dừng do tăng trưởng hàm mũ
                    exp_series = data[i]
                    data[i] = np.diff(np.log(exp_series), prepend=np.log(exp_series.iloc[0]))  # Log-sai phân
                else:  # Trường hợp khác, dùng sai phân thông thường
                    data[i] = data[i].diff()
                data = data.dropna(subset=[i])
    return data

def is_trend(series):
    from scipy.stats import linregress
    x = np.arange(len(series))
    y = series.values
    slope, _, _, p_value, _ = linregress(x, y)
    return p_value < 0.05 and abs(slope) > 0.001  # Kiểm tra ý nghĩa của độ dốc

def is_exponential_growth(series):
    try:
        log_series = np.log(series.replace(0, np.nan).dropna())  # Log của chuỗi (loại bỏ giá trị 0)
        return is_trend(log_series)  # Kiểm tra xu hướng tuyến tính trên log
    except:
        return False
    
def Min_max_scaler(data,min,max):
    # Khởi tạo scaler
    scaler = MinMaxScaler(feature_range=(min, max))

    # Áp dụng scaler cho tất cả các cột số
    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return [scaled_data,scaler]

def Zero_mean_scaler(data):
    # Tính mean của dữ liệu
    mean_data = np.mean(data)
    # Chuẩn hóa dữ liệu về zero mean
    data_zero_mean = data - mean_data 
    return data_zero_mean, mean_data

def Inverse_zero_mean(scaled_data, mean_data):
    if isinstance(mean_data, (np.ndarray, pd.Series)):
        mean_data = np.array(mean_data)
    
    # Kiểm tra và broadcast nếu cần
    if scaled_data.shape != mean_data.shape:
        mean_data = np.broadcast_to(mean_data, scaled_data.shape)
    
    # Đảo ngược chuẩn hóa
    original_data = scaled_data + mean_data
    return original_data
def time_warping(data, warp_factor=1.1):
    """
    Biến dạng thời gian bằng cách nội suy.
    warp_factor > 1: Giãn thời gian.
    warp_factor < 1: Nén thời gian.
    """
    original_indices = np.arange(data.shape[0])
    new_indices = np.linspace(0, data.shape[0]-1, int(data.shape[0] * warp_factor))
    f = interp1d(original_indices, data, axis=0, fill_value="extrapolate")
    warped_data = f(new_indices)
    return warped_data
def devide_train_test(data,ratio):
    if data is None: 
        pass
    else:
        train_data, test_data = train_test_split(data, test_size=ratio, shuffle=False)
        return [train_data,test_data]

def find_lag(train_data):
    # Tạo danh sách để lưu các giá trị AIC cho mỗi độ trễ
    lag_aic_values = []

    # Khởi tạo các giá trị lag để thử nghiệm (ở đây từ 1 đến 15)
    lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,18,19,20,21,22,23,24,25]

    # Chạy vòng lặp để tính AIC cho từng độ trễ
    for lag in lags:
        model = VAR(train_data)
        results = model.fit(lag)
        lag_aic_values.append((lag, results.aic))
    # Chuyển danh sách thành DataFrame để so sánh
    aic_df = pd.DataFrame(lag_aic_values, columns=['Lag', 'AIC'])

    # Tìm độ trễ tốt nhất dựa trên AIC
    best_aic_lag = aic_df.loc[aic_df['AIC'].idxmin()]
    lag = int(best_aic_lag["Lag"])
    return lag

def train_VAR(train_data,test_data,lag):
    # Huấn luyện mô hình VAR
    start_train_time = time.time()
    var_model = VAR(train_data)
    var_result = var_model.fit(maxlags=lag)
    pred_var = var_result.fittedvalues
    y_test_pre = var_result.forecast(train_data.values[-lag:], steps=len(test_data))

    # Tính toán các chỉ số đánh giá
    y_test = test_data.values  # Sử dụng toàn bộ tập kiểm tra


    #Du doan
    forecast = var_result.forecast(y_test, steps=1)
    # Tính MSE, MAE, MAPE giữa dự đoán và giá trị thực tế
    mse_var = mean_squared_error(y_test, y_test_pre)
    mae_var = mean_absolute_error(y_test, y_test_pre)
    rmse_var = np.sqrt(mse_var)
    mean_y_test = np.mean(y_test)
    cv_rmse_var = (rmse_var / mean_y_test) * 100
    end_train_time = time.time()
    test_time=end_train_time-start_train_time
    return [forecast,y_test,y_test_pre,mse_var,mae_var,cv_rmse_var,rmse_var,test_time,pred_var]

def prepare_data_for_ffnn(train_data,test_data,lag):
    X_train = np.array([train_data.values[i:i+lag] for i in range(len(train_data)-lag)])
    forecast,y_test,y_test_pre,mse_var,mae_var,cv_rmse_var,rmse_var,test_time,pred_var = train_VAR(train_data,test_data,lag)
    y_train=pred_var
    return [X_train,y_train]

    
# Tìm tham số tối ưu
def devide_train_val(train_data,test_data,lag,ratio):
    X_train, y_train=prepare_data_for_ffnn(train_data,test_data,lag)
    # Chia tập dữ liệu huấn luyện thành tập huấn luyện thực sự và tập validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=ratio, random_state=42)
    return [X_train_split, X_val_split, y_train_split, y_val_split]

def find_parameter_for_ffnn_optuna(train_data,test_data, ratio_train_val,lag):
    # Hàm mục tiêu cho Optuna
    def objective(trial):
        # Các tham số cho Optuna tối ưu
        epochs = trial.suggest_int("epochs", 50, 300)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        lstm_units = trial.suggest_int("lstm_units", 10, 100)
        learning_rate=trial.suggest_categorical("learning_rate",[0.0001,0.001,0.01,0.1])

        # Khởi tạo mô hình 
        model = Sequential()
        model.add(LSTM(lstm_units, activation='sigmoid', input_shape=(lag, train_data.shape[1])))
        model.add(Dense(train_data.shape[1]))
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')

        X_train_split, X_val_split, y_train_split, y_val_split=devide_train_val(train_data,test_data,lag,ratio_train_val)
        # Huấn luyện mô hình với tập validation

        model.fit(X_train_split, y_train_split, epochs=epochs, batch_size=batch_size, verbose=0, 
              validation_data=(X_val_split, y_val_split))

        # Tính toán loss trên tập validation
        val_loss = model.evaluate(X_val_split, y_val_split, verbose=0)
        return val_loss
    # Khởi tạo và chạy Optuna để tối ưu epochs và batch_size
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)  # Thực hiện 20 lần thử nghiệm
    
        # Kết quả tối ưu
    lstm_unit=study.best_params["lstm_units"]
    epochs=study.best_params["epochs"]
    batch_size=study.best_params["batch_size"]
    learning_rate=study.best_params["learning_rate"]
    return [lstm_unit,epochs,batch_size,learning_rate]
def find_parameter_for_ffnn_bayesian(train_data, test_data, ratio_train_val, lag):
    def objective(epochs, batch_size_idx, lstm_units, learning_rate_idx):
        epochs = int(epochs)
        lstm_units = int(lstm_units)
        
        batch_size_list = [16, 32, 64, 128]
        learning_rate_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]
        
        batch_size = batch_size_list[int(batch_size_idx)]
        learning_rate = learning_rate_list[int(learning_rate_idx)]
        
        # Khởi tạo mô hình
        model = Sequential()
        model.add(LSTM(lstm_units, activation='sigmoid', input_shape=(lag, train_data.shape[1])))
        model.add(Dense(train_data.shape[1]))
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        
        X_train_split, X_val_split, y_train_split, y_val_split = devide_train_val(train_data, test_data, lag, ratio_train_val)
        
        # Huấn luyện mô hình với tập validation
        model.fit(X_train_split, y_train_split, epochs=epochs, batch_size=batch_size, verbose=0, 
                  validation_data=(X_val_split, y_val_split))
        
        # Tính toán loss trên tập validation
        val_loss = model.evaluate(X_val_split, y_val_split, verbose=0)
        return -val_loss  # Đảo ngược giá trị vì Bayesian Optimization tối đa hóa hàm mục tiêu
    
    # Định nghĩa không gian tìm kiếm tham số
    pbounds = {
        "epochs": (50, 300),
        "batch_size_idx": (0, 3),  # Chỉ số của batch_size_list
        "lstm_units": (10, 100),
        "learning_rate_idx": (0, 4)  # Chỉ số của learning_rate_list
    }
    
    optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)
    optimizer.maximize(init_points=5, n_iter=15)
    
    best_params = optimizer.max["params"]
    
    best_batch_size = [16, 32, 64, 128][int(best_params["batch_size_idx"])]
    best_learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1][int(best_params["learning_rate_idx"])]
    
    return [int(best_params["lstm_units"]), int(best_params["epochs"]), best_batch_size, best_learning_rate]

def find_parameter_for_ffnn_random(train_data, test_data, ratio_train_val, lag):
    n_trials=20
    def objective(epochs, batch_size, lstm_units, learning_rate):
        epochs = int(epochs)
        batch_size = int(batch_size)
        lstm_units = int(lstm_units)
        
        # Khởi tạo mô hình
        model = Sequential()
        model.add(LSTM(lstm_units, activation='sigmoid', input_shape=(lag, train_data.shape[1])))
        model.add(Dense(train_data.shape[1]))
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        
        X_train_split, X_val_split, y_train_split, y_val_split = devide_train_val(train_data, test_data, lag, ratio_train_val)
        
        # Huấn luyện mô hình với tập validation
        model.fit(X_train_split, y_train_split, epochs=epochs, batch_size=batch_size, verbose=0, 
                  validation_data=(X_val_split, y_val_split))
        
        # Tính toán loss trên tập validation
        val_loss = model.evaluate(X_val_split, y_val_split, verbose=0)
        return val_loss  # Không cần đảo ngược vì Random Search tối thiểu hóa hàm mục tiêu
    
    # Định nghĩa không gian tìm kiếm tham số
    epochs_range = [50, 100, 150, 200, 250, 300]
    batch_size_range = [16, 32, 64, 128]
    lstm_units_range = [10, 20, 50, 100]
    learning_rate_options = [0.0001, 0.001, 0.01, 0.1]
    
    best_loss = float("inf")
    best_params = None
    
    for _ in range(n_trials):
        epochs = random.choice(epochs_range)
        batch_size = random.choice(batch_size_range)
        lstm_units = random.choice(lstm_units_range)
        learning_rate = random.choice(learning_rate_options)
        
        loss = objective(epochs, batch_size, lstm_units, learning_rate)
        
        if loss < best_loss:
            best_loss = loss
            best_params = [lstm_units, epochs, batch_size, learning_rate]
    
    return best_params
def train_varnn(train_data,test_data, lag,epochs,lstm_unit,batch_size,learning_rate):
    
    varnn_model = Sequential()
    varnn_model.add(LSTM(lstm_unit, activation='sigmoid', input_shape=(lag, train_data.shape[1])))
    varnn_model.add(Dense(train_data.shape[1]))
    optimizer = Adam(learning_rate=learning_rate)
    varnn_model.compile(optimizer=optimizer, loss='mse')

    X_train,y_train=prepare_data_for_ffnn(train_data,test_data,lag)
    history=varnn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2, verbose=1)

    X_test = np.array([test_data.values[i:i+lag] for i in range(len(test_data)-lag)])
    y_test = test_data.values[lag:]

    start_train_time = time.time()
    y_test_pre = varnn_model.predict(X_test)
    latest_data = y_test[-lag:].reshape(1, lag, y_test.shape[1])
    latest_prediction = varnn_model.predict(latest_data)
    # Tính toán các chỉ số đánh giá
    mse_varnn = mean_squared_error(y_test, y_test_pre)
    mae_varnn = mean_absolute_error(y_test, y_test_pre)
    #mape_varnn = mean_absolute_percentage_error(y_test, y_test_pre)
    rmse_varnn = np.sqrt(mse_varnn)
    mean_y_test = np.mean(y_test)
    cv_rmse_varnn = (rmse_varnn / mean_y_test) * 100
    end_train_time = time.time()
    test_time=end_train_time-start_train_time
    return [history,latest_prediction,y_test,y_test_pre,mse_varnn, mae_varnn, cv_rmse_varnn,rmse_varnn,test_time]

def train_ffnn(train_data,test_data, lag,epochs,lstm_unit,batch_size,learning_rate):
    ffnn_model = Sequential()
    ffnn_model.add(LSTM(lstm_unit, activation='relu', input_shape=(lag, train_data.shape[1])))
    ffnn_model.add(Dense(train_data.shape[1]))
    optimizer = Adam(learning_rate=learning_rate)
    ffnn_model.compile(optimizer=optimizer, loss='mse')


    X_train = np.array([train_data.values[i:i+lag] for i in range(len(train_data)-lag)])
    y_train = train_data.values[lag:]
    history=ffnn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,verbose=1)

    X_test = np.array([test_data.values[i:i+lag] for i in range(len(test_data)-lag)])
    y_test = test_data.values[lag:]

    start_train_time = time.time()
    y_test_pre = ffnn_model.predict(X_test)
    latest_data = y_test[-lag:].reshape(1, lag, y_test.shape[1])
    latest_prediction = ffnn_model.predict(latest_data)
    # Tính toán các chỉ số đánh giá
    mse_ffnn = mean_squared_error(y_test, y_test_pre)
    mae_ffnn = mean_absolute_error(y_test, y_test_pre)
    
    rmse_ffnn = np.sqrt(mse_ffnn)
    mean_y_test = np.mean(y_test)
    cv_rmse_ffnn = (rmse_ffnn / mean_y_test) * 100
    end_train_time = time.time()
    test_time=end_train_time-start_train_time
    return [history,latest_prediction,y_test,y_test_pre,mse_ffnn, mae_ffnn, cv_rmse_ffnn,rmse_ffnn,test_time]
class DeepVAR(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(DeepVAR, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def train_model(X_train, y_train, X_val, y_val, input_dim, hidden_dim, num_layers, epochs, lr):
    model = DeepVAR(input_dim, hidden_dim, num_layers, input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Chuyển DataFrame thành NumPy array trước khi tạo tensor
    X_train = X_train.to_numpy() if hasattr(X_train, 'to_numpy') else X_train
    y_train = y_train.to_numpy() if hasattr(y_train, 'to_numpy') else y_train
    X_val = X_val.to_numpy() if hasattr(X_val, 'to_numpy') else X_val
    y_val = y_val.to_numpy() if hasattr(y_val, 'to_numpy') else y_val
    

    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32)
    X_val_torch = torch.tensor(X_val, dtype=torch.float32)
    y_val_torch = torch.tensor(y_val, dtype=torch.float32)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_torch)
        loss = criterion(output, y_train_torch)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_output = model(X_val_torch)
        val_loss = criterion(val_output, y_val_torch).item()
    
    return val_loss, model

def find_parameter_for_deepvar(train_data, test_data, ratio_train_val, lag):
    def objective(trial):
        hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        lr = trial.suggest_categorical("lr", [0.0001, 0.001, 0.01, 0.1])
        epochs = trial.suggest_int("epochs", 50, 300)
        
        X_train_split, X_val_split, y_train_split, y_val_split = devide_train_val(train_data, test_data, lag, ratio_train_val)
        val_loss, _ = train_model(X_train_split, y_train_split, X_val_split, y_val_split, X_train_split.shape[2], hidden_dim, num_layers, epochs, lr)
        return val_loss
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    
    best_params = study.best_params
    return [best_params["hidden_dim"], best_params["num_layers"], best_params["epochs"], best_params["lr"]]

def train_deepvar(train_data, test_data, lag, hidden_dim, num_layers, epochs, lr):
    X_train,y_train=prepare_data_for_ffnn(train_data,test_data,lag)
    
    X_test = np.array([test_data.values[i:i+lag] for i in range(len(test_data)-lag)])
    y_test = test_data.values[lag:]
    
    start_train_time = time.time()
    _, deepvar_model = train_model(X_train, y_train, X_test, y_test, X_train.shape[2], hidden_dim, num_layers, epochs, lr)
    
    deepvar_model.eval()
    with torch.no_grad():
        X_test_torch = torch.tensor(X_test, dtype=torch.float32)
        y_test_pre = deepvar_model(X_test_torch).numpy()
        
        latest_data = torch.tensor(X_test[-1].reshape(1, lag, X_test.shape[2]), dtype=torch.float32)
        latest_prediction = deepvar_model(latest_data).numpy()
    mse = mean_squared_error(y_test, y_test_pre)
    mae = mean_absolute_error(y_test, y_test_pre)
    rmse = np.sqrt(mse)
    mean_y_test = np.mean(y_test)
    cv_rmse = (rmse / mean_y_test) * 100
    end_train_time = time.time()
    test_time = end_train_time - start_train_time
    
    return [latest_prediction, y_test, y_test_pre, mse, mae, cv_rmse, rmse, test_time]
