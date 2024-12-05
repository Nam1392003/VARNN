import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from pandasai.responses.response_parser import ResponseParser
import tensorflow as tf
import varnn as v
import json
import time
import plotly.graph_objs as go
# Thiết lập tiêu đề ứng dụng
st.title("Ứng dụng Dự đoán Chuỗi Thời Gian đa biến")

# Sidebar
st.sidebar.header("Thiết lập cấu hình", divider='rainbow')
class OutputParser(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)
    def parse(self, result):
        if result['type'] == "dataframe":
            st.dataframe(result['value'])
        elif result['type'] == 'plot':
            st.image(result["value"])
        else:
            st.write(result['value'])
        return


def get_model():
    st.sidebar.header("Chọn mô hình", divider='rainbow')
    model = st.sidebar.selectbox("Chọn mô hình:",
                           ["VARNN",
                            "VAR",
                            "FFNN"]
                           )
    return model
def load_model_to_predict(model):
    if model=="VARNN":
        model = tf.keras.models.load_model("varnn_model.keras")
    elif model=="FFNN":
        model=tf.keras.models.load_model("ffnn_model.keras")
    elif model=="VAR":
        model=tf.keras.models.load_model("var_model.keras")
def get_data(uploaded_file):
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        display_option = st.sidebar.selectbox("Chọn cách hiển thị dữ liệu", ["Số liệu", "Biểu đồ", "Cả hai"])
        st.session_state["data"]=data
        if display_option == "Số liệu":
            st.header("Dữ liệu :", divider='rainbow')
            st.write(data.head(10))
        elif display_option == "Biểu đồ":
            st.header("Biểu đồ", divider='rainbow')
            column_name=get_option_column_to_draw(data)
            draw(data,column_name)
        elif display_option == "Cả hai":
            st.header("Dữ liệu :", divider='rainbow')
            st.write(data)
            st.header("Biểu đồ", divider='rainbow')
            column_name=get_option_column_to_draw(data)
            draw(data,column_name)
        return st.session_state["data"]
    
def get_option_standardize():
    st.sidebar.header("Chuẩn hóa", divider='rainbow')

    option = st.sidebar.selectbox("Chọn cách chuẩn hóa:",
                           ["Không chuẩn hóa",
                            "min-max",
                            "zero-min"]
                           )
    return option

def get_ratio_train_test():
    st.sidebar.header("Chia tỉ lệ train test", divider='rainbow')
    # Tạo thanh trượt để chọn tỉ lệ phần trăm
    percentage_train =  st.sidebar.slider(
    "Chọn tỉ lệ phần trăm :", 
    min_value=0, 
    max_value=100, 
    value=50, 
    step=1)
    
    percentage_test=100-percentage_train

    # Hiển thị giá trị đã chọn
    st.sidebar.write(f"Tỉ lệ phần trăm tập train là: {percentage_train}%")
    st.sidebar.write(f"Tỉ lệ phần trăm tập test là: {percentage_test}%")
    return percentage_test/100
def get_option_min_max():
    st.sidebar.write("Nhập giá trị min, max")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        min=st.text_input("Min")
    with col2:  
        max=st.text_input("Max")
    if min == "" or max == "":
        #st.warning("Vui lòng nhập cả giá trị min và max.")
        return 0, 1  # Giá trị mặc định (hoặc xử lý theo cách bạn muốn)

    try:
        min_val = int(min)
        max_val = int(max)
        return min_val, max_val
    except ValueError:
        st.error("Giá trị nhập vào phải là số nguyên.")
        return 0, 1  # Giá trị mặc định khi lỗi xảy ra
def get_ratio_val():
    st.sidebar.header("Chia tỉ lệ train val", divider='rainbow')
    # Tạo thanh trượt để chọn tỉ lệ phần trăm
    percentage_train =  st.sidebar.slider(
    "Chọn tỉ lệ phần trăm:", 
    min_value=0, 
    max_value=100, 
    value=50, 
    step=1)
    
    percentage_val=100-percentage_train

    # Hiển thị giá trị đã chọn
    st.sidebar.write(f"Tỉ lệ phần trăm tập train là: {percentage_train}%")
    st.sidebar.write(f"Tỉ lệ phần trăm tập val là: {percentage_val}%")
    return percentage_val/100
def get_option_column_to_draw(data):
    st.sidebar.header("Chọn cột",  divider='rainbow')
    column_name=data.columns
    option = st.sidebar.selectbox("Chọn cột", list(column_name[1:]),key="select_column_to_draw")
    return option
def get_option_column_to_draw_df_chuan_hoa(data):
    st.sidebar.write("Chọn cột")
    column_name=data.columns
    option = st.sidebar.selectbox("Chọn cột", list(column_name[0:]),key="select_column_to_draw_df_chuan_hoa")
    return option
def get_option_column_to_check_stationary(data):
    st.sidebar.header("Chọn cột kiểm định chuỗi dừng",  divider='rainbow')
    column_name=data.columns
    option = st.sidebar.selectbox("Chọn cột", list(column_name[0:]),key="get_option_column_to_check_stationary")
    return option
def get_option_column_to_draw_data_augumentation(data):
    st.sidebar.header("Chọn cột kiểm định chuỗi dừng",  divider='rainbow')
    column_name=data.columns
    option = st.sidebar.selectbox("Chọn cột", list(column_name[0:]),key="get_option_column_to_draw_data_augumentation")
    return option
def get_option_column_to_draw_test_data(data):
    st.sidebar.header("Chọn cột để test",  divider='rainbow')
    column_name=data.columns
    option = st.sidebar.selectbox("Chọn cột", list(column_name[0:]),key="get_option_column_to_draw_test_data")
    return option
def get_option_data_argumentation():
    st.sidebar.header("Thêm data",  divider='rainbow')
    option = st.sidebar.selectbox("Chọn thêm", ["Không",
                            "Có"])
    return option
def data_argumentation(data):
    op=get_option_data_argumentation()
    if op=="Không":
        return data
    else:
        df=v.time_warping(data)
        df=pd.DataFrame(df)
        df.columns=data.columns
        st.session_state["df"]=df
        op=get_option_display_data_augumentation()
        if op=="Số liệu":
            st.header("Dữ liệu đã thêm:", divider='rainbow')
            st.dataframe(st.session_state["df"])
            return st.session_state["df"]
        elif op=="Biểu đồ":
            column_name=get_option_column_to_draw_data_augumentation(df)
            st.header("Biểu đồ sau khi thêm dữ liệu", divider='rainbow')           
            draw_df_chuan_hoa(df,column_name)
            return st.session_state["df"]
        else:
            st.header("Dữ liệu đã thêm:", divider='rainbow')
            st.dataframe(st.session_state["df"])
            column_name=get_option_column_to_draw_data_augumentation(df)
            st.header("Biểu đồ sau khi thêm dữ liệu", divider='rainbow')           
            draw_df_chuan_hoa(df,column_name)
            return st.session_state["df"]
        
def get_option_display_data_augumentation():
    display_option = st.sidebar.selectbox("Chọn cách hiển thị dữ liệu", ["Số liệu", "Biểu đồ", "Cả hai"],key="display_option_data_augumentation")    
    return display_option
def get_option_column_to_draw_data_augumentation(data):
    st.sidebar.write("Chọn cột")
    column_name=data.columns
    option = st.sidebar.selectbox("Chọn cột", list(column_name[1:]),key="select_column_to_draw_data_augumentation")
    return option
def check_stationary(data, column):
    if st.sidebar.button("Kiểm định chuỗi dừng", key="check_stationary_button"):
        st.header("Kiểm định chuỗi dừng", divider='rainbow')
        str, k = v.kiemdinh(data, column)
        st.write(str)
        return k
            

def chuanhoa(data,op):
    if op=="Không chuẩn hóa":
        st.session_state["data"]=data
        return data,data
    elif op=="min-max":
        min,max=get_option_min_max()
        st.header("Dữ liệu đã chuẩn hóa:", divider='rainbow')       
        df,scaler=v.Min_max_scaler(data,min,max)
        display_option=get_option_display_df_chuan_hoa()
        st.session_state["df"]=df
        st.session_state["scaler"]=scaler

       
        if display_option=="Số liệu":
            #st.header("Dữ liệu :", divider='rainbow')
            st.dataframe(st.session_state["df"])
            return st.session_state["df"],st.session_state["scaler"]
        elif display_option == "Biểu đồ":
            column_name=get_option_column_to_draw_df_chuan_hoa(df)
            st.header("Biểu đồ", divider='rainbow')           
            draw_df_chuan_hoa(df,column_name)
            return st.session_state["df"],st.session_state["scaler"]
        elif display_option == "Cả hai":
            column_name=get_option_column_to_draw_df_chuan_hoa(df)
            st.header("Dữ liệu :", divider='rainbow')
            st.dataframe(st.session_state["df"])            
            st.header("Biểu đồ", divider='rainbow')
            draw_df_chuan_hoa(st.session_state["df"],column_name)
            return st.session_state["df"],st.session_state["scaler"]
        
    elif op=="zero-min":
        st.header("Dữ liệu đã chuẩn hóa:", divider='rainbow')
        df,scaler=v.Zero_min_scaler(data)
        st.session_state["df"]=df
        st.session_state["scaler"]=scaler
        st.dataframe(st.session_state["df"])
        return st.session_state["df"],st.session_state["scaler"]
        
        
def get_option_display_df_chuan_hoa():
    display_option = st.sidebar.selectbox("Chọn cách hiển thị dữ liệu", ["Số liệu", "Biểu đồ", "Cả hai"],key="display_option_chuan_hoa")    
    return display_option

@st.cache_data
def draw(data,column):
    fig, ax = plt.subplots(figsize=(15, 5))
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(data.index, data[column], label=column, color='blue')
    ax.set_title(f"{column} Over Time (Sampled)", fontsize=14)
    ax.set_xlabel("Date Time")
    ax.set_ylabel(column)
    ax.grid()
    ax.legend()
    st.pyplot(fig)
@st.cache_data
def draw_test_data(data1,data2,col):
    fig = go.Figure()
    #Thêm dữ liệu giá trị thực
    fig.add_trace(go.Scatter(
    y=data1[col].values,
    mode='lines',
    name='Giá trị thực (y_test)',
    line=dict(color='blue')
    ))
                
    # Thêm dữ liệu giá trị dự đoán
    fig.add_trace(go.Scatter(
    y=data2[col].values,
    mode='lines',
    name='Giá trị dự đoán (y_test_pre)',
    line=dict(color='red')
    ))

    # # # Cập nhật layout
    fig.update_layout(
    title=f"So sánh Giá trị Thực và Dự đoán cột {col}",
    xaxis_title="Thời gian (hoặc mẫu)",
    yaxis_title="Giá trị",
    legend=dict(x=0.01, y=0.99),
    height=500,
    width=800
    )
    st.plotly_chart(fig)
    
@st.cache_data
def draw_df_chuan_hoa(data,column):
    data = data.iloc[::6]
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(data.index, data[column], label=column, color='blue')
    ax.set_title(f"{column} Over Time", fontsize=14)
    #ax.set_xlabel("Date Time")
    ax.set_ylabel(column)
    ax.grid()
    ax.legend()
    st.pyplot(fig)
    
def data_preprocessing(data):
    st.session_state["df_tr"]=v.tranformation(data)
    if "find_hyperparam_clicked" not in st.session_state:
        st.session_state["find_hyperparam_clicked"] = False

    if st.sidebar.button("Tiền xử lí",key="data_preprocessing"):
        st.session_state["find_hyperparam_clicked"] = True
    if st.session_state["find_hyperparam_clicked"]:
        st.header("Dữ liệu đã tiền xử lí:", divider='rainbow')
        st.dataframe(st.session_state["df_tr"])
        return st.session_state["df_tr"]

def find_hyperparameter(train_data,test_data,ratio_train_val,lag):
    if st.sidebar.button("Tìm tham số tối ưu"):
        st.session_state["lstm_unit"],st.session_state["epochs"],st.session_state["batch_size"]=v.find_parameter_for_ffnn(train_data,test_data,ratio_train_val,lag)
        st.header("Các tham số tối ưu:", divider='rainbow')
        lstm_unit=st.session_state["lstm_unit"]
        epochs=st.session_state["epochs"]
        batch_size=st.session_state["batch_size"]
        st.write(f"Số đơn vị ẩn:{lstm_unit}")
        st.write(f"Epochs: {epochs}")
        st.write(f"Batch_size:{batch_size}")
        st.write(f"Độ trễ: {lag}")
        return [st.session_state["lstm_unit"],st.session_state["epochs"], st.session_state["batch_size"]]

def train_model(df_chuan_hoa,scaler,op_data):  
    model=get_model()
    ratio_train_test=get_ratio_train_test()
    train_data,test_data= v.devide_train_test(df_chuan_hoa,ratio_train_test)
    lag=v.find_lag(train_data) 
    
    if model=="VAR":
        if "button_clicked" not in st.session_state:
            st.session_state.button_clicked = False
        if st.sidebar.button("Train"):
            st.session_state.button_clicked = True
        if st.session_state.button_clicked:
            start_train_time = time.time()
            var_model,forecast,st.session_state.y_test,st.session_state.y_test_pre,st.session_state.mse_var,st.session_state.mae_var,st.session_state.cv_rmse_var=v.train_VAR(train_data,test_data,lag)
       
            st.session_state.forecast=forecast                     
            end_train_time = time.time()
            train_time=end_train_time-start_train_time

            st.header("Kết quả huấn luyện:", divider='rainbow')
            st.write(f"Thời gian huấn luyện: {train_time:.2f} giây")

            if op == "Không chuẩn hóa":
                y_test=pd.DataFrame(st.session_state.y_test)
                y_test_pre=pd.DataFrame(st.session_state.y_test_pre)
                y_test.columns=df_chuan_hoa.columns
                y_test_pre.columns=df_chuan_hoa.columns
                    
            else:
                    
                y_test=scaler.inverse_transform(st.session_state.y_test)
                y_test_pre=scaler.inverse_transform(st.session_state.y_test_pre)
                y_test=pd.DataFrame(y_test)
                y_test_pre=pd.DataFrame(y_test_pre)
                y_test.columns=df_chuan_hoa.columns
                y_test_pre.columns=df_chuan_hoa.columns
            #st.header("Kết quả dự đoán:", divider='rainbow')
            kiem_tra_mo_hinh(model,st.session_state.mse_var,st.session_state.mae_var,st.session_state.cv_rmse_var,y_test,y_test_pre)
            if st.sidebar.button("Dự đoán"):
                if op == "Không chuẩn hóa":
                    data=pd.DataFrame(st.session_state.forecast)
                    data.columns=df_chuan_hoa.columns            
                    st.dataframe(data.reset_index(drop=True))
                else:
                    st.session_state.forecast=scaler.inverse_transform(st.session_state.forecast)
                    #st.session_state["df"]=v.get_data_target(df_chuan_hoa).columns
                    data=pd.DataFrame(st.session_state.forecast)
                    data.columns=df_chuan_hoa.columns
            
                    st.dataframe(data.reset_index(drop=True))
            
    elif model=="VARNN" or model=="FFNN":
        ratio_train_val=get_ratio_val()
        #lstm_unit, epochs, batch_size = None, None, None  # Giá trị mặc định
        
        if st.sidebar.button("Tìm tham số tối ưu"):
            #st.session_state.lstm_unit, st.session_state.epochs, st.session_state.batch_size=v.find_parameter_for_ffnn(train_data,test_data,ratio_train_val,lag)
            lstm_unit = None
            epochs = None
            batch_size = None
            if ratio_train_test==0.2 and ratio_train_val==0.2:
                with open("optimized\optimized_params_80_20.json", "r") as file:
                    loaded_params = json.load(file)
            elif ratio_train_test==0.2 and ratio_train_val==0.25:
                with open("optimized\optimized_params_80_20_75_25.json", "r") as file:
                    loaded_params = json.load(file)
            elif ratio_train_test==0.2 and ratio_train_val==0.3:
                with open("optimized\optimized_params_80_20_70_30.json", "r") as file:
                    loaded_params = json.load(file)
            elif ratio_train_test==0.25 and ratio_train_val==0.2:
                with open("optimized\optimized_params_75_25_80_20.json", "r") as file:
                    loaded_params = json.load(file)
            elif ratio_train_test==0.25 and ratio_train_val==0.25:
                with open("optimized\optimized_params_75_25_75_25.json", "r") as file:
                    loaded_params = json.load(file)
            elif ratio_train_test==0.25 and ratio_train_val==0.3:
                with open("optimized\optimized_params_75_25_70_30.json", "r") as file:
                    loaded_params = json.load(file)
            elif ratio_train_test==0.3 and ratio_train_val==0.2:
                with open("optimized\optimized_params_70_30_80_20.json", "r") as file:
                    loaded_params = json.load(file)
            elif ratio_train_test==0.3 and ratio_train_val==0.25:
                with open("optimized\optimized_params_70_30_75_25.json", "r") as file:
                    loaded_params = json.load(file)
            elif ratio_train_test==0.3 and ratio_train_val==0.3:
                with open("optimized\optimized_params_70_30_70_30.json", "r") as file:
                    loaded_params = json.load(file)
            else:
                st.session_state.lstm_unit, st.session_state.epochs, st.session_state.batch_size=v.find_parameter_for_ffnn(train_data,test_data,ratio_train_val,lag)
                lstm_unit = st.session_state.lstm_unit
                epochs = st.session_state.epochs
                batch_size = st.session_state.batch_size
            
            if lstm_unit is None or epochs is None or batch_size is None:
                st.session_state.lstm_unit=loaded_params[op_data].get("lstm_units")
                st.session_state.epochs=loaded_params[op_data].get("epochs")
                st.session_state.batch_size=loaded_params[op_data].get("batch_size")
                lstm_unit = st.session_state.lstm_unit
                epochs = st.session_state.epochs
                batch_size = st.session_state.batch_size
            st.header("Các tham số tối ưu:", divider='rainbow')
            st.write(f"Hidden units:{lstm_unit}")
            st.write(f"Epochs: {epochs}")
            st.write(f"Batch size:{batch_size}")
            st.write(f"Lag: {lag}") 
            
        if model=="VARNN": 
            if "button_clicked" not in st.session_state:
                st.session_state.button_clicked = False
            if st.sidebar.button("Train"):
                st.session_state.button_clicked = True
            if st.session_state.button_clicked:
                lstm_unit = st.session_state.lstm_unit
                epochs = st.session_state.epochs
                batch_size = st.session_state.batch_size

                #Tiến hành train
                start_train_time = time.time()
                history,latest_prediction,st.session_state.y_test,st.session_state.y_test_pre,st.session_state.mse_varnn, st.session_state.mae_varnn, st.session_state.cv_rmse_varnn=v.train_varnn(train_data,test_data,lag,epochs,lstm_unit,batch_size)
                st.session_state.latest_prediction=latest_prediction           
                end_train_time = time.time()

                train_time=end_train_time-start_train_time

                #Vẽ biểu đồ 2 đường loss và val_loss
                st.header("Kết quả huấn luyện:", divider='rainbow')
                # Vẽ biểu đồ loss/accuracy
                fig_metrics = go.Figure()
                fig_metrics.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss'))
                fig_metrics.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
                fig_metrics.update_layout(title='Loss theo epoch',
                    xaxis_title='Epoch',
                    yaxis_title='Loss')
                st.plotly_chart(fig_metrics)
                #In ra các thông số train
                st.write(f"Thời gian huấn luyện: {train_time:.2f} giây")
                st.write(f"Loss cuối cùng: {history.history['loss'][-1]:.4f}")
                st.write(f"Số epochs hoàn thành: {len(history.history['loss'])}")
                st.write(f"Validation Loss cuối cùng: {history.history['val_loss'][-1]:.4f}")  
                
                #st.dataframe(y_test)
                #st.header("Kiểm tra mô hình varnn:", divider='rainbow')
                
                if op == "Không chuẩn hóa":
                    y_test=pd.DataFrame(st.session_state.y_test)
                    y_test_pre=pd.DataFrame(st.session_state.y_test_pre)
                    y_test.columns=df_chuan_hoa.columns
                    y_test_pre.columns=df_chuan_hoa.columns
                    
                else:
                    
                    y_test=scaler.inverse_transform(st.session_state.y_test)
                    y_test_pre=scaler.inverse_transform(st.session_state.y_test_pre)
                    y_test=pd.DataFrame(y_test)
                    y_test_pre=pd.DataFrame(y_test_pre)
                    y_test.columns=df_chuan_hoa.columns
                    y_test_pre.columns=df_chuan_hoa.columns
                kiem_tra_mo_hinh(model,st.session_state.mse_varnn,st.session_state.mae_varnn,st.session_state.cv_rmse_varnn,y_test,y_test_pre)
                if st.sidebar.button("Dự đoán"):
                    if op == "Không chuẩn hóa":
                        st.header("Kết quả dự đoán",divider="rainbow")
                        data=pd.DataFrame(st.session_state.latest_prediction)
                        data.columns=df_chuan_hoa.columns
                
                        st.dataframe(data.reset_index(drop=True))
                    else:
                        st.header("Kết quả dự đoán",divider="rainbow")
                        st.session_state.latest_prediction=scaler.inverse_transform(st.session_state.latest_prediction)
                        data=pd.DataFrame(st.session_state.latest_prediction)
                        data.columns=df_chuan_hoa.columns
                
                        st.dataframe(data.reset_index(drop=True))
            else:
                pass
                
                
        elif model=="FFNN":
            if "button_clicked" not in st.session_state:
                st.session_state.button_clicked = False
            if st.sidebar.button("Train"):
                st.session_state.button_clicked = True
            if st.session_state.button_clicked:
                lstm_unit = st.session_state.lstm_unit
                epochs = st.session_state.epochs
                batch_size = st.session_state.batch_size

                start_train_time = time.time()

                history,latest_prediction,st.session_state.y_test,st.session_state.y_test_pre,st.session_state.mse_ffnn, st.session_state.mae_ffnn, st.session_state.cv_rmse_ffnn=v.train_ffnn(train_data,test_data,lag,epochs,lstm_unit,batch_size)
                st.session_state.latest_prediction=latest_prediction
                #y_test=pd.DataFrame(st.session_state.y_test)
                #y_test_pre=pd.DataFrame(st.session_state.y_test_pre)
                #y_test.columns=df_chuan_hoa.columns
                #y_test_pre.columns=df_chuan_hoa.columns
                
                end_train_time = time.time()

                train_time=end_train_time-start_train_time
                
                #Vẽ biểu đồ 2 đường loss và val_loss
                st.header("Kết quả huấn luyện:", divider='rainbow')
                # Vẽ biểu đồ loss/accuracy
                fig_metrics = go.Figure()
                fig_metrics.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss'))
                fig_metrics.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
                fig_metrics.update_layout(title='Loss theo epoch',
                    xaxis_title='Epoch',
                    yaxis_title='Loss')
                st.plotly_chart(fig_metrics)
                #In ra các thông số train
                st.write(f"Thời gian huấn luyện: {train_time:.2f} giây")
                st.write(f"Loss cuối cùng: {history.history['loss'][-1]:.4f}")
                st.write(f"Số epochs hoàn thành: {len(history.history['loss'])}")
                st.write(f"Validation Loss cuối cùng: {history.history['val_loss'][-1]:.4f}")  
                
                #st.dataframe(y_test)
                #st.header("Kiểm tra mô hình ffnn:", divider='rainbow')
                if op == "Không chuẩn hóa":
                    y_test=pd.DataFrame(st.session_state.y_test)
                    y_test_pre=pd.DataFrame(st.session_state.y_test_pre)
                    y_test.columns=df_chuan_hoa.columns
                    y_test_pre.columns=df_chuan_hoa.columns
                    kiem_tra_mo_hinh(model,st.session_state.mse_ffnn,st.session_state.mae_ffnn,st.session_state.cv_rmse_ffnn,y_test,y_test_pre)
                else:
                    
                    y_test=scaler.inverse_transform(st.session_state.y_test)
                    y_test_pre=scaler.inverse_transform(st.session_state.y_test_pre)
                    y_test=pd.DataFrame(y_test)
                    y_test_pre=pd.DataFrame(y_test_pre)
                    y_test.columns=df_chuan_hoa.columns
                    y_test_pre.columns=df_chuan_hoa.columns
                    kiem_tra_mo_hinh(model,st.session_state.mse_ffnn,st.session_state.mae_ffnn,st.session_state.cv_rmse_ffnn,y_test,y_test_pre)
                if st.sidebar.button("Dự đoán"):
                    if op == "Không chuẩn hóa":
                        st.header("Kết quả dự đoán",divider="rainbow")
                        data=pd.DataFrame(st.session_state.latest_prediction)
                        data.columns=df_chuan_hoa.columns
                
                        st.dataframe(data.reset_index(drop=True))
                    else:
                        st.header("Kết quả dự đoán",divider="rainbow")
                        st.session_state.latest_prediction=scaler.inverse_transform(st.session_state.latest_prediction)
                        data=pd.DataFrame(st.session_state.latest_prediction)
                        data.columns=df_chuan_hoa.columns
                
                        st.dataframe(data.reset_index(drop=True))
def get_option_kiem_tra():
    display_option = st.sidebar.selectbox("Chọn cách kiem tra", ["khong","kiem tra"],key="display_option_data_augumentation")    
    return display_option
def kiem_tra_mo_hinh(model,mse,mae,cv_rmse,y_test,y_test_pre):
    #op=get_option_kiem_tra()
    #if st.sidebar.button("Kiểm tra mô hình",key="kiem_tra"):    
    #if op=="kiem tra":
    if "button_clicked" not in st.session_state:
            st.session_state.button_clicked = False
    if "results" not in st.session_state:
        st.session_state.results = None
    if st.sidebar.button("Kiểm tra mô hình"):
        st.header(f"Kiểm tra mô hình {model}", divider='rainbow')
        st.session_state.button_clicked = True
        st.session_state.results = {
            "MSE": mse,
            "MAE": mae,
            "CV_RMSE": cv_rmse,
            "y_test_pre": y_test_pre,
            "y_test": y_test,
        }
    if st.session_state.button_clicked and st.session_state.results:
    #if st.sidebar.checkbox("Kiểm tra mô hình"):
        st.header("Đánh giá mô hình:")
        st.write(f"MSE: {st.session_state.results['MSE']}")
        st.write(f"MAE: {st.session_state.results['MAE']}")
        st.write(f"CV_RMSE: {st.session_state.results['CV_RMSE']}")
        st.dataframe(st.session_state.results["y_test_pre"])

        # Gọi hàm vẽ đồ thị (giả sử các hàm này đã được định nghĩa)
        op_d = get_option_column_to_draw_test_data(st.session_state.results["y_test"])
        draw_test_data(
            st.session_state.results["y_test"],
            st.session_state.results["y_test_pre"],
            op_d,
        )
    else:
        pass
#Load data
uploaded_file = st.sidebar.file_uploader("Chọn tập dữ liệu (CSV)", type=["csv"])
data = get_data(uploaded_file)
op_data=None
if uploaded_file is None:
    pass
else:
    if uploaded_file.name=="jena_climate_2009_2016.csv":
        op_data=0
    elif uploaded_file.name=="wind_dataset.csv":
        op_data=1
    elif uploaded_file.name=="Tetuan City power consumption.csv":
        op_data=2
    else:
        op_data=3
#Tiền xử lí
data_preprocess=data_preprocessing(data) 
k=None
#column=None
if data_preprocess is None:
    pass
else:
    column=get_option_column_to_check_stationary(data_preprocess)
    
    k=check_stationary(data_preprocess,column)
    
    if k==True: 
        pass
    else:
        op=st.sidebar.selectbox("Chuẩn hóa chuỗi dừng",[ "Không chuẩn hóa","Chuẩn hóa"])
        if op=="Chuẩn hóa":
            data_preprocess=v.chuanhoachuoidung(data_preprocess)
            st.write("Dữ liệu đã chuẩn hóa chuỗi dừng")
            st.dataframe(data_preprocess)
        else:
            pass
data_augumentation=data_argumentation(data_preprocess)

#Chuan hoa
op=get_option_standardize()
df_chuan_hoa, scaler=chuanhoa(data_augumentation,op)

if df_chuan_hoa is None and scaler is None:
    pass
else:
    train_model(df_chuan_hoa,scaler,op_data)