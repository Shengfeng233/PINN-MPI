import numpy as np

from read_data import *
from pinn_model import *
from sub_functions import identify_sub_location
import matplotlib.pyplot as plt
import matplotlib as mpl


# 按逐个子网络进行预测
# predict where there is true data
def predict_steady(filename_raw_data, file_predict, size_of_process, pinn_example, lb, ub):
    filename_model = file_predict+'/{:02d}NS_model_train.pt'
    side_length = int(size_of_process ** 0.5)  # 每边（维度）有几个子网络
    x_raw, y_raw, u_raw, v_raw, feature_mat = read_2D_data_without_p(filename_raw_data)
    x_pre = np.zeros_like(x_raw)
    y_pre = np.zeros_like(y_raw)
    u_pre = np.zeros_like(u_raw)
    v_pre = np.zeros_like(v_raw)
    for i in range(size_of_process):
        filename_load_model = filename_model.format(i)
        pinn_example.load_state_dict(torch.load(filename_load_model, map_location=device))
        sub_net_info = identify_sub_location(i, size_of_process, lb, ub)
        x_sub_index = np.where((x_raw >= sub_net_info.sub_lb[0, 0]) & (x_raw <= sub_net_info.sub_ub[0, 0]))[0].reshape(-1, 1)
        y_sub_index = np.where((y_raw >= sub_net_info.sub_lb[0, 1]) & (y_raw <= sub_net_info.sub_ub[0, 1]))[0].reshape(-1, 1)
        index_sub = np.intersect1d(x_sub_index, y_sub_index, assume_unique=False, return_indices=False).reshape(-1, 1)
        x_sub = x_raw[index_sub, 0].clone().requires_grad_(True).to(device)
        y_sub = y_raw[index_sub, 0].clone().requires_grad_(True).to(device)
        u_sub, v_sub, p_sub = pinn_example.predict(x_sub, y_sub)
        x_pre[index_sub, 0] = x_sub.detach().numpy()
        y_pre[index_sub, 0] = y_sub.detach().numpy()
        u_pre[index_sub, 0] = u_sub.detach().numpy()
        v_pre[index_sub, 0] = v_sub.detach().numpy()
    return x_pre, y_pre, u_pre, v_pre


# 对比真实场和预测场
def compare_steady(filename_raw_data, x_pre, y_pre, u_pre, v_pre):
    x_raw, y_raw, u_raw, v_raw, feature_mat = read_2D_data_without_p(filename_raw_data)
    x_raw = x_raw.numpy()
    min_u = feature_mat.numpy()[1, 2]
    max_u = feature_mat.numpy()[0, 2]
    min_v = feature_mat.numpy()[1, 3]
    max_v = feature_mat.numpy()[0, 3]
    u_raw_mat = u_raw.numpy()
    v_raw_mat = v_raw.numpy()
    grid_num = x_raw.shape[0]
    grid_num_per_axis = int(np.sqrt(grid_num).item())
    x_values = np.unique(x_pre).reshape(-1, 1)
    y_values = np.unique(y_pre).reshape(-1, 1)
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    u_select = u_raw_mat.reshape(x_mesh.shape)
    v_select = v_raw_mat.reshape(y_mesh.shape)
    u_predict = u_pre.reshape(x_mesh.shape)
    v_predict = v_pre.reshape(x_mesh.shape)
    plot_compare(u_select, u_predict, name='u', min_value=min_u, max_value=max_u)
    plot_compare(v_select, v_predict, name='v', min_value=min_v, max_value=max_v)
    return


# plot
def plot_compare(q_selected, q_predict, min_value, max_value, name='q'):
    v_norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(q_selected, cmap='jet', norm=v_norm)
    plt.title("True_" + name + " _value:")
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(q_predict, cmap='jet', norm=v_norm)
    plt.title("Predict" + name + " _value:")
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.colorbar()
    plt.savefig('./gif_make/' + name + '.png')
    plt.show()


# 对于定常流场进行预测
if __name__ == "__main__":
    low_bound = np.array([0, 0]).reshape(1, -1)  # 二维定常，仅含x,y
    up_bound = np.array([1, 1]).reshape(1, -1)    # 二维定常，仅含x,y
    layer_mat_psi = [2, 20, 20, 2]  # 网络结构
    layer_mat = layer_mat_psi
    size_of_process = 576
    pinn_net = PINN_Net(layer_mat)
    pinn_net = pinn_net.to(device)
    filename_raw_data = './stack_Re100.mat'
    file_predict = './data/exp_d_1/sub_write'
    x_predict, y_predict, u_predict, v_predict = predict_steady(filename_raw_data, file_predict, size_of_process, pinn_net, low_bound, up_bound)
    compare_steady(filename_raw_data, x_predict, y_predict, u_predict, v_predict)
    print('ok')
