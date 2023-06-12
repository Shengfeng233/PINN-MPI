"""
predict and plot
weighted sum on overlapping domain
时空并行画图
对重叠区域进行加权平均
"""
import os.path
import numpy as np
from read_data import *
from pinn_model import *
from sub_functions import identify_sub_location
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio

mpl.use('Agg')


# Loading non-repetitive predicted points while traversing the subnetwork
# 一边遍历子网络，一边加载非重复预测点
def predict_unsteady(filename_raw_data, file_predict, size_of_process, time_division, extension_ratio, layer_mat):
    filename_model = file_predict + '/{:02d}NS_model_train.pt'
    x_raw, y_raw, t_raw, u_raw, v_raw, p_raw, feature_mat = read_3D_data(filename_raw_data)
    low_bound = feature_mat.numpy()[0, 0:3].reshape(1, -1)
    up_bound = feature_mat.numpy()[1, 0:3].reshape(1, -1)
    x_pre = np.zeros_like(x_raw.numpy())
    y_pre = np.zeros_like(y_raw.numpy())
    t_pre = np.zeros_like(t_raw.numpy())
    u_pre = np.zeros_like(u_raw.numpy())
    v_pre = np.zeros_like(v_raw.numpy())
    p_pre = np.zeros_like(p_raw.numpy())
    count_pre = np.zeros_like(p_raw.numpy()).astype(int)
    for i in range(size_of_process):
        filename_load_model = filename_model.format(i)
        sub_net_info = identify_sub_location(i, size_of_process, time_division, extension_ratio, low_bound, up_bound)
        pinn_net = PINN_Net(layer_mat, sub_net_info.sub_lb, sub_net_info.sub_ub)
        pinn_net = pinn_net.to(device)
        pinn_net.load_state_dict(torch.load(filename_load_model, map_location=device))
        x_sub_index = np.where((x_raw >= sub_net_info.extended_lb[0, 0]) & (x_raw <= sub_net_info.extended_ub[0, 0]))[0].reshape(-1, 1)
        y_sub_index = np.where((y_raw >= sub_net_info.extended_lb[0, 1]) & (y_raw <= sub_net_info.extended_ub[0, 1]))[0].reshape(-1, 1)
        t_sub_index = np.where((t_raw >= sub_net_info.extended_lb[0, 2]) & (t_raw <= sub_net_info.extended_ub[0, 2]))[0].reshape(-1, 1)
        index_sub_space = np.intersect1d(x_sub_index, y_sub_index, assume_unique=False, return_indices=False).reshape(-1, 1)
        index_sub = np.intersect1d(index_sub_space, t_sub_index, assume_unique=False, return_indices=False).reshape(-1, 1)
        x_sub = x_raw[index_sub, 0].clone().detach().requires_grad_(True).to(device)
        y_sub = y_raw[index_sub, 0].clone().detach().requires_grad_(True).to(device)
        t_sub = t_raw[index_sub, 0].clone().detach().requires_grad_(True).to(device)
        u_sub, v_sub, p_sub = pinn_net.predict(x_sub, y_sub, t_sub)
        x_pre[index_sub, 0] += x_sub.clone().detach().numpy()
        y_pre[index_sub, 0] += y_sub.clone().detach().numpy()
        t_pre[index_sub, 0] += t_sub.clone().detach().numpy()
        u_pre[index_sub, 0] += u_sub.clone().detach().numpy()
        v_pre[index_sub, 0] += v_sub.clone().detach().numpy()
        p_pre[index_sub, 0] += p_sub.clone().detach().numpy()
        count_pre[index_sub, 0] += 1
    x_pre = x_pre / count_pre
    y_pre = y_pre / count_pre
    t_pre = t_pre / count_pre
    u_pre = u_pre / count_pre
    v_pre = v_pre / count_pre
    p_pre = p_pre / count_pre
    return x_pre, y_pre, t_pre, u_pre, v_pre, p_pre


# 按逐个子网络进行提取数据并预测
# 获得预测集,预测集中有重复, 此方法耗时较多
# predict where there is true data
def predict_unsteady_overlap(filename_raw_data, file_predict, size_of_process, time_division, extension_ratio,
                             layer_mat, lb, ub):
    filename_model = file_predict + '/{:02d}NS_model_train.pt'
    x_raw, y_raw, t_raw, u_raw, v_raw, p_raw, feature_mat = read_3D_data(filename_raw_data)
    x_pre_overlap = np.empty((0, 1), dtype=np.float32)
    y_pre_overlap = np.empty((0, 1), dtype=np.float32)
    t_pre_overlap = np.empty((0, 1), dtype=np.float32)
    u_pre_overlap = np.empty((0, 1), dtype=np.float32)
    v_pre_overlap = np.empty((0, 1), dtype=np.float32)
    p_pre_overlap = np.empty((0, 1), dtype=np.float32)
    for i in range(size_of_process):
        filename_load_model = filename_model.format(i)
        sub_net_info = identify_sub_location(i, size_of_process, time_division, extension_ratio, lb, ub)
        pinn_net = PINN_Net(layer_mat, sub_net_info.sub_lb, sub_net_info.sub_ub)
        pinn_net = pinn_net.to(device)
        pinn_net.load_state_dict(torch.load(filename_load_model, map_location=device))
        x_sub_index = np.where((x_raw >= sub_net_info.sub_lb[0, 0]) & (x_raw <= sub_net_info.sub_ub[0, 0]))[0].reshape(
            -1, 1)
        y_sub_index = np.where((y_raw >= sub_net_info.sub_lb[0, 1]) & (y_raw <= sub_net_info.sub_ub[0, 1]))[0].reshape(
            -1, 1)
        t_sub_index = np.where((t_raw >= sub_net_info.sub_lb[0, 2]) & (t_raw <= sub_net_info.sub_ub[0, 2]))[0].reshape(
            -1, 1)
        index_sub_space = np.intersect1d(x_sub_index, y_sub_index, assume_unique=False, return_indices=False).reshape(
            -1, 1)
        index_sub = np.intersect1d(index_sub_space, t_sub_index, assume_unique=False, return_indices=False).reshape(-1,
                                                                                                                    1)
        x_sub = x_raw[index_sub, 0].clone().detach().requires_grad_(True).to(device)
        y_sub = y_raw[index_sub, 0].clone().detach().requires_grad_(True).to(device)
        t_sub = t_raw[index_sub, 0].clone().detach().requires_grad_(True).to(device)
        u_sub, v_sub, p_sub = pinn_net.predict(x_sub, y_sub, t_sub)
        x_add = x_sub.clone().detach().numpy()
        y_add = y_sub.clone().detach().numpy()
        t_add = t_sub.clone().detach().numpy()
        u_add = u_sub.clone().detach().numpy()
        v_add = v_sub.clone().detach().numpy()
        p_add = p_sub.clone().detach().numpy()
        numpy.append(x_pre_overlap, x_add, axis=0)
        numpy.append(y_pre_overlap, y_add, axis=0)
        numpy.append(t_pre_overlap, t_add, axis=0)
        numpy.append(u_pre_overlap, u_add, axis=0)
        numpy.append(v_pre_overlap, v_add, axis=0)
        numpy.append(p_pre_overlap, p_add, axis=0)
    return x_pre_overlap, y_pre_overlap, t_pre_overlap, u_pre_overlap, v_pre_overlap, p_pre_overlap


# 通过平均，预测重叠区域的值
def average_overlap(x_pre_overlap, y_pre_overlap, t_pre_overlap, u_pre_overlap, v_pre_overlap, p_pre_overlap):
    overlap_stack = np.concatenate(
        (x_pre_overlap, y_pre_overlap, t_pre_overlap, u_pre_overlap, v_pre_overlap, p_pre_overlap), axis=1)
    # 提取unique信息
    unique_rows, unique_indices, unique_inverse = np.unique(overlap_stack[:, :3], axis=0, return_index=True,
                                                            return_inverse=True)
    merged_stack = np.zeros((len(unique_rows), overlap_stack.shape[1]))
    # 对重叠区域平均
    for i, index in enumerate(unique_indices):
        merged_stack[i, :] = np.mean(overlap_stack[np.where(unique_inverse == i)], axis=0)
    # 将x y t u v p提取出
    x_pre = merged_stack[:, 0].reshape(-1, 1)
    y_pre = merged_stack[:, 1].reshape(-1, 1)
    t_pre = merged_stack[:, 2].reshape(-1, 1)
    u_pre = merged_stack[:, 3].reshape(-1, 1)
    v_pre = merged_stack[:, 4].reshape(-1, 1)
    p_pre = merged_stack[:, 5].reshape(-1, 1)
    return x_pre, y_pre, t_pre, u_pre, v_pre, p_pre


# Comparison of real field and predicted field----divided by time snapshot
# 对比真实场和预测场_按时间快照划分
def compare_unsteady(filename_raw_data, start_index, end_index, u_pre, v_pre, p_pre):
    x_raw, y_raw, t_raw, u_raw, v_raw, p_raw, feature_mat = read_3D_data(filename_raw_data)
    x_raw_mat = x_raw.numpy()
    y_raw_mat = y_raw.numpy()
    t_raw_mat = t_raw.numpy()
    u_raw_mat = u_raw.numpy()
    v_raw_mat = v_raw.numpy()
    p_raw_mat = p_raw.numpy()
    t_unique = np.unique(t_raw_mat).reshape(-1, 1)
    x_unique = np.unique(x_raw_mat).reshape(-1, 1)
    y_unique = np.unique(y_raw_mat).reshape(-1, 1)
    mesh_x, mesh_y = np.meshgrid(x_unique, y_unique)
    time_series = t_unique[start_index:end_index, 0].reshape(-1, 1)
    min_data = feature_mat.numpy()[0, :].reshape(1, -1)
    max_data = feature_mat.numpy()[1, :].reshape(1, -1)
    v_norm_u = mpl.colors.Normalize(vmin=min_data[0, 3], vmax=max_data[0, 3])
    v_norm_v = mpl.colors.Normalize(vmin=min_data[0, 4], vmax=max_data[0, 4])
    v_norm_p = mpl.colors.Normalize(vmin=min_data[0, 5], vmax=max_data[0, 5])
    for select_time in time_series:
        time = select_time.item()
        index_selected = np.where(t_raw_mat == select_time)[0].reshape(-1, 1)
        u_selected = u_raw_mat[index_selected].reshape(mesh_x.shape)
        v_selected = v_raw_mat[index_selected].reshape(mesh_x.shape)
        p_selected = p_raw_mat[index_selected].reshape(mesh_x.shape)
        u_predicted = u_pre[index_selected].reshape(mesh_x.shape)
        v_predicted = v_pre[index_selected].reshape(mesh_x.shape)
        p_predicted = p_pre[index_selected].reshape(mesh_x.shape)
        plot_compare_time_series(mesh_x, mesh_y, u_selected, u_predicted, time, v_norm_u, name='u')
        plot_compare_time_series(mesh_x, mesh_y, v_selected, v_predicted, time, v_norm_v, name='v')
        plot_compare_time_series(mesh_x, mesh_y, p_selected, p_predicted, time, v_norm_p, name='p')
    return


# plot
def plot_compare_time_series(x_mesh, y_mesh, q_selected, q_predict, select_time, v_norm, name='q'):
    plt.cla()
    # v_norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    mappable = ax1.contourf(x_mesh, y_mesh, q_selected, levels=200, cmap='jet', norm=v_norm)
    fig.colorbar(mappable, cax=cbar_ax)
    ax1.set_title("True_" + name + "at " + " t=" + "{:.2f}".format(select_time))
    ax1.set_ylabel('Y')
    ax1.set_xlabel('X')
    ax2.contourf(x_mesh, y_mesh, q_predict, levels=200, cmap='jet', norm=v_norm)
    ax2.set_title("Predict_" + name + " at " + " t=" + "{:.2f}".format(select_time))
    ax2.set_ylabel('Y')
    ax2.set_xlabel('X')
    if not os.path.exists('gif_make'):
        os.makedirs('gif_make')
    plt.savefig('./gif_make/' + 'time' + "{:.2f}".format(select_time) + name + '.png')
    plt.close('all')


# bias between true value and predicted value
# 真实值和预测值偏差
def subtract_unsteady(filename_raw_data, start_index, end_index, u_pre, v_pre, p_pre, mode):
    x_raw, y_raw, t_raw, u_raw, v_raw, p_raw, feature_mat = read_3D_data(filename_raw_data)
    x_raw_mat = x_raw.numpy()
    y_raw_mat = y_raw.numpy()
    t_raw_mat = t_raw.numpy()
    u_raw_mat = u_raw.numpy()
    v_raw_mat = v_raw.numpy()
    p_raw_mat = p_raw.numpy()
    t_unique = np.unique(t_raw_mat).reshape(-1, 1)
    x_unique = np.unique(x_raw_mat).reshape(-1, 1)
    y_unique = np.unique(y_raw_mat).reshape(-1, 1)
    mesh_x, mesh_y = np.meshgrid(x_unique, y_unique)
    time_series = t_unique[start_index:end_index, 0].reshape(-1, 1)
    min_data = feature_mat.numpy()[0, :].reshape(1, -1)
    max_data = feature_mat.numpy()[1, :].reshape(1, -1)
    for select_time in time_series:
        time = select_time.item()
        index_selected = np.where(t_raw_mat == select_time)[0].reshape(-1, 1)
        u_selected = u_raw_mat[index_selected].reshape(mesh_x.shape)
        v_selected = v_raw_mat[index_selected].reshape(mesh_x.shape)
        p_selected = p_raw_mat[index_selected].reshape(mesh_x.shape)
        u_predicted = u_pre[index_selected].reshape(mesh_x.shape)
        v_predicted = v_pre[index_selected].reshape(mesh_x.shape)
        p_predicted = p_pre[index_selected].reshape(mesh_x.shape)
        plot_subtract_time_series(mesh_x, mesh_y, u_selected, u_predicted, time, name='u', mode=mode,
                                  min_value=min_data[0, 3], max_value=max_data[0, 3])
        plot_subtract_time_series(mesh_x, mesh_y, v_selected, v_predicted, time, name='v', mode=mode,
                                  min_value=min_data[0, 4], max_value=max_data[0, 4])
        plot_subtract_time_series(mesh_x, mesh_y, p_selected, p_predicted, time, name='p', mode=mode,
                                  min_value=min_data[0, 5], max_value=max_data[0, 5]*2)
    return


def plot_subtract_time_series(x_mesh, y_mesh, q_selected, q_predict, select_time, min_value, max_value, mode, name='q'):
    plt.cla()
    q_bias = np.abs(q_predict - q_selected)
    v_norm = mpl.colors.Normalize(vmin=0.0, vmax=0.2*max_value)
    plt.figure(figsize=(8, 6))
    # plt.imshow(q_bias, cmap='jet', norm=v_norm)
    plt.contourf(x_mesh, y_mesh, q_bias, levels=np.linspace(0.0,0.2*max_value,100), cmap='jet', norm=v_norm)
    plt.title(mode + "_" + name + " at " + " t=" + "{:.2f}".format(select_time))
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.colorbar()
    if not os.path.exists('gif_make'):
        os.makedirs('gif_make')
    plt.savefig('./gif_make/' + 'time' + "{:.2f}".format(select_time) + name + '.png')
    plt.close('all')




def make_flow_gif(start_index, end_index, t_predict, mode, name='q', fps_num=5):
    gif_images = []
    t_unique = np.unique(t_predict).reshape(-1, 1)
    time_series = t_unique[start_index:end_index, 0].reshape(-1, 1)
    for select_time in time_series:
        time = select_time.item()
        gif_images.append(imageio.imread('./gif_make/' + 'time' + "{:.2f}".format(time) + name + '.png'))
    imageio.mimsave((mode + name + '.gif'), gif_images, fps=fps_num)


if __name__ == "__main__":
    layer_mat_1 = [3, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 3]  # 网络结构 neural network
    layer_mat_2 = [3, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 3]  # 网络结构 neural network
    layer_mat_3 = [3, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 3]
    layer_mat = layer_mat_2
    start_index = 0
    end_index = 100
    size_of_process = 40
    time_division = 10
    extension_ratio = 0.05
    mode = 'compare'
    filename_raw_data = './cylinder_Re3900_ke_all_100snaps.mat'
    file_predict = './data/exp_1_ke/sub_data/sub_write'
    x_predict, y_predict, t_predict, u_predict, v_predict, p_predict = predict_unsteady(filename_raw_data, file_predict,
                                                                                        size_of_process, time_division,
                                                                                        extension_ratio, layer_mat)

    if mode == 'compare':
        compare_unsteady(filename_raw_data, start_index, end_index, u_predict, v_predict, p_predict)
        make_flow_gif(start_index, end_index, t_predict, mode, name='u', fps_num=10)
        make_flow_gif(start_index, end_index, t_predict, mode, name='v', fps_num=10)
        make_flow_gif(start_index, end_index, t_predict, mode, name='p', fps_num=10)
    elif mode == 'subtract':
        subtract_unsteady(filename_raw_data, start_index, end_index, u_predict, v_predict, p_predict, mode)
        make_flow_gif(start_index, end_index, t_predict, mode, name='u', fps_num=10)
        make_flow_gif(start_index, end_index, t_predict, mode, name='v', fps_num=10)
        make_flow_gif(start_index, end_index, t_predict, mode, name='p', fps_num=10)
    else:
        print("WRONG MODE NAME")
    print('ok')
