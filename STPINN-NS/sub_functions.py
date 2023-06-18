"""
core functions of parallel algorithm
并行算法核心函数
"""
import numpy as np
import scipy
import pandas as pd

from pinn_model import *
from learning_schdule import ChainedScheduler


class SubnetInfo:
    def __init__(self):
        self.row = 0
        self.col = 0
        self.t_index = 0
        self.sub_lb = np.zeros((1, 3))
        self.sub_ub = np.ones((1, 3))
        self.extended_lb = np.zeros((1, 3))
        self.extended_ub = np.zeros((1, 3))
        self.contain_data = True  # 对于时空并行应该默认每个子网络均含有数据 default True


# Used for process number decomposition of rows and columns
# 用于进程数分解行和列
def crack(integer):
    start = int(np.sqrt(integer))
    factor = integer / start
    while not is_integer(factor):
        start += 1
        factor = integer / start
    return min(int(factor), start), max(int(factor), start)


def is_integer(number):
    if int(number) == number:
        return True
    else:
        return False


# Each process determines the region where the subnetwork is located
# - rectangular partitioning - and distributes the number in the order of xyt
# 每一个进程确定该子网络所在的区域-矩形划分-按xyt的顺序分发序号
def identify_sub_location(rank, size, time_division, extension_ratio, low_bound, up_bound):
    space_division = int(size/time_division)
    rows, cols = crack(space_division)  # 每边（维度）有几个子网络 number of subnetwork along col and row
    time_index = rank // space_division
    space_index = rank % space_division
    # distribute locations to subnetwork
    # 给子网络分发位置坐标
    col_num = space_index % cols
    row_num = space_index // cols
    x_starts = np.linspace(low_bound[0, 0], up_bound[0, 0], cols + 1).reshape(1, -1)
    y_starts = np.linspace(low_bound[0, 1], up_bound[0, 1], rows + 1).reshape(1, -1)
    t_starts = np.linspace(low_bound[0, 2], up_bound[0, 2], time_division + 1).reshape(1, -1)
    # determine data exchange boundary
    # 划定数据交互边界
    sub_lb = np.array([x_starts[0, col_num], y_starts[0, row_num], t_starts[0, time_index]]).reshape(1, -1)
    sub_ub = np.array([x_starts[0, col_num + 1], y_starts[0, row_num + 1], t_starts[0, time_index+1]]).reshape(1, -1)
    # determine extended boundary
    # 划定延拓边界
    side_lengths = sub_ub - sub_lb
    cal_extended_lb = sub_lb - extension_ratio * side_lengths
    cal_extended_ub = sub_ub + extension_ratio * side_lengths
    extended_lb = np.maximum(low_bound, cal_extended_lb).reshape(1, -1)
    extended_ub = np.minimum(up_bound, cal_extended_ub).reshape(1, -1)
    # write information
    # 将信息写入对象
    sub_net_info = SubnetInfo()
    sub_net_info.row = row_num
    sub_net_info.col = col_num
    sub_net_info.t_index = time_index
    sub_net_info.sub_lb = sub_lb
    sub_net_info.sub_ub = sub_ub
    sub_net_info.extended_lb = extended_lb
    sub_net_info.extended_ub = extended_ub
    print("pinn subnet No:%d of %d nets" % (rank + 1, size))
    print("coordinates: (%d, %d, %d)" % (col_num, row_num, time_index))
    return sub_net_info


# Determine the boundary grid of each subnetwork
# 每一个子网络的边界网格确定
def identify_sub_boundaries(sub_net_info, pseudo_points_surface_set):
    x_min = sub_net_info.sub_lb[0, 0]
    y_min = sub_net_info.sub_lb[0, 1]
    t_min = sub_net_info.sub_lb[0, 2]
    x_max = sub_net_info.sub_ub[0, 0]
    y_max = sub_net_info.sub_ub[0, 1]
    t_max = sub_net_info.sub_ub[0, 2]
    xx = np.linspace(x_min, x_max, pseudo_points_surface_set[0, 0]).reshape(-1, 1)
    yy = np.linspace(y_min, y_max, pseudo_points_surface_set[0, 1]).reshape(-1, 1)
    tt = np.linspace(t_min, t_max, pseudo_points_surface_set[0, 2]).reshape(-1, 1)
    mesh_xt_x, mesh_xt_t = np.meshgrid(xx, tt)
    flatten_xt_x = mesh_xt_x.reshape(-1, 1)
    flatten_xt_t = mesh_xt_t.reshape(-1, 1)
    mesh_yt_y, mesh_yt_t = np.meshgrid(yy, tt)
    flatten_yt_y = mesh_yt_y.reshape(-1, 1)
    flatten_yt_t = mesh_yt_t.reshape(-1, 1)
    mesh_xy_x, mesh_xy_y = np.meshgrid(xx, yy)
    flatten_xy_x = mesh_xy_x.reshape(-1, 1)
    flatten_xy_y = mesh_xy_y.reshape(-1, 1)
    up_coord = np.concatenate((flatten_xt_x, np.tile(y_max, (flatten_xt_x.shape[0], 1)), flatten_xt_t), axis=1).astype(np.float32)
    down_coord = np.concatenate((flatten_xt_x, np.tile(y_min, (flatten_xt_x.shape[0], 1)), flatten_xt_t), axis=1).astype(np.float32)
    left_coord = np.concatenate((np.tile(x_min, (flatten_yt_y.shape[0], 1)), flatten_yt_y, flatten_yt_t), axis=1).astype(np.float32)
    right_coord = np.concatenate((np.tile(x_max, (flatten_yt_y.shape[0], 1)), flatten_yt_y, flatten_yt_t), axis=1).astype(np.float32)
    front_coord = np.concatenate((flatten_xy_x, flatten_xy_y, np.tile(t_min, (flatten_xy_x.shape[0], 1))), axis=1).astype(np.float32)
    back_coord = np.concatenate((flatten_xy_x, flatten_xy_y, np.tile(t_max, (flatten_xy_x.shape[0], 1))), axis=1).astype(np.float32)
    return up_coord, down_coord, left_coord, right_coord, front_coord, back_coord


# load data points-for subnetwork
# 子网络加载数据点
def load_data_points(sub_net_info, filename):
    # for each process, load data points within the extended boundary
    # 每一个进程读取相应区域延拓边界内的数据点
    lb = sub_net_info.extended_lb
    ub = sub_net_info.extended_ub
    # 读取原始数据
    data_mat = scipy.io.loadmat(filename)
    stack = data_mat['stack']  # N*4 (x,y,u,v)
    x = stack[:, 0].reshape(-1, 1)
    y = stack[:, 1].reshape(-1, 1)
    t = stack[:, 2].reshape(-1, 1)
    u = stack[:, 3].reshape(-1, 1)
    v = stack[:, 4].reshape(-1, 1)
    p = stack[:, 5].reshape(-1, 1)
    x_remain_index = (np.where((x >= lb[0, 0]) & (x <= ub[0, 0])))[0].reshape(-1, 1)
    y_remain_index = (np.where((y >= lb[0, 1]) & (y <= ub[0, 1])))[0].reshape(-1, 1)
    t_remain_index = (np.where((t >= lb[0, 2]) & (t <= ub[0, 2])))[0].reshape(-1, 1)
    remain_index_mid = (np.intersect1d(x_remain_index, y_remain_index)).reshape(-1, 1)
    remain_index = (np.intersect1d(remain_index_mid, t_remain_index)).reshape(-1, 1)
    x_sub = x[remain_index].reshape(-1, 1)
    y_sub = y[remain_index].reshape(-1, 1)
    t_sub = t[remain_index].reshape(-1, 1)
    u_sub = u[remain_index].reshape(-1, 1)
    v_sub = v[remain_index].reshape(-1, 1)
    p_sub = p[remain_index].reshape(-1, 1)
    if x_sub.shape[0] == 0:  # 若该块不含真实数据，则进行标记, 并触发警告 alert if no data is contained
        sub_net_info.contain_data = False
        print("ALERT!!! Sub net contains no data")
    x_sub_Ts = torch.tensor(x_sub, dtype=torch.float32)
    y_sub_Ts = torch.tensor(y_sub, dtype=torch.float32)
    t_sub_Ts = torch.tensor(t_sub, dtype=torch.float32)
    u_sub_Ts = torch.tensor(u_sub, dtype=torch.float32)
    v_sub_Ts = torch.tensor(v_sub, dtype=torch.float32)
    p_sub_Ts = torch.tensor(p_sub, dtype=torch.float32)
    return x_sub_Ts, y_sub_Ts, t_sub_Ts, u_sub_Ts, v_sub_Ts, p_sub_Ts, sub_net_info


# load collocation points-for subnetwork
# 子网络加载方程点——拉丁超立方
def load_equation_points_lhs(low_bound, up_bound, dimension, points):
    eqa_xyzt = low_bound + (up_bound - low_bound) * lhs(dimension, points)
    Eqa_points = torch.from_numpy(eqa_xyzt).float()
    Eqa_points = Eqa_points[torch.randperm(Eqa_points.size(0))]
    return Eqa_points


# load uniform collocation points
# 子网络加载方程点——均匀网格
def load_equation_points_uniform(low_bound, up_bound, dimension, points_set):
    if dimension == 2:
        axis_0 = np.linspace(low_bound[0, 0], up_bound[0, 0], points_set[0]).reshape(-1, 1)
        axis_1 = np.linspace(low_bound[0, 1], up_bound[0, 1], points_set[1]).reshape(-1, 1)
        mesh_0, mesh_1 = np.meshgrid(axis_0, axis_1)
        mesh_0_flatten = mesh_0.reshape(-1, 1)
        mesh_1_flatten = mesh_1.reshape(-1, 1)
        grid = np.concatenate((mesh_0_flatten, mesh_1_flatten), 1)
        Eqa_points = torch.from_numpy(grid).float()
        return Eqa_points
    if dimension == 3:
        axis_0 = np.linspace(low_bound[0, 0], up_bound[0, 0], points_set[0]).reshape(-1, 1)
        axis_1 = np.linspace(low_bound[0, 1], up_bound[0, 1], points_set[1]).reshape(-1, 1)
        axis_2 = np.linspace(low_bound[0, 2], up_bound[0, 2], points_set[2]).reshape(-1, 1)
        mesh_0, mesh_1, mesh_2 = np.meshgrid(axis_0, axis_1, axis_2)
        mesh_0_flatten = mesh_0.reshape(-1, 1)
        mesh_1_flatten = mesh_1.reshape(-1, 1)
        mesh_2_flatten = mesh_2.reshape(-1, 1)
        grid = np.concatenate((mesh_0_flatten, mesh_1_flatten, mesh_2_flatten), 1)
        Eqa_points = torch.from_numpy(grid).float()
        return Eqa_points
    if dimension == 4:
        axis_0 = np.linspace(low_bound[0, 0], up_bound[0, 0], points_set[0]).reshape(-1, 1)
        axis_1 = np.linspace(low_bound[0, 1], up_bound[0, 1], points_set[1]).reshape(-1, 1)
        axis_2 = np.linspace(low_bound[0, 2], up_bound[0, 2], points_set[2]).reshape(-1, 1)
        axis_3 = np.linspace(low_bound[0, 3], up_bound[0, 3], points_set[3]).reshape(-1, 1)
        mesh_0, mesh_1, mesh_2, mesh_3 = np.meshgrid(axis_0, axis_1, axis_2, axis_3)
        mesh_0_flatten = mesh_0.reshape(-1, 1)
        mesh_1_flatten = mesh_1.reshape(-1, 1)
        mesh_2_flatten = mesh_2.reshape(-1, 1)
        mesh_3_flatten = mesh_3.reshape(-1, 1)
        grid = np.concatenate((mesh_0_flatten, mesh_1_flatten, mesh_2_flatten, mesh_3_flatten), 1)
        Eqa_points = torch.from_numpy(grid).float()
        return Eqa_points
    else:
        print("Wrong Dimension of Equation Points")


# split batch and automatically fill batch size
# batch 划分与自动填充
def batch_split(Set, iter_num, dim=0):
    batches = torch.chunk(Set, iter_num, dim=dim)
    # 自动填充
    num_of_batches = len(batches)
    if num_of_batches == 1:
        batches = batches * iter_num
        return batches
    if num_of_batches < iter_num:
        for i in range(iter_num - num_of_batches):
            index = i % num_of_batches
            add_tuple = batches[-(index + 2):-(index + 1)]
            batches = batches + add_tuple
        return batches
    else:
        return batches


# preprocessing before training
# 训练前预处理
def pre_train_loading(filename_data, rank, size, time_division, dimension, N_eqa, extension_ratio, low_bound, up_bound, batch_size):
    filename_data_sub = filename_data.format(rank)
    sub_net_info = identify_sub_location(rank, size, time_division, extension_ratio, low_bound, up_bound)
    # load collocation points(only once)
    # 加载方程点(仅一次)
    Eqa_points = load_equation_points_lhs(sub_net_info.extended_lb, sub_net_info.extended_ub, dimension, N_eqa)
    Eqa_points_batches = torch.split(Eqa_points, batch_size, dim=0)
    iter_num = len(Eqa_points_batches)
    # 每一个进程读取数据
    # load data points(only once)
    # 加载真实数据点(仅一次)
    x_sub_Ts, y_sub_Ts, t_sub_Ts, u_sub_Ts, v_sub_Ts, p_sub_Ts, sub_net_info = load_data_points(sub_net_info, filename_data_sub)
    if x_sub_Ts.shape[0] > 0:
        data_sub = torch.cat([x_sub_Ts, y_sub_Ts, t_sub_Ts, u_sub_Ts, v_sub_Ts, p_sub_Ts], 1)
        True_dataset = data_sub[torch.randperm(data_sub.size(0))]  # 乱序
    else:
        True_dataset = None

    return True_dataset, Eqa_points_batches, sub_net_info, iter_num


# train data points, collocation points and interface points--with batch training
# 同时训练数据点，方程点，伪数据点———有batch
def train(inner_epochs, pinn_example, optimizer_all, scheduler_all, iter_num, Dataset, Eqa_points_batches,
          Pseudo_batches, Re, weight_data, weight_eqa, weight_pseudo, EPOCH, debug_key):
    loss_sum = np.array([0.0]).reshape(1, 1)
    loss_data = np.array([0.0]).reshape(1, 1)
    loss_eqa = np.array([0.0]).reshape(1, 1)
    loss_pseudo = np.array([0.0]).reshape(1, 1)
    # no batch split for data points
    # 数据点不分batch，每一次batch均训练所有的数据点
    x_train = Dataset[:, 0].reshape(-1, 1).requires_grad_(True).to(device)
    y_train = Dataset[:, 1].reshape(-1, 1).requires_grad_(True).to(device)
    t_train = Dataset[:, 2].reshape(-1, 1).requires_grad_(True).to(device)
    u_train = Dataset[:, 3].reshape(-1, 1).to(device)
    v_train = Dataset[:, 4].reshape(-1, 1).to(device)
    p_train = Dataset[:, 5].reshape(-1, 1).to(device)
    #  默认无数据点和无伪数据点并不会同时发生
    if Dataset is None:  # 无数据点时，不进行数据点的训练
        for epoch in range(inner_epochs):
            for batch_iter in range(iter_num):
                optimizer_all.zero_grad()
                x_pseudo = Pseudo_batches[batch_iter][:, 0].reshape(-1, 1).requires_grad_(True).to(device)
                y_pseudo = Pseudo_batches[batch_iter][:, 1].reshape(-1, 1).requires_grad_(True).to(device)
                t_pseudo = Pseudo_batches[batch_iter][:, 2].reshape(-1, 1).requires_grad_(True).to(device)
                u_pseudo = Pseudo_batches[batch_iter][:, 3].reshape(-1, 1).to(device)
                v_pseudo = Pseudo_batches[batch_iter][:, 4].reshape(-1, 1).to(device)
                p_pseudo = Pseudo_batches[batch_iter][:, 5].reshape(-1, 1).to(device)
                x_eqa = Eqa_points_batches[batch_iter][:, 0].reshape(-1, 1).requires_grad_(True).to(device)
                y_eqa = Eqa_points_batches[batch_iter][:, 1].reshape(-1, 1).requires_grad_(True).to(device)
                t_eqa = Eqa_points_batches[batch_iter][:, 2].reshape(-1, 1).requires_grad_(True).to(device)
                mse_equation = pinn_example.equation_mse_dimensionless(x_eqa, y_eqa, t_eqa, Re)
                mse_pseudo = pinn_example.data_mse(x_pseudo, y_pseudo, t_pseudo, u_pseudo, v_pseudo, p_pseudo)
                # 计算损失函数,不引入压强场的真实值
                loss = weight_pseudo * mse_pseudo + weight_eqa * mse_equation
                loss.backward()
                optimizer_all.step()
                with torch.autograd.no_grad():
                    loss_sum = loss.data.numpy()
                    loss_pseudo = mse_pseudo.data.numpy()
                    loss_eqa = mse_equation.data.numpy()
                    # 输出状态
                    if (batch_iter + 1) % iter_num == 0 and debug_key == 1:
                        print("EPOCH:", (EPOCH + 1), "  inner_iter:", epoch + 1, " Training-data Loss:",
                              round(float(loss.data), 8))
            scheduler_all.step()
        return loss_sum, loss_data, loss_eqa, loss_pseudo
    if Pseudo_batches[0].shape[0] == 0:  # 单网络时，不进行伪数据点的训练--------为兼容单进程训练
        for epoch in range(inner_epochs):
            for batch_iter in range(iter_num):
                optimizer_all.zero_grad()
                x_eqa = Eqa_points_batches[batch_iter][:, 0].reshape(-1, 1).requires_grad_(True).to(device)
                y_eqa = Eqa_points_batches[batch_iter][:, 1].reshape(-1, 1).requires_grad_(True).to(device)
                t_eqa = Eqa_points_batches[batch_iter][:, 2].reshape(-1, 1).requires_grad_(True).to(device)
                mse_data = pinn_example.data_mse(x_train, y_train, t_train, u_train, v_train, p_train)
                mse_equation = pinn_example.equation_mse_dimensionless(x_eqa, y_eqa, t_eqa, Re=Re)
                # 计算损失函数,不引入压强场的真实值
                loss = weight_data * mse_data + weight_eqa * mse_equation
                loss.backward()
                optimizer_all.step()
                with torch.autograd.no_grad():
                    loss_sum = loss.data.numpy()
                    loss_data = mse_data.data.numpy()
                    loss_eqa = mse_equation.data.numpy()
                    # 输出状态
                    if (batch_iter + 1) % iter_num == 0 and debug_key == 1:
                        print("EPOCH:", (EPOCH + 1), "  inner_iter:", epoch + 1, " Training-data Loss:",
                              round(float(loss.data), 8))
            scheduler_all.step()
        return loss_sum, loss_data, loss_eqa, loss_pseudo
    for epoch in range(inner_epochs):  # 其余情况，所有点都训练 normal situation, train all the points
        for batch_iter in range(iter_num):
            optimizer_all.zero_grad()
            x_eqa = Eqa_points_batches[batch_iter][:, 0].reshape(-1, 1).requires_grad_(True).to(device)
            y_eqa = Eqa_points_batches[batch_iter][:, 1].reshape(-1, 1).requires_grad_(True).to(device)
            t_eqa = Eqa_points_batches[batch_iter][:, 2].reshape(-1, 1).requires_grad_(True).to(device)
            x_pseudo = Pseudo_batches[batch_iter][:, 0].reshape(-1, 1).requires_grad_(True).to(device)
            y_pseudo = Pseudo_batches[batch_iter][:, 1].reshape(-1, 1).requires_grad_(True).to(device)
            t_pseudo = Pseudo_batches[batch_iter][:, 2].reshape(-1, 1).requires_grad_(True).to(device)
            u_pseudo = Pseudo_batches[batch_iter][:, 3].reshape(-1, 1).to(device)
            v_pseudo = Pseudo_batches[batch_iter][:, 4].reshape(-1, 1).to(device)
            p_pseudo = Pseudo_batches[batch_iter][:, 5].reshape(-1, 1).to(device)
            mse_data = pinn_example.data_mse(x_train, y_train, t_train, u_train, v_train, p_train)
            mse_equation = pinn_example.equation_mse_dimensionless(x_eqa, y_eqa, t_eqa, Re=Re)
            mse_pseudo = pinn_example.data_mse(x_pseudo, y_pseudo, t_pseudo, u_pseudo, v_pseudo, p_pseudo)
            # calculate loss
            # 计算损失函数
            loss = weight_data * mse_data + weight_eqa * mse_equation + weight_pseudo * mse_pseudo
            loss.backward()
            optimizer_all.step()
            with torch.autograd.no_grad():
                loss_sum = loss.data.numpy()
                loss_data = mse_data.data.numpy()
                loss_eqa = mse_equation.data.numpy()
                loss_pseudo = mse_pseudo.data.numpy()
                # print status
                # 输出状态
                if (batch_iter + 1) % iter_num == 0 and debug_key == 1:
                    print("EPOCH:", (EPOCH + 1), "  inner_iter:", epoch + 1, " Training-data Loss:",
                          round(float(loss.data), 8))
        scheduler_all.step()
    return loss_sum, loss_data, loss_eqa, loss_pseudo


# predict interface points
# 预测边界点，输出数据为Tensor——（x,y,u,v,p）,传递数据为numpy
def sub_predict_boundaries(pinn_example, up_coord, down_coord, left_coord, right_coord, front_coord, back_coord, Re):
    up_Tensor = torch.from_numpy(up_coord)
    down_Tensor = torch.from_numpy(down_coord)
    left_Tensor = torch.from_numpy(left_coord)
    right_Tensor = torch.from_numpy(right_coord)
    front_Tensor = torch.from_numpy(front_coord)
    back_Tensor = torch.from_numpy(back_coord)
    # 上边界
    x_up = up_Tensor[:, 0].reshape(-1, 1).requires_grad_(True)
    y_up = up_Tensor[:, 1].reshape(-1, 1).requires_grad_(True)
    t_up = up_Tensor[:, 2].reshape(-1, 1).requires_grad_(True)
    u_up, v_up, p_up = pinn_example.predict(x_up, y_up, t_up)
    up_set = torch.concat((x_up, y_up, t_up, u_up, v_up, p_up), dim=1)
    # 下边界
    x_down = down_Tensor[:, 0].reshape(-1, 1).requires_grad_(True)
    y_down = down_Tensor[:, 1].reshape(-1, 1).requires_grad_(True)
    t_down = down_Tensor[:, 2].reshape(-1, 1).requires_grad_(True)
    u_down, v_down, p_down = pinn_example.predict(x_down, y_down, t_down)
    down_set = torch.concat((x_down, y_down, t_down, u_down, v_down, p_down), dim=1)
    # 左边界
    x_left = left_Tensor[:, 0].reshape(-1, 1).requires_grad_(True)
    y_left = left_Tensor[:, 1].reshape(-1, 1).requires_grad_(True)
    t_left = left_Tensor[:, 2].reshape(-1, 1).requires_grad_(True)
    u_left, v_left, p_left = pinn_example.predict(x_left, y_left, t_left)
    left_set = torch.concat((x_left, y_left, t_left, u_left, v_left, p_left), dim=1)
    # 右边界
    x_right = right_Tensor[:, 0].reshape(-1, 1).requires_grad_(True)
    y_right = right_Tensor[:, 1].reshape(-1, 1).requires_grad_(True)
    t_right = right_Tensor[:, 2].reshape(-1, 1).requires_grad_(True)
    u_right, v_right, p_right = pinn_example.predict(x_right, y_right, t_right)
    right_set = torch.concat((x_right, y_right, t_right, u_right, v_right, p_right), dim=1)
    # 前边界
    x_front = front_Tensor[:, 0].reshape(-1, 1).requires_grad_(True)
    y_front = front_Tensor[:, 1].reshape(-1, 1).requires_grad_(True)
    t_front = front_Tensor[:, 2].reshape(-1, 1).requires_grad_(True)
    u_front, v_front, p_front = pinn_example.predict(x_front, y_front, t_front)
    front_set = torch.concat((x_front, y_front, t_front, u_front, v_front, p_front), dim=1)
    # 后边界
    x_back = back_Tensor[:, 0].reshape(-1, 1).requires_grad_(True)
    y_back = back_Tensor[:, 1].reshape(-1, 1).requires_grad_(True)
    t_back = back_Tensor[:, 2].reshape(-1, 1).requires_grad_(True)
    u_back, v_back, p_back = pinn_example.predict(x_back, y_back, t_back)
    back_set = torch.concat((x_back, y_back, t_back, u_back, v_back, p_back), dim=1)
    # tensor转换为numpy进行传递
    up_set = up_set.detach().numpy()
    down_set = down_set.detach().numpy()
    left_set = left_set.detach().numpy()
    right_set = right_set.detach().numpy()
    front_set = front_set.detach().numpy()
    back_set = back_set.detach().numpy()

    return up_set, down_set, left_set, right_set, front_set, back_set


# communicate subnetworks
# 将子网络在边界上预测出的值进行通信
# 不确定MPI是否能传输Tensor，故传输numpy
def communicate(rank, size, time_division, comm, sub_net_info, up_set, down_set, left_set, right_set, front_set, back_set):
    if size == 1:  # 单网络不进行交互 no communication if size==1
        boundary_set = np.empty((0, 6), dtype=np.float32)
        return boundary_set
    space_division = int(size / time_division)
    rows, cols = crack(space_division)
    max_col = cols - 1
    max_row = rows - 1
    max_t_index = time_division - 1
    up_temp = np.empty((0, up_set.shape[1]), dtype=np.float32)
    down_temp = np.empty((0, down_set.shape[1]), dtype=np.float32)
    left_temp = np.empty((0, left_set.shape[1]), dtype=np.float32)
    right_temp = np.empty((0, right_set.shape[1]), dtype=np.float32)
    front_temp = np.empty((0, front_set.shape[1]), dtype=np.float32)
    back_temp = np.empty((0, back_set.shape[1]), dtype=np.float32)
    # communicate upward
    # 向上通信，下方子块发送，上方子块接收
    if sub_net_info.row < max_row:
        id_receive_up = rank + cols
        comm.send(up_set, dest=id_receive_up, tag=1)
    if sub_net_info.row > 0:
        id_send_up = rank - cols
        up_temp = comm.recv(source=id_send_up, tag=1)
    comm.Barrier()  # 同步

    # communicate downward
    # 向下通信，上方子块发送，下方子块接收
    if sub_net_info.row > 0:
        id_receive_down = rank - cols
        comm.send(down_set, dest=id_receive_down, tag=2)
    if sub_net_info.row < max_row:
        id_send_down = rank + cols
        down_temp = comm.recv(source=id_send_down, tag=2)
    comm.Barrier()  # 同步

    # communicate leftward
    # 向左通信，右方子块发送，左方子块接收
    if sub_net_info.col > 0:
        id_receive_left = rank - 1
        comm.send(left_set, dest=id_receive_left, tag=3)
    if sub_net_info.col < max_col:
        id_send_left = rank + 1
        left_temp = comm.recv(source=id_send_left, tag=3)
    comm.Barrier()  # 同步

    # communicate rightward
    # 向右通信，左方子块发送，右方子块接收
    if sub_net_info.col < max_col:
        id_receive_right = rank + 1
        comm.send(right_set, dest=id_receive_right, tag=4)
    if sub_net_info.col > 0:
        id_send_right = rank - 1
        right_temp = comm.recv(source=id_send_right, tag=4)
    comm.Barrier()  # 同步

    # communicate forward
    # 向前通信，后方子块发送，前方子块接收
    if sub_net_info.t_index > 0:
        id_receive_front = rank - space_division
        comm.send(front_set, dest=id_receive_front, tag=5)
    if sub_net_info.t_index < max_t_index:
        id_send_front = rank + space_division
        front_temp = comm.recv(source=id_send_front, tag=5)
    comm.Barrier()  # 同步

    # communicate backward
    # 向后通信，前方子块发送，后方子块接收
    if sub_net_info.t_index < max_t_index:
        id_receive_back = rank + space_division
        comm.send(back_set, dest=id_receive_back, tag=6)
    if sub_net_info.t_index > 0:
        id_send_back = rank - space_division
        back_temp = comm.recv(source=id_send_back, tag=6)
    comm.Barrier()  # 同步

    boundary_set = np.concatenate((up_temp, down_temp, left_temp, right_temp, front_temp, back_temp), axis=0)
    return boundary_set


# make pseudo data points by averaging neighbouring prediction values on interface points
# 在边界上依据两相邻网络的均值制造伪数据点,并分割为batch
def make_pseudo_data(boundary_set, pinn_example, iter_num):
    boundary_set = torch.from_numpy(boundary_set)
    x_boundary = boundary_set[:, 0].reshape(-1, 1).requires_grad_(True).to(device)
    y_boundary = boundary_set[:, 1].reshape(-1, 1).requires_grad_(True).to(device)
    t_boundary = boundary_set[:, 2].reshape(-1, 1).requires_grad_(True).to(device)
    u_sub_1, v_sub_1, p_sub_1 = pinn_example.predict(x_boundary, y_boundary, t_boundary)
    u_ave = (boundary_set[:, 3].reshape(-1, 1) + u_sub_1) / 2
    v_ave = (boundary_set[:, 4].reshape(-1, 1) + v_sub_1) / 2
    p_ave = (boundary_set[:, 5].reshape(-1, 1) + p_sub_1) / 2
    Pseudo_dataset = torch.concat((x_boundary, y_boundary, t_boundary, u_ave, v_ave, p_ave), dim=1).detach()
    Pseudo_dataset = Pseudo_dataset[torch.randperm(Pseudo_dataset.size(0))]  # 乱序
    Pseudo_dataset_batches = batch_split(Pseudo_dataset, iter_num, dim=0)
    return Pseudo_dataset_batches


# record loss-for subnetwork
# 各个子网络记录子网络的loss
def record_loss_local(loss_sum, loss_data, loss_eqa, loss_pseudo, filename_loss_local):
    loss_sum_value = loss_sum.reshape(1, 1)
    loss_data_value = loss_data.reshape(1, 1)
    loss_eqa_value = loss_eqa.reshape(1, 1)
    loss_pseudo_value = loss_pseudo.reshape(1, 1)
    loss_set = np.concatenate((loss_sum_value, loss_data_value, loss_eqa_value, loss_pseudo_value), 1).reshape(1, -1)
    loss_save = pd.DataFrame(loss_set)
    loss_save.to_csv(filename_loss_local, index=False, header=False, mode='a')
    return loss_set


# print overall loss on root process
# 根节点输出总loss
def record_loss_overall(comm, rank, losses, filename_loss_overall):
    send_data = losses[-1, :]
    rec_data = comm.gather(send_data, root=0)
    if rank == 0:
        total_loss = np.array(rec_data).sum(axis=0).reshape(1, -1)
        loss_sum_all = total_loss[0, 0].item()
        # print("process {} gather all loss data {}...".format(rank, total_loss))
        loss_save = pd.DataFrame(total_loss)
        loss_save.to_csv(filename_loss_overall, index=False, header=False, mode='a')
        # 将总loss进行广播
    else:
        loss_sum_all = None
    loss_sum_all = comm.bcast(loss_sum_all, root=0)
    # print("process {} recv data {}...".format(rank, loss_sum))
    return loss_sum_all


