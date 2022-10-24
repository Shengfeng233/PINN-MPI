import numpy as np
import scipy
import pandas as pd

from pinn_model import *
from learning_schdule import ChainedScheduler


class SubnetInfo:
    def __init__(self):
        self.row = 0
        self.col = 0
        self.sub_lb = np.zeros((1, 2))
        self.sub_ub = np.ones((1, 2))
        self.contain_data = True


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


# 每一个进程确定该子网络所在的区域-矩形划分
def identify_sub_location(rank, size, low_bound, up_bound):
    rows, cols = crack(size)  # 每边（维度）有几个子网络
    col_num = rank % cols
    row_num = rank // cols
    x_starts = np.linspace(low_bound[0, 0], up_bound[0, 0], cols + 1).reshape(1, -1)
    y_starts = np.linspace(low_bound[0, 1], up_bound[0, 1], rows + 1).reshape(1, -1)
    sub_lb = np.array([x_starts[0, col_num], y_starts[0, row_num]]).reshape(1, -1)
    sub_up = np.array([x_starts[0, col_num + 1], y_starts[0, row_num + 1]]).reshape(1, -1)
    sub_net_info = SubnetInfo()
    sub_net_info.row = row_num
    sub_net_info.col = col_num
    sub_net_info.sub_lb = sub_lb
    sub_net_info.sub_ub = sub_up
    print("pinn subnet No:%d of %d nets" % (rank+1, size))
    print("coordinates: (%d, %d)" % (row_num, col_num))
    return sub_net_info


# 每一个子网络的边界网格确定
def identify_sub_boundaries(sub_net_info, pseudo_points_per_line):
    x_min = sub_net_info.sub_lb[0, 0]
    y_min = sub_net_info.sub_lb[0, 1]
    x_max = sub_net_info.sub_ub[0, 0]
    y_max = sub_net_info.sub_ub[0, 1]
    xx = np.linspace(x_min, x_max, pseudo_points_per_line).reshape(-1, 1)
    yy = np.linspace(y_min, y_max, pseudo_points_per_line).reshape(-1, 1)
    up_coord = np.concatenate((xx, np.tile(y_max, (pseudo_points_per_line, 1))), axis=1).astype(np.float32)
    down_coord = np.concatenate((xx, np.tile(y_min, (pseudo_points_per_line, 1))), axis=1).astype(np.float32)
    left_coord = np.concatenate((np.tile(x_min, (pseudo_points_per_line, 1)), yy), axis=1).astype(np.float32)
    right_coord = np.concatenate((np.tile(x_max, (pseudo_points_per_line, 1)), yy), axis=1).astype(np.float32)

    return up_coord, down_coord, left_coord, right_coord


# 子网络加载数据点
def load_data_points(sub_net_info, filename):
    # 每一个进程仅读取相应区域的数据点
    lb = sub_net_info.sub_lb
    up = sub_net_info.sub_ub
    # 读取原始数据
    data_mat = scipy.io.loadmat(filename)
    stack = data_mat['stack']  # N*4 (x,y,u,v)
    x = stack[:, 0].reshape(-1, 1)
    y = stack[:, 1].reshape(-1, 1)
    u = stack[:, 2].reshape(-1, 1)
    v = stack[:, 3].reshape(-1, 1)
    x_remain_index = (np.where((x >= lb[0, 0]) & (x <= up[0, 0])))[0].reshape(-1, 1)
    y_remain_index = (np.where((y >= lb[0, 1]) & (y <= up[0, 1])))[0].reshape(-1, 1)
    remain_index = (np.intersect1d(x_remain_index, y_remain_index)).reshape(-1, 1)
    x_sub = x[remain_index].reshape(-1, 1)
    y_sub = y[remain_index].reshape(-1, 1)
    u_sub = u[remain_index].reshape(-1, 1)
    v_sub = v[remain_index].reshape(-1, 1)
    if x_sub.shape[0] == 0:  # 若该块不含真实数据，则进行标记
        sub_net_info.contain_data = False
    x_sub_Ts = torch.tensor(x_sub, dtype=torch.float32)
    y_sub_Ts = torch.tensor(y_sub, dtype=torch.float32)
    u_sub_Ts = torch.tensor(u_sub, dtype=torch.float32)
    v_sub_Ts = torch.tensor(v_sub, dtype=torch.float32)
    return x_sub_Ts, y_sub_Ts, u_sub_Ts, v_sub_Ts, sub_net_info


# 子网络加载方程点
def load_equation_points(low_bound, up_bound, dimension, points):
    eqa_xyzt = low_bound + (up_bound - low_bound) * lhs(dimension, points)
    per = np.random.permutation(eqa_xyzt.shape[0])
    new_xyzt = eqa_xyzt[per, :]
    Eqa_points = torch.from_numpy(new_xyzt).float()
    return Eqa_points


# batch 划分与自动填充
def batch_split(Set, iter_num, dim=0):
    batches = torch.chunk(Set, iter_num, dim=dim)
    # 自动填充
    num_of_batches = len(batches)
    if num_of_batches < iter_num:
        add_tuple = batches[-(iter_num - num_of_batches):]
        final_batches = batches + add_tuple
        return final_batches
    else:
        return batches


# 训练前预处理
def pre_train_loading(filename_data, rank, size, dimension, N_eqa, low_bound, up_bound, iter_num):
    filename_data_sub = filename_data.format(rank)
    sub_net_info = identify_sub_location(rank, size, low_bound, up_bound)
    # 每一个进程读取数据
    # 加载真实数据点(仅一次)
    x_sub_Ts, y_sub_Ts, u_sub_Ts, v_sub_Ts, sub_net_info = load_data_points(sub_net_info, filename_data_sub)
    True_dataset = shuffle_data(x_sub_Ts, y_sub_Ts, u_sub_Ts, v_sub_Ts)
    # 加载方程点(仅一次)
    Eqa_points = load_equation_points(sub_net_info.sub_lb, sub_net_info.sub_ub, dimension, N_eqa)
    Eqa_points_batches = batch_split(Eqa_points, iter_num, dim=0)
    return True_dataset, Eqa_points_batches, sub_net_info


# 同时训练数据点，方程点，伪数据点———有batch
def train(inner_epochs, pinn_example, optimizer_all, scheduler_all, Dataset_batches, Eqa_points_batches, Re, weight_data, weight_eqa, EPOCH, debug_key):
    loss_data = np.array([0.0]).reshape(1, 1)
    loss_eqa = np.array([0.0]).reshape(1, 1)
    for epoch in range(inner_epochs):
        for batch_iter in range(len(Dataset_batches)):
            optimizer_all.zero_grad()
            x_train = Dataset_batches[batch_iter][:, 0].reshape(-1, 1).clone().requires_grad_(True).to(device)
            y_train = Dataset_batches[batch_iter][:, 1].reshape(-1, 1).clone().requires_grad_(True).to(device)
            u_train = Dataset_batches[batch_iter][:, 2].reshape(-1, 1).clone().requires_grad_(True).to(device)
            v_train = Dataset_batches[batch_iter][:, 3].reshape(-1, 1).clone().requires_grad_(True).to(device)
            x_eqa = Eqa_points_batches[batch_iter][:, 0].reshape(-1, 1).clone().requires_grad_(True).to(device)
            y_eqa = Eqa_points_batches[batch_iter][:, 1].reshape(-1, 1).clone().requires_grad_(True).to(device)
            mse_predict = pinn_example.data_mse_psi(x_train, y_train, u_train, v_train)
            mse_equation = pinn_example.equation_mse_dimensionless_psi(x_eqa, y_eqa, Re=Re)
            # 计算损失函数,不引入压强场的真实值
            loss = weight_data * mse_predict + weight_eqa * mse_equation
            loss.backward()
            optimizer_all.step()
            with torch.autograd.no_grad():
                loss_data = mse_predict.data.numpy()
                loss_eqa = mse_equation.data.numpy()
                # 输出状态
                if (epoch + 1) % 5 == 0 and debug_key == 1:
                    print("EPOCH:", (EPOCH + 1), "  inner_iter:", epoch + 1, " Training-data Loss:",
                          round(float(loss.data), 8))
            scheduler_all.step()
    return loss_data, loss_eqa


# 预测边界点，输出数据为Tensor——（x,y,u,v,p）,传递数据为numpy
def sub_predict_boundaries(pinn_example, up_coord, down_coord, left_coord, right_coord):
    up_Tensor = torch.from_numpy(up_coord)
    down_Tensor = torch.from_numpy(down_coord)
    left_Tensor = torch.from_numpy(left_coord)
    right_Tensor = torch.from_numpy(right_coord)
    # 上边界
    x_up = up_Tensor[:, 0].reshape(-1, 1).clone().requires_grad_(True)
    y_up = up_Tensor[:, 1].reshape(-1, 1).clone().requires_grad_(True)
    u_up, v_up, p_up = pinn_example.predict(x_up, y_up)
    up_set = torch.concat((x_up, y_up, u_up, v_up, p_up), dim=1)
    # 下边界
    x_down = down_Tensor[:, 0].reshape(-1, 1).clone().requires_grad_(True)
    y_down = down_Tensor[:, 1].reshape(-1, 1).clone().requires_grad_(True)
    u_down, v_down, p_down = pinn_example.predict(x_down, y_down)
    down_set = torch.concat((x_down, y_down, u_down, v_down, p_down), dim=1)
    # 左边界
    x_left = left_Tensor[:, 0].reshape(-1, 1).clone().requires_grad_(True)
    y_left = left_Tensor[:, 1].reshape(-1, 1).clone().requires_grad_(True)
    u_left, v_left, p_left = pinn_example.predict(x_left, y_left)
    left_set = torch.concat((x_left, y_left, u_left, v_left, p_left), dim=1)
    # 右边界
    x_right = right_Tensor[:, 0].reshape(-1, 1).clone().requires_grad_(True)
    y_right = right_Tensor[:, 1].reshape(-1, 1).clone().requires_grad_(True)
    u_right, v_right, p_right = pinn_example.predict(x_right, y_right)
    right_set = torch.concat((x_right, y_right, u_right, v_right, p_right), dim=1)

    # tensor转换为numpy进行传递
    up_set = up_set.detach().numpy()
    down_set = down_set.detach().numpy()
    left_set = left_set.detach().numpy()
    right_set = right_set.detach().numpy()

    return up_set, down_set, left_set, right_set


# 将子网络在边界上预测出的值进行通信
# 不确定MPI是否能传输Tensor，故传输numpy
def communicate(rank, size, comm, sub_net_info, up_set, down_set, left_set, right_set, EPOCH):
    if size == 1:  # 单网络不进行交互
        boundary_set = np.empty((0, 5), dtype=np.float32)
        return boundary_set
    rows, cols = crack(size)
    max_col = cols - 1
    max_row = rows - 1
    up_temp = np.empty((0, up_set.shape[1]), dtype=np.float32)
    down_temp = np.empty((0, down_set.shape[1]), dtype=np.float32)
    left_temp = np.empty((0, left_set.shape[1]), dtype=np.float32)
    right_temp = np.empty((0, right_set.shape[1]), dtype=np.float32)
    # 向上通信，下方子块发送，上方子块接收
    if sub_net_info.row < max_row:
        id_receive = rank + cols
        comm.send(up_set, dest=id_receive, tag=10000+EPOCH)
    if sub_net_info.row > 0:
        id_send = rank - cols
        up_temp = comm.recv(source=id_send, tag=10000+EPOCH)
    comm.Barrier()  # 同步

    # 向下通信，上方子块发送，下方子块接收
    if sub_net_info.row > 0:
        id_receive = rank - cols
        comm.send(down_set, dest=id_receive, tag=20000+EPOCH)
    if sub_net_info.row < max_row:
        id_send = rank + cols
        down_temp = comm.recv(source=id_send, tag=20000+EPOCH)
    comm.Barrier()  # 同步

    # 向左通信，右方子块发送，左方子块接收
    if sub_net_info.col > 0:
        id_receive = rank - 1
        comm.send(left_set, dest=id_receive, tag=30000+EPOCH)
    if sub_net_info.col < max_col:
        id_send = rank + 1
        left_temp = comm.recv(source=id_send, tag=30000+EPOCH)
    comm.Barrier()  # 同步

    # 向右通信，左方子块发送，右方子块接收
    if sub_net_info.col < max_col:
        id_receive = rank + 1
        comm.send(right_set, dest=id_receive, tag=40000+EPOCH)
    if sub_net_info.col > 0:
        id_send = rank - 1
        right_temp = comm.recv(source=id_send, tag=40000+EPOCH)
    comm.Barrier()  # 同步

    boundary_set = np.concatenate((up_temp, down_temp, left_temp, right_temp), axis=0)
    return boundary_set


# 在边界上依据两相邻网络的均值制造伪数据点,并分割为batch
def make_pseudo_data(boundary_set, pinn_example):
    boundary_set = torch.from_numpy(boundary_set)
    x_boundary = boundary_set[:, 0].reshape(-1, 1).clone().requires_grad_(True).to(device)
    y_boundary = boundary_set[:, 1].reshape(-1, 1).clone().requires_grad_(True).to(device)
    u_sub_1, v_sub_1, p_sub_1 = pinn_example.predict(x_boundary, y_boundary)
    u_ave = (boundary_set[:, 2].reshape(-1, 1)+u_sub_1)/2
    v_ave = (boundary_set[:, 3].reshape(-1, 1)+v_sub_1)/2
    p_ave = (boundary_set[:, 4].reshape(-1, 1)+p_sub_1)/2
    # 此处为将数据点和伪数据点等效，未计入p的值
    Pseudo_dataset = torch.concat((x_boundary, y_boundary, u_ave, v_ave), dim=1).detach()
    # Pseudo_dataset = Pseudo_dataset[torch.randperm(Pseudo_dataset.size(0))]  # 乱序
    return Pseudo_dataset


# 将数据点和伪数据点地位等同，合并为Dataset
def make_data(True_dataset, Pseudo_dataset):
    if Pseudo_dataset.shape[0] == 0:  # 兼容单进程
        Dataset = True_dataset
        return Dataset
    Dataset = torch.concat((True_dataset, Pseudo_dataset), dim=0)
    Dataset = Dataset[torch.randperm(Dataset.size(0))]  # 整体乱序
    return Dataset


# 将dataset分成小batch
def batch_of_data(True_dataset, Pseudo_dataset, iter_num):
    Dataset = make_data(True_dataset, Pseudo_dataset)
    Dataset_batches = batch_split(Dataset, iter_num, dim=0)
    return Dataset_batches


# 各个子网络记录子网络的loss
def record_loss_local(loss_data, loss_eqa, filename_loss_local):
    loss_data_value = loss_data.reshape(1, 1)
    loss_eqa_value = loss_eqa.reshape(1, 1)
    loss_sum = loss_data_value+loss_eqa_value
    loss_set = np.concatenate((loss_sum, loss_data_value, loss_eqa_value), 1).reshape(1, -1)
    loss_save = pd.DataFrame(loss_set)
    loss_save.to_csv(filename_loss_local, index=False, header=False, mode='a')
    return loss_set


# 根节点输出总loss
def record_loss_overall(comm, rank, losses, filename_loss_overall):
    send_data = losses[-1, :]
    rec_data = comm.gather(send_data, root=0)
    if rank == 0:
        total_loss = np.array(rec_data).sum(axis=0).reshape(1, -1)
        # print("process {} gather all loss data {}...".format(rank, total_loss))
        loss_save = pd.DataFrame(total_loss)
        loss_save.to_csv(filename_loss_overall, index=False, header=False, mode='a')

