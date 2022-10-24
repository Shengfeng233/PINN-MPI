"""
2维定常
lid-driven cave
空间分布式并行——矩形子网络划分
非周期性边界条件并行
分布式并行计算
采用同时训练的架构，不分batch
without data of p
"""
import numpy as np
from mpi4py import MPI
from sub_functions import *

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()
start = MPI.Wtime()
filename_model = './sub_data/sub_write/{:02d}NS_model_train.pt'
filename_save_model = filename_model.format(rank)
filename_data = './sub_data/sub_read/stack_Re100_{:02d}.mat'  # 训练数据
filename_loss = './sub_data/sub_write/{:02d}loss.csv'
filename_loss = filename_loss.format(rank)
filename_loss_overall = './loss.csv'
device = torch.device("cpu")  # 使用CPU并行

# 重要参数
low_bound = np.array([0, 0]).reshape(1, -1)  # 二维定常，仅含x,y
up_bound = np.array([1, 1]).reshape(1, -1)    # 二维定常，仅含x,y
dimension = 2
Re = 100  # 雷诺数
layer_mat_psi = [2, 20, 20, 2]  # 网络结构
layer_mat = layer_mat_psi
learning_rate = 0.001  # 学习率
T_0 = 10
T_mul = 2
iter_num = 10
inner_epochs = 1  # 训练周期数——小周期数
outer_epochs = 2500  # 训练周期数——大周期数
weight_of_data = 10
weight_of_eqa = 1
decay_rate = 0.99  # 用以控制最大学习率
N_eqa = 32
pseudo_points_per_line = 4
debug_key = 0
local_debug_key = 0

if local_debug_key == 1:
    size = 9  # debug only
    rank = 4  # debug only

# 训练前加载工作(仅加载一次)
# 加载数据点
# 加载方程点
# 加载伪数据点(边界)坐标
pinn_net = PINN_Net(layer_mat)
pinn_net = pinn_net.to(device)  # 每一个进程创建一个网络
True_dataset, Eqa_points_batches, sub_net_info = pre_train_loading(filename_data, rank, size, dimension, N_eqa, low_bound, up_bound, iter_num)
up_coord, down_coord, left_coord, right_coord = identify_sub_boundaries(sub_net_info, pseudo_points_per_line)

# 优化器和学习率衰减设置-1个优化器和衰减器
optimizer_all = torch.optim.Adam(pinn_net.parameters(), lr=learning_rate)
scheduler_all = ChainedScheduler(optimizer_all, T_0=T_0, T_mul=T_mul, eta_min=0, gamma=decay_rate, max_lr=learning_rate, warmup_steps=2)
# 用以记录各部分损失的列表-数据loss，伪数据loss，方程loss
# losses = np.empty((0, 4), dtype=float)

# 训练主循环
for EPOCH in range(outer_epochs):
    # 对伪数据点(对边界)的预测
    up_set, down_set, left_set, right_set = sub_predict_boundaries(pinn_net, up_coord, down_coord, left_coord, right_coord)

    # 子网络信息交互-(x,y,u,v,p)
    boundary_set = communicate(rank, size, comm, sub_net_info, up_set, down_set, left_set, right_set, EPOCH)

    # 制造伪数据点
    Pseudo_dataset = make_pseudo_data(boundary_set, pinn_net)
    if debug_key == 1:
        print(Pseudo_dataset.shape)

    # 制造数据点（真实数据点+伪数据点）集
    Dataset_batches = batch_of_data(True_dataset, Pseudo_dataset, iter_num)
    # 对数据点，伪数据点，方程点进行同时训练
    loss_data, loss_eqa = train(inner_epochs, pinn_net, optimizer_all, scheduler_all, Dataset_batches,
                                                  Eqa_points_batches, Re, weight_of_data, weight_of_eqa, EPOCH, debug_key)

    # loss记录
    loss_set = record_loss_local(loss_data, loss_eqa, filename_loss)  # 记录子网络loss
    record_loss_overall(comm, rank, loss_set, filename_loss_overall)  # 记录总loss
    comm.Barrier()


# 每个子网络保存模型
torch.save(pinn_net.state_dict(), filename_save_model)
end = MPI.Wtime()
if rank == 0:  # 根节点输出
    print("all networks finished")
    print("Time used: %d s" % (end - start))

