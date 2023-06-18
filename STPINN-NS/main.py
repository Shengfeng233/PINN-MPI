"""
2-dimensional unsteady NS equation
Spatio-temporal parallel----rectangle domain decomposition
overlapping domain decomposition
simultaneous training
batch is split in iterations
2维非定常-NS方程
时空并行——矩形子网络划分
子网络有重叠域(Overlapping domain)
采用同时训练的架构，划分batch
x y t u v p
"""
import numpy as np
from mpi4py import MPI
from sub_functions import *


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()
start = MPI.Wtime()
dir_name_write = './sub_data/sub_write'
filename_model = './sub_data/sub_write/{:02d}NS_model_train.pt'
filename_save_model = filename_model.format(rank)
filename_data = './sub_data/sub_read/stack_unsteady_{:02d}.mat'  # 训练数据 training data
filename_loss = './sub_data/sub_write/{:02d}loss.csv'
filename_loss = filename_loss.format(rank)
filename_loss_overall = './loss.csv'
device = torch.device("cpu")  # 使用CPU并行 parallel using CPU
torch.manual_seed(3407)
np.random.seed(3407)

# important parameters
# 重要参数
time_division = 10
extension_ratio = 0.05
low_bound_all = np.array([1.0000, -2.0000, 0.0]).reshape(1, -1)  # x,y,t
up_bound_all = np.array([4.98998, 2.0000, 42.93729]).reshape(1, -1)  # x,y,t
dimension = 2 + 1
Re = 3900  # 雷诺数 Reynolds number
layer_mat_1 = [3, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 3]  # 网络结构 neural network
layer_mat_2 = [3, 80, 80, 80, 80, 80, 3]   # 网络结构 neural network
layer_mat_3 = [3, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 3]   # 网络结构 neural network
layer_mat = layer_mat_3
learning_rate = 0.001  # 学习率 learning rate
T_0 = 100  # 学习率控制参数 parameter used to control learning rate
T_mult = 2
batch_size = 128
inner_epochs = 1  # 训练周期数——小周期数 epoch-inner epoch
EPOCH_of_annealing = 7
epoch_list = generate_epoch_list(T_0, T_mult, EPOCH_of_annealing)  # 回火周期数组 annealing epochs
outer_epochs = epoch_list[0, -1]  # 训练周期数 epoch-outer epoch
weight_of_data = 1
weight_of_eqa = 1
weight_pseudo = 1
decay_rate = 0.9  # 用以控制最大学习率 to control learning rate decaying
N_eqa = 8000
pseudo_points_surface_set = np.array([25, 25, 10]).reshape(1, -1).astype(int)
debug_key = 0
local_debug_key = 0

if local_debug_key == 1:
    size = 40  # debug only
    rank = 4  # debug only

# pretraining-loading data
# loading data points
# loading collocation points
# loading pseudo-data points(interface points)
# only load once
# 训练前加载工作(仅加载一次)
# 加载数据点
# 加载方程点
# 加载伪数据点(边界)坐标
True_dataset, Eqa_points_batches, sub_net_info, iter_num = pre_train_loading(filename_data, rank, size, time_division,
                                                                             dimension, N_eqa, extension_ratio,
                                                                             low_bound_all, up_bound_all, batch_size)

up_coord, down_coord, left_coord, right_coord, front_coord, back_coord = identify_sub_boundaries(sub_net_info,
                                                                                                 pseudo_points_surface_set)
pinn_net = PINN_Net(layer_mat, sub_net_info.sub_lb, sub_net_info.sub_ub)
pinn_net = pinn_net.to(device)  # 每一个进程创建一个网络 allocate a separate network for each process

# 优化器和学习率衰减设置- optimizer and learning rate schedule
optimizer_all = torch.optim.Adam(pinn_net.parameters(), lr=learning_rate)
scheduler_all = ChainedScheduler(optimizer_all, T_0=T_0, T_mul=T_mult, eta_min=0, gamma=decay_rate, max_lr=learning_rate,
                                 warmup_steps=0)

# 训练主循环 main loop
for EPOCH in range(outer_epochs):
    # 对伪数据点(对边界)的预测 predict on the interface points
    up_set, down_set, left_set, right_set, front_set, back_set = sub_predict_boundaries(pinn_net, up_coord, down_coord,
                                                                                        left_coord, right_coord,
                                                                                        front_coord, back_coord, Re)

    # 子网络信息交互-(x,y,u,v,p) communicate among sub neural networks
    boundary_set = communicate(rank, size, time_division, comm, sub_net_info, up_set, down_set, left_set, right_set,
                               front_set, back_set)

    # 制造伪数据点 generate pseudo-data points
    Pseudo_dataset_batches = make_pseudo_data(boundary_set, pinn_net, iter_num)

    # 对数据点，伪数据点，方程点进行同时训练 train
    loss_sum, loss_data, loss_eqa, loss_pseudo = train(inner_epochs, pinn_net, optimizer_all, scheduler_all,
                                                       iter_num, True_dataset, Eqa_points_batches,
                                                       Pseudo_dataset_batches, Re, weight_of_data, weight_of_eqa,
                                                       weight_pseudo, EPOCH, debug_key)

    # loss记录 record loss
    loss_set = record_loss_local(loss_sum, loss_data, loss_eqa, loss_pseudo, filename_loss)  # 记录子网络loss
    loss_sum_all = record_loss_overall(comm, rank, loss_set, filename_loss_overall)  # 记录总loss至根节点,并分发至各个子节点
    comm.Barrier()

    # 在每一个回火周期保存各个子网络模型 save model at the end of each annealing epoch
    if np.isin(EPOCH + 1, epoch_list).item():
        annealing_num = np.where(EPOCH + 1 == epoch_list)[1].item() + 1
        dir_name = dir_name_write + '/annealing' + str(annealing_num)
        os.makedirs(dir_name, exist_ok=True)
        torch.save(pinn_net.state_dict(), dir_name + '/{:02d}NS_model_train.pt'.format(rank))

# 每个子网络保存模型 save the final model
torch.save(pinn_net.state_dict(), filename_save_model)
end = MPI.Wtime()
if rank == 0:  # 根节点输出 print on the root process
    print("all networks finished")
    print("Time used: %d s" % (end - start))
