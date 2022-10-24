from after_train import predict_steady
from pinn_model import *
import matplotlib.pyplot as plt
import matplotlib as mpl


def compare_multi_steady(raw_filename, filelist, size_of_process, pinn_example, low_bound, up_bound):
    num_of_plots = len(filelist)
    x_raw, y_raw, u_raw, v_raw, feature_mat = read_2D_data_without_p(raw_filename)
    min_u = feature_mat.numpy()[1, 2]
    max_u = feature_mat.numpy()[0, 2]
    min_v = feature_mat.numpy()[1, 3]
    max_v = feature_mat.numpy()[0, 3]
    x_values = np.unique(x_raw).reshape(-1, 1)
    y_values = np.unique(y_raw).reshape(-1, 1)
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    u_predict_set = np.empty([num_of_plots+1, x_mesh.shape[0], x_mesh.shape[1]], dtype=float)
    v_predict_set = np.empty([num_of_plots+1, x_mesh.shape[0], x_mesh.shape[1]], dtype=float)
    u_predict_set[0] = u_raw.data.numpy().reshape(x_mesh.shape)
    v_predict_set[0] = v_raw.data.numpy().reshape(x_mesh.shape)
    for i in range(1, num_of_plots+1):
        x_i, y_i, u_i, v_i = predict_steady(raw_filename, filelist[i-1], size_of_process, pinn_example,
                                                                    low_bound, up_bound)
        u_predict_set[i] = u_i.reshape(x_mesh.shape)
        v_predict_set[i] = v_i.reshape(x_mesh.shape)
    plot_compare_multiset(u_predict_set, num_of_plots, min_u, max_u, name='u')
    plot_compare_multiset(v_predict_set, num_of_plots, min_v, max_v, name='v')
    return


def plot_compare_multiset(q_predict_set, num_of_plots, min_value, max_value, name='q'):
    fig_q = plt.figure(figsize=(19.2, 10.8))
    v_norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    for i in range(num_of_plots+1):
        plt.subplot(1, num_of_plots+1, i+1)
        plt.imshow(q_predict_set[i], cmap='jet', norm=v_norm)
        plt.title("Predict " + name)
        plt.ylabel('Y')
        plt.xlabel('X')
        if i == 0:
            plt.title("True " + name)
    position = fig_q.add_axes([0.92, 0.45, 0.01, 0.08])  # 位置[左,下,右,上]
    cb = plt.colorbar(fraction=0.005, pad=0.005, cax=position)
    colorbarfontdict = {"size": 15, "color": "k", 'family': 'Times New Roman'}
    cb.ax.set_title(name, fontdict=colorbarfontdict, pad=8)
    # plt.savefig(name+".png", dpi=600)
    # plt.savefig(name+".svg", dpi=600)
    plt.show()
    return


# 对于定常流场进行预测
if __name__ == '__main__':
    file_1 = './data/exp_b_1/sub_write'
    file_2 = './data/exp_b_2/sub_write'
    file_3 = './data/exp_b_3/sub_write'
    file_list = [file_1, file_2, file_3]
    low_bound = np.array([0, 0]).reshape(1, -1)  # 二维定常，仅含x,y
    up_bound = np.array([1, 1]).reshape(1, -1)    # 二维定常，仅含x,y
    layer_mat_psi = [2, 20, 20, 20, 20, 20, 2]  # 网络结构
    layer_mat = layer_mat_psi
    size_of_process = 4
    pinn_net = PINN_Net(layer_mat)
    pinn_net = pinn_net.to(device)
    filename_raw_data = './stack_Re100.mat'
    compare_multi_steady(filename_raw_data, file_list, size_of_process, pinn_net, low_bound, up_bound)
    print('ok')