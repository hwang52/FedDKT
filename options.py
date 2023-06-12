import argparse
import os
from utils.param_aug import ParamDiffAug


def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)
    parser.add_argument('--data_name', type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_online_clients', type=int, default=8)
    parser.add_argument('--num_rounds', type=int, default=200)
    parser.add_argument('--server_lr', type=float, default=1.0)
    parser.add_argument('--num_epochs_local_training', type=int, default=10)
    parser.add_argument('--batch_size_local_training', type=int, default=64)
    parser.add_argument('--match_epoch', type=int, default=100)
    parser.add_argument('--crt_epoch', type=int, default=300)
    parser.add_argument('--batch_real', type=int, default=32)
    parser.add_argument('--num_of_feature', type=int, default=100)
    parser.add_argument('--lr_feature', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_size_test', type=int, default=100)
    parser.add_argument('--lr_local_training', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--non_iid_alpha', type=float, default=0.4) # 非独立同分布程度
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor') # 长尾分布程度
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--save_path', type=str, default=os.path.join(path_dir, 'result/'))
    parser.add_argument('--method', type=str, default='DSA', help='DC/DSA')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    # FedProx
    parser.add_argument('--mu', type=float, default=0.01)
    # FedAvgM
    parser.add_argument('--init_belta', type=float, default=0.97)
    # DP差分隐私
    parser.add_argument('--dp_mechanism', type=str, default='no_dp',
                        help='differential privacy mechanism') # 选择什么样的噪声进行差分隐私: no_dp代表不用dp, Laplace拉普拉斯噪声, Gaussian高斯噪声
    parser.add_argument('--dp_epsilon', type=float, default=10,
                        help='differential privacy epsilon') # 整差分隐私定义所能提供的”隐私量”,较小时对于隐私保护的要求就会比较高
    parser.add_argument('--dp_delta', type=float, default=1e-5,
                        help='differential privacy delta') # delta代表扰动, 用于限制模型行为任意改变的概率，通常设置为一个小的常数, 正则化
    parser.add_argument('--dp_clip', type=float, default=10,
                        help='differential privacy clip') # 样本对应的梯度裁剪到一个固定范围, 也就是通过clip来归一化
    
    args = parser.parse_args()
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    return args
