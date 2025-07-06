import torch as t
import torch.nn as nn
import numpy as np
import time
import os
import onnx
import onnx.utils
import onnx.version_converter
import seaborn as sns
import matplotlib.pyplot as plt
from thop import profile
from torch.utils.data import DataLoader
from dynamic_model.configs import DividerTarget
from keys import CACHE_CHECKPOINT_DIR,ModelNameSegment

from logger_config import setup_logger
logger,file_prefix = setup_logger()

class BasicModule(t.nn.Module):
    '''
    封装了nn.Module，主要提供save和load两个方法
    '''

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = model_name_formatter(str(self.__class__))

    def load(self, path):
        '''
        可加载指定路径的模型
        '''
        try:
            self.load_state_dict(t.load(path))
        except:
            # 可能出现多余的内容，这时候只能去掉这些多余的内容
            self.load_state_dict(
                t.load(path),
                strict=False
            )
    def total_parameters(self):
        """计算模型所有参数的总数

        Returns:
            int: 模型所有参数总数
        """
        total_sum = sum(p.numel() for p in self.parameters())
        return total_sum
    
    def save(self, name=None,folder = None):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名，
        如AlexNet_0710_23:57:29.pth
        '''
        # 是否指定类名
        if name is None:
            model_name = self.model_name + '_' + time.strftime('%Y%m%d_%H_%M.pth')
        else:
            model_name = name +  '_' + time.strftime('%m%d_%H_%M.pth')
        # 是否指定文件夹
        if not folder :
            folder = CACHE_CHECKPOINT_DIR
        else:
            folder = folder
        fp_model = os.path.join(
                folder,
                model_name,
            )
        t.save(self.state_dict(), fp_model)
        logger.info('model saved at ' + fp_model)
        return fp_model
    
    def save_model(
            self,
            folder = None,
            num_episode = None,
        ):
        """暂存模型参数
        """
        # 可以指定文件夹
        if num_episode is None:
            self.save(name='DecisionNet_final',folder=folder)
        else:
            self.save(name='DecisionNet_{}episode'.format(num_episode),folder=folder)



class BasicDataset(t.utils.data.Dataset):
    def __init__(
        self,
        normalize_temp_out = False,
        add_input_noise = False,
        noise_percent = DividerTarget.noise
    ):
        super().__init__()
        self.normalize_temp_out = normalize_temp_out
        self.add_input_noise = add_input_noise
        self.noise_percent = noise_percent

    def process_response(self,response):
        if self.normalize_temp_out:
            divider = t.tensor([
                DividerTarget.voltage,
                DividerTarget.temperature
            ])
            return response/divider
        else:
            return response
    
    def apply_noise(self,inputs):
        # NOTE: 只对历史输入项施加噪声，不对设定值施加噪声
        # NOTE: 只有训练过程需要施加噪声，在展示阶段不需要施加噪声？
        # NOTE: 可以直接对所有项施加噪声？
        if self.add_input_noise:
            noise = t.randn(inputs.shape)
            feature_indicator = inputs.mean(axis = 0)*(self.noise_percent/100.)
            return inputs + (noise*feature_indicator)
        else:
            return inputs

def train_torch_model(
        model,
        dynamic_train_loader,
        num_epoch,
        dual_input = False,
        verbose = 1,
        return_loss_seq = False,
        lr = 1E-4,
    ):
    """给出指定的模型，在给定输入输出的情况下，训练torch模型
    基于现在的理解，torch模型的训练应当总体是流程一样的，所以我们只需要一个标准化的流程应该就可以了

    Args:
        model (torch.nn.Module): 要训练的模型
        dynamic_train_loader (torch.nn.DataLoader): 输入的数据，注意不要加num_workers
        num_epoch (int): 要训练的周期数
        verbose(int): 是否需要输出每个训练过程的loss

    Returns:
        torch.nn.Module, float: 返回训练好的模型，这时模型仍然在device上，同时返回训练的误差
    """
    # 将模型传输到指定设备
    running_loss = []
    loss_sequence = []
    process_sequence = []
    model = model.to(torch_device())
    model.train()
    # 定义损失函数和优化器
    criterion = nn.MSELoss().to(torch_device())
    optimizer = t.optim.Adam(
        model.parameters(),
        lr = lr
    )
    # 定义长度并开始训练
    total_step = len(dynamic_train_loader)
    if dual_input:
        for epoch in range(num_epoch):
            
            for i,(inputs,set_values,target) in enumerate(dynamic_train_loader):
                inputs = inputs.to(torch_device())
                set_values = set_values.to(torch_device())
                target = target.to(torch_device()).squeeze()

                output = model(inputs,set_values)
                loss = criterion(output,target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss.append(loss.item())
                if (i+1)%1000 == 0:
                    if return_loss_seq:
                        loss_sequence.append(np.mean(running_loss))
                        process_sequence.append(
                            epoch+1+((i+1)/total_step)
                        ) # 表示模型的训练进程
                    if verbose == 1:
                        print(
                            'Epoch [{},{}], Step [{}/{}], Loss: {:.4f}'.format(
                                epoch+1,num_epoch,i+1,total_step,np.mean(running_loss)
                            )
                        )
                    running_loss = []
    else:
        for epoch in range(num_epoch):
            
            for i,(inputs,target) in enumerate(dynamic_train_loader):
                inputs = inputs.to(torch_device())
                target = target.to(torch_device()).squeeze()

                output = model(inputs)
                loss = criterion(output,target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss.append(loss.item())
                if (i+1)%1000 == 0:
                    print(loss_sequence)
                    if return_loss_seq:
                        loss_sequence.append(np.mean(running_loss))
                        process_sequence.append(
                            epoch+1+((i+1)/total_step)
                        ) # 表示模型的训练进程
                    if verbose == 1:
                        print(
                            'Epoch [{},{}], Step [{}/{}], Loss: {:.4f}'.format(
                                epoch+1,num_epoch,i+1,total_step,np.mean(running_loss)
                            )
                        )
                    running_loss = []
    model.eval()
    if return_loss_seq:
        return model, np.mean(running_loss),loss_sequence,process_sequence
    return model, np.mean(running_loss)

def save_onnx_graph(model_class, dataset_class,model_steps,dual_input = False):
    """保存model为onnx格式，以便能够使用netron进行可视化，应当注意model和input的位置，应在一个设备上

    Args:
        model (nn.Module): 要保存的torch模型
        inputs (t.tensor): 要保存的模型的输入数据，应当注意这两者的位置，应在一个设备上
    """
    model = model_class(model_steps)
    data_set = dataset_class(model_steps)
    if dual_input:
        inputs,set_value,response = data_set.load_demonstration_data()
        inputs = (inputs,set_value)
    else:
        inputs,response = data_set.load_demonstration_data()

    model_name = model_name_formatter(str(model.__class__))
    
    fp_model_visualization = '../.cache/model_visualization/'
    check_folder(fp_model_visualization)
    prefix = fp_model_visualization + model_name + '_'
    fp_model = time.strftime(prefix + '%Y%m%d_%H_%M.onnx')

    t.onnx.export(
        model,
        inputs,
        fp_model,
        export_params=True,
        opset_version=8
    )
    onnx_model = onnx.load(fp_model)
    onnx.save(
        onnx.shape_inference.infer_shapes(
            onnx_model
        ),
        fp_model
    )
    print('graph saved at '+ fp_model)

def torch_device():
    """正常情况下一定要使用的方法，主要用在t.to(torch_device())，可以指定使用的GPU还是CPU

    Returns:
        str: 'cuda' or 'cpu'
    """
    try:
        t.cuda.set_device(0)
        device = t.device(
            'cuda' if t.cuda.is_available() else 'cpu'
        )
    except:
        device = 'cpu'
    return device


def mean_absolute_percentage_error(y_true, y_pred): 
    """计算平均绝对值百分比误差

    Args:
        y_true (list[float]): 真实值
        y_pred (list[float]): 预测值

    Returns:
        float: 平均绝对值百分比误差
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def plot_model_truth_pred_temp_out(y_true,y_pred):
    """画图展示真实出口温度与预测出口温度

    Args:
        y_true (list[float]): 真实值
        y_pred (list[float]): 预测值
    """
    plt.figure()
    plt.plot(range(len(y_true)),y_true)
    plt.plot(range(len(y_true)),y_pred)
    plt.legend(['truth','pred'])
    

def plot_model_error_temp_out(error,new_figure = True):
    """画图展示误差的分布情况"""
    if new_figure:
        plt.figure()
    bins = 150
    counts,_ = np.histogram(error,bins=bins)

    _ = sns.histplot(
        error,
        bins=bins,
        kde=True
    )
    plt.plot(
        [0,0],
        [0,max(counts)*1.1],
        'r'
    )

def check_folder(fp):
    if not os.path.exists(fp):
        os.makedirs(fp)

def model_name_formatter(model_name):
    """主要用于自动返回的model class type方法中有很多非法字符，无法用来保存，所以需要格式化一下

    Args:
        model_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    if "'" in model_name:
        model_name = model_name.split("'")[1]
    if '.' in model_name:
        model_name = model_name.split('.')[1]
    return model_name



def dynamic_model_name_formatter(
    num_epoch,
    model_steps,
    hidden_channels,
    polar,
    prob_attention,
    set_value_key,
    add_input_noise,
    noise_percent,
):
    model_name = 'Dynamic_model_{}steps_{}channels_{}epochs'.format(
        model_steps,hidden_channels,num_epoch
    )
    if polar:
        model_name += '_'+ModelNameSegment.polar
    if prob_attention:
        model_name += '_'+ModelNameSegment.prob_attention
    if set_value_key:
        model_name += '_'+ModelNameSegment.set_value_key
    if add_input_noise:
        model_name += '_'+ModelNameSegment.noise_percent.format(noise_percent)
    return model_name


def plot_model_results( 
        loss, inference_time,
        mape_v_dynamic,mape_t_dynamic,mape_v_normal,mape_t_normal,
        y_true_v,y_true_t,y_pred_v,y_pred_t,
        flops,params,
        normalize_temp_out
    ):
    """专门为展示同时预测电压和温度的模型输出比对设计的展示函数

    Args:
        loss (float): 模型的训练损失
        inference_time (float): 模型在验证集上的推理时间
        mape_v_dynamic (float): 动态验证集上的MAPE
        mape_t_dynamic (float): 动态验证集上的MAPE
        mape_v_normal (float): _description_
        mape_t_normal (float): _description_
        y_true_v (list[float]): 真实动态验证集的电压序列
        y_true_t (list[float]): 真实动态验证集的温度序列
        y_pred_v (list[float]): 预测的动态验证集上的电压序列
        y_pred_t (list[float]): 预测的动态验证集上的温度序列
        error_v (list[float]): 动态验证集的误差，用来展示误差的分布
        error_t (list[float]): _description_
    """
    plt.figure(figsize=(14,10))
    if normalize_temp_out:
        y_true_t *= DividerTarget.temperature
        y_pred_t *= DividerTarget.temperature
        y_true_v *= DividerTarget.voltage
        y_pred_v *= DividerTarget.voltage
    
    plt.subplot(2,2,1)
    plt.title('inference time: {:.4f}'.format(inference_time)+', model loss: {:.5f}'.format(loss))
    plt.plot(y_true_v)
    plt.plot(y_pred_v)
    plt.text(150,0.3,'mape_v_d: {:.3f}'.format(mape_v_dynamic))
    plt.text(150,0.1,'mape_v_n: {:.3f}'.format(mape_v_normal))
    plt.xlabel('voltage')



    plt.subplot(2,2,2)
    plt.title('FLOPS: {:E}'.format(flops)+', Parameters: {:E}'.format(params))
    plt.plot(y_true_v)
    plt.plot(y_pred_v)
    plt.xlim([800,1600])
    plt.ylim([1.6,2.1])
    plt.xlabel('voltage')
    # plot_model_error_temp_out(error_v,new_figure=False)
    # plt.xlabel('error of voltage inference')

    plt.subplot(2,2,3)
    plt.text(200,20,'mape_t_d: {:.3f}'.format(mape_t_dynamic))
    plt.text(200,10,'mape_t_n: {:.3f}'.format(mape_t_normal))
    plt.plot(y_true_t)
    plt.plot(y_pred_t)

    plt.xlabel('temperature outlet')


    plt.subplot(2,2,4)
    plt.plot(y_true_t)
    plt.plot(y_pred_t)
    plt.xlim([1000,1500])
    plt.ylim([
        65,
        90
    ])
    plt.xlabel('temperature outlet')