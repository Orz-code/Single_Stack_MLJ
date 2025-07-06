import numpy as np

def read_infrared_image(file_path):
    """用于读取李昊转换过的红外图像的TXT文件，并且自动转置以符合拍摄角度

    Args:
        file_path (_type_): 文件名

    Returns:
        np.array: 图像的np矩阵，应为160*120 （高*宽）
    """
    file_txt = open(file_path)
    contents = file_txt.readlines()
    file_txt.close()
    temp_list = []
    for line in contents:
        temp_cur= list(map(lambda x:float(x.split(']')[-1]),line.split('\t')[:-1]))
        temp_list.append(temp_cur)
    return np.array(temp_list).T

def multi_gaussian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amp = params[i]
        cen = params[i+1]
        wid = params[i+2]
        y += amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))
    return y