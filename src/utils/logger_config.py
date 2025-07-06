# logger_config.py
import logging
from keys import LOGGER_RL_TRAIN_DIR,TimeForms,LOGGER_RL_TEST_DIR

import os
import time

# TODO:这里应该根据不同的项目内容去初始化logger的文件夹，暂时不应该这么处理

def setup_logger():
    
    # 创建一个logger
    logger = logging.getLogger(
        'AWES'
    )
    logger.setLevel(logging.DEBUG)
    # 如果logger已经有handler，就不再添加
    if (logger.hasHandlers()):
        logger.handlers.clear()
    # 创建一个handler，用于写入日志文件
    file_name = 'AWES_RL_training_log_'+ time.strftime('%Y%m%d_%H_%M_%S.log')
    file_prefix = file_name.split('.log')[0]
    folder = os.path.join(
            LOGGER_RL_TRAIN_DIR,
            time.strftime(TimeForms.date),
            file_prefix # 增加一级文件夹，以方便查阅
        )
    if not os.path.exists(folder):
        os.makedirs(folder)

    fh = logging.FileHandler(
        os.path.join(
            folder,
            file_name
        )
    )
    fh.setLevel(logging.DEBUG)

    # 创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger,file_prefix


def setup_tester_logger():
    # 创建一个logger
    logger = logging.getLogger(
        'AWES_TEST'
    )
    logger.setLevel(logging.DEBUG)
    # 如果logger已经有handler，就不再添加
    if (logger.hasHandlers()):
        logger.handlers.clear()
    # 创建一个handler，用于写入日志文件
    file_name = 'TEST_log_'+ time.strftime('%Y%m%d_%H_%M_%S.log')
    file_prefix = file_name.split('.log')[0]
    folder = os.path.join(
            LOGGER_RL_TEST_DIR,
            time.strftime(TimeForms.date),
            file_prefix # 增加一级文件夹，以方便查阅
        )
    if not os.path.exists(folder):
        os.makedirs(folder)

    fh = logging.FileHandler(
        os.path.join(
            folder,
            file_name
        )
    )
    fh.setLevel(logging.DEBUG)

    # 创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger,file_prefix