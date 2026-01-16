# src/utils/logger.py
"""
日志工具类
"""

import logging
import sys
from pathlib import Path
from config import LOGGING, get_path

def setup_logger(name: str = 'nba_predictor') -> logging.Logger:
    """设置日志记录器"""
    
    # 获取配置
    log_config = LOGGING
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_config['level']))
    
    # 清除现有处理器
    logger.handlers.clear()
    
    # 控制台处理器
    if log_config['console']['enabled']:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_config['level']))
        console_format = logging.Formatter(log_config['console']['format'])
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_config['file']['enabled']:
        from logging.handlers import RotatingFileHandler
        
        log_path = Path(log_config['file']['path'])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=log_config['file']['max_size'],
            backupCount=log_config['file']['backup_count']
        )
        file_handler.setLevel(getattr(logging, log_config['level']))
        file_format = logging.Formatter(log_config['format'])
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger

# 全局日志记录器
logger = setup_logger()