# config/__init__.py
"""
配置工厂 - 类似Django的settings模块
"""

import os
from typing import Optional

# 环境变量
ENVIRONMENT = os.getenv('NBA_ENV', 'development').lower()

def get_config(env: Optional[str] = None):
    """获取配置实例"""
    if env is None:
        env = ENVIRONMENT
    
    if env == 'production':
        from config.production import ProductionConfig
        return ProductionConfig()
    # elif env == 'testing':
    #     from config.testing import TestingConfig
    #     return TestingConfig()
    else:  # development or default
        from config.development import DevelopmentConfig
        return DevelopmentConfig()

# 全局配置实例
config = get_config()

# 导出常用属性
PROJECT_NAME = config.PROJECT_NAME
PROJECT_VERSION = config.PROJECT_VERSION
DEBUG = config.DEBUG
PATHS = config.PATHS
FEATURES = config.FEATURES
MODELS = config.MODELS
DATA = config.DATA
LOGGING = config.LOGGING
EXPERIMENT = config.EXPERIMENT
EVALUATION = config.EVALUATION

# 导出便捷函数
def get_path(key: str):
    """获取路径的便捷函数"""
    return config.get_path(key)

def get_model_config(model_name: str):
    """获取模型配置的便捷函数"""
    return config.get_model_config(model_name)

def setup_directories():
    """设置目录的便捷函数"""
    return config.setup_directories()