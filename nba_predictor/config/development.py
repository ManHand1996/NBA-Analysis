# config/development.py
"""
开发环境配置
"""
from typing import Dict, Any
from config.base import BaseConfig

class DevelopmentConfig(BaseConfig):
    """开发环境配置"""
    
    DEBUG = True
    
    @property
    def DATA(self) -> Dict[str, Any]:
        data_config = super().DATA.copy()
        data_config.update({
            'sample_size': 5000,  # 开发环境使用较少数据
            'use_sample_data': True,  # 使用示例数据
        })
        return data_config
    
    @property
    def MODELS(self) -> Dict[str, Dict[str, Any]]:
        models_config = super().MODELS.copy()
        # 开发环境使用较小的模型
        for model_name in models_config:
            if 'n_estimators' in models_config[model_name]:
                models_config[model_name]['n_estimators'] = 50
        return models_config