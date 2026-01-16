# config/production.py
"""
生产环境配置
"""
from typing import Dict, Any
from config.base import BaseConfig

class ProductionConfig(BaseConfig):
    """生产环境配置"""
    
    DEBUG = False
    
    @property
    def DATA(self) -> Dict[str, Any]:
        data_config = super().DATA.copy()
        data_config.update({
            'use_sample_data': False,  # 不使用示例数据
            'data_sources': {
                'primary': '/data/nba/production/games_processed.csv',
                'backup': '/data/nba/backup/games_raw.csv',
            }
        })
        return data_config
    
    @property
    def LOGGING(self) -> Dict[str, Any]:
        logging_config = super().LOGGING.copy()
        logging_config.update({
            'level': 'WARNING',
            'file': {
                'enabled': True,
                'path': '/var/log/nba_predictor/nba_predictor.log',
                'max_size': 104857600,  # 100MB
                'backup_count': 10,
            }
        })
        return logging_config