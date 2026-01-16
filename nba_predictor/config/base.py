# config/base.py
"""
基础配置类 - 类似Django的settings.py
"""

import os
from pathlib import Path
from typing import Dict, List, Any

# 项目根目录
BASE_DIR = Path(__file__).parent.parent

class BaseConfig:
    """基础配置类"""
    
    # 项目信息
    PROJECT_NAME = "NBA胜负预测系统"
    PROJECT_VERSION = "1.0.0"
    PROJECT_DESCRIPTION = "基于机器学习的NBA比赛胜负预测系统"
    
    # 环境设置
    DEBUG = True
    SEED = 42
    
    # 路径配置
    @property
    def PATHS(self) -> Dict[str, Path]:
        return {
            'BASE': BASE_DIR,
            'SRC': BASE_DIR / 'src',
            'DATA': BASE_DIR / 'data',
            # 'MODELS': BASE_DIR / 'model_experiments',
            'PIPELINES': BASE_DIR / 'pipelines',
            # 'REPORTS': BASE_DIR / 'reports',
            'LOGS': BASE_DIR / 'logs',
            'TESTS': BASE_DIR / 'tests',
            'NOTEBOOKS': BASE_DIR / 'notebooks',
        }
    
    # 实验配置
    @property
    def EXPERIMENT(self) -> Dict[str, Any]:
        return {
            'test_seasons': 3,           # 测试赛季数
            'validation_seasons': 1,     # 验证赛季数
            'min_season_games': 20,      # 最小赛季比赛数
            'random_state': self.SEED,
        }
    
    # 特征配置
    @property
    def FEATURES(self) -> Dict[str, Any]:
        return {
            'use_time_split': True,      # 使用时间划分
            'use_scaler': True,          # 使用标准化
            'scaler_type': 'standard',   # 标准化类型: standard/minmax/robust
            'fillna_strategy': 'neutral', # 填充策略: neutral/mean/median/zero
            
            # 需要排除的特征列
            'exclude_columns': [
                'home_team_id', 'away_team_id',
                'home_last_game_date', 'away_last_game_date',
                'game_id', 'match_id',
            ],
            
            # 标识列（不用于训练）
            'identifier_columns': [
                'home_win', 'game_date', 'season', 'point_diff',
                'home_team', 'away_team',
            ],
        }
    
    # 模型配置
    @property
    def MODELS(self) -> Dict[str, Dict[str, Any]]:
        return {
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.SEED,
                'eval_metric': 'logloss',
                'use_label_encoder': False,
            },
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': self.SEED,
                'n_jobs': -1,
                'class_weight': 'balanced',
            },
            'gradient_boosting': {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'random_state': self.SEED,
            },
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': self.SEED,
                'solver': 'lbfgs',
            }
        }
    
    # 评估配置
    @property
    def EVALUATION(self) -> Dict[str, Any]:
        return {
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            'cross_validation_folds': 5,
            'scoring': 'roc_auc',
            'thresholds': {
                'high_confidence': 0.7,
                'medium_confidence': 0.6,
                'low_confidence': 0.55,
            }
        }
    
    # 数据配置
    @property
    def DATA(self) -> Dict[str, Any]:
        return {
            'sample_size': 10000,        # 示例数据大小
            'train_test_ratio': 0.8,     # 训练测试比例
            'min_games_per_team': 10,    # 每队最小比赛数
            'data_sources': {
                'primary': 'data/processed/nba_games_processed.csv',
                'backup': 'data/raw/nba_games_raw.csv',
                'external': 'data/external/',
            },
            'preprocessing': {
                'impute_strategy': 'median',
                'outlier_threshold': 3.0,
                'categorical_encoding': 'label',
            }
        }
    
    # 日志配置
    @property
    def LOGGING(self) -> Dict[str, Any]:
        return {
            'level': 'INFO' if not self.DEBUG else 'DEBUG',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': {
                'enabled': True,
                'path': BASE_DIR / 'logs' / 'nba_predictor.log',
                'max_size': 10485760,  # 10MB
                'backup_count': 5,
            },
            'console': {
                'enabled': True,
                'format': '%(levelname)s - %(message)s',
            }
        }
    
    # API配置（如果以后需要）
    @property
    def API(self) -> Dict[str, Any]:
        return {
            'host': 'localhost',
            'port': 8000,
            'debug': self.DEBUG,
            'workers': 1,
            'timeout': 30,
        }
    
    def setup_directories(self) -> None:
        """创建所有必要的目录"""
        for name, path in self.PATHS.items():
            if name not in ['BASE', 'SRC']:  # 这些应该已存在
                path.mkdir(parents=True, exist_ok=True)
                print(f"📁 确保目录存在: {name} -> {path}")
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """获取指定模型的配置"""
        return self.MODELS.get(model_name, {})
    
    def get_path(self, key: str) -> Path:
        """获取路径配置"""
        return self.PATHS.get(key, BASE_DIR)