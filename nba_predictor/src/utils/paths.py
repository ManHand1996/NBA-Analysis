# src/utils/paths.py
"""
路径工具类
"""

from pathlib import Path
from config import get_path

class PathManager:
    """路径管理器"""
    
    def __init__(self):
        self.base_dir = get_path('BASE')
        self.src_dir = get_path('SRC')
        self.data_dir = get_path('DATA')
        self.models_dir = get_path('MODELS')
        self.pipelines_dir = get_path('PIPELINES')
        self.reports_dir = get_path('REPORTS')
        self.logs_dir = get_path('LOGS')
    
    def get_data_path(self, filename: str = None, subdir: str = None) -> Path:
        """获取数据文件路径"""
        path = self.data_dir
        if subdir:
            path = path / subdir
        if filename:
            path = path / filename
        return path
    
    def get_model_path(self, experiment_id: str = None, model_type: str = None) -> Path:
        """获取模型路径"""
        path = self.models_dir
        if experiment_id:
            path = path / experiment_id
            if model_type:
                path = path / 'models' / model_type
        return path
    
    def get_pipeline_path(self, pipeline_name: str) -> Path:
        """获取管道路径"""
        return self.pipelines_dir / pipeline_name
    
    def ensure_directories(self) -> None:
        """确保所有目录存在"""
        directories = [
            self.data_dir / 'raw',
            self.data_dir / 'processed',
            self.data_dir / 'external',
            self.models_dir,
            self.pipelines_dir,
            self.reports_dir,
            self.logs_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def list_experiments(self) -> list:
        """列出所有实验"""
        experiments = []
        if self.models_dir.exists():
            for item in self.models_dir.iterdir():
                if item.is_dir() and item.name.startswith('exp_'):
                    experiments.append(item.name)
        return sorted(experiments)

# 全局实例
path_manager = PathManager()