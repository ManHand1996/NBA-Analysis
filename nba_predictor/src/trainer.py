# src/multi_model_trainer.py

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
import numpy as np

# src/multi_model_trainer.py
"""
纯粹的多模型训练器 - 专注于特征处理和模型训练
不包含版本管理功能，版本管理由NBAModelPipeline负责
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle

class MultiModelTrainer:
    """
    纯粹的多模型训练器
    职责：特征处理、数据划分、模型训练
    不负责：版本管理、结果保存、实验记录
    """
    
    def __init__(self, use_time_split=True, use_scaler=True):
        """
        初始化训练器
        
        Args:
            use_time_split: 是否使用时间感知划分（True=按赛季划分，False=随机划分）
            use_scaler: 是否使用标准化
        """
        self.use_time_split = use_time_split
        self.use_scaler = use_scaler
        self.scaler = None
        self.feature_names = None
        self.data_info = {}
    
    def prepare_features(self, df, test_seasons=3):
        """
        准备特征和标签，支持两种划分方式
        
        Args:
            df: 包含特征的DataFrame，必须有'season'列
            test_seasons: 用最后几个赛季作为测试集（仅当use_time_split=True时有效）
            
        Returns:
            X_train, X_test, y_train, y_test, feature_names, scaler
        """
        print(f"🔧 准备特征数据...")
        print(f"   时间划分: {self.use_time_split}")
        print(f"   标准化: {self.use_scaler}")
        
        # 验证必需的列
        required_columns = ['home_win', 'season']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"DataFrame必须包含'{col}'列")
        
        # 确定特征列（排除标识列和标签列）
        exclude_columns = [
            'home_team_id', 'away_team_id',
            'home_win', 'game_date', 'season', 'pts_diff'
            ]
        
        feature_names = [col for col in df.columns if col not in exclude_columns]
        self.feature_names = feature_names
        
        # 划分数据集
        if self.use_time_split:
            X_train, X_test, y_train, y_test = self._time_based_split(
                df, feature_names, test_seasons
            )
        else:
            X_train, X_test, y_train, y_test = self._random_split(
                df, feature_names
            )
        
        # 标准化处理
        scaler = None
        if self.use_scaler:
            X_train, X_test, scaler = self._apply_scaling(X_train, X_test)
            self.scaler = scaler
        
        # 记录数据信息
        self.data_info = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': len(feature_names),
            'train_pos_ratio': y_train.mean(),
            'test_pos_ratio': y_test.mean(),
            'use_time_split': self.use_time_split,
            'use_scaler': self.use_scaler
        }
        
        print(f"✅ 特征准备完成")
        print(f"   训练集: {len(X_train)} 样本")
        print(f"   测试集: {len(X_test)} 样本")
        print(f"   特征数: {len(feature_names)}")
        print(f"   标准化器: {'已创建' if scaler else '未使用'}")
        
        return X_train, X_test, y_train, y_test, feature_names, scaler
    
    def _time_based_split(self, df, feature_names, test_seasons):
        """时间感知划分（无数据泄漏）"""
        # 按赛季排序
        seasons = sorted(df['season'].unique())
        
        if len(seasons) <= test_seasons:
            raise ValueError(f"赛季数量({len(seasons)})少于测试赛季数({test_seasons})")
        
        test_season_cutoff = seasons[-test_seasons]
        
        # 创建掩码
        train_mask = df['season'] < test_season_cutoff
        test_mask = df['season'] >= test_season_cutoff
        
        # 提取数据
        X_train = df.loc[train_mask, feature_names]
        y_train = df.loc[train_mask, 'home_win']
        X_test = df.loc[test_mask, feature_names]
        y_test = df.loc[test_mask, 'home_win']
        
        print(f"   时间划分: 训练集({seasons[0]}-{test_season_cutoff-1}) "
              f"测试集({test_season_cutoff}-{seasons[-1]})")
        
        return X_train, X_test, y_train, y_test
    
    def _random_split(self, df, feature_names):
        """随机划分（用于快速验证）"""
        X = df[feature_names]
        y = df['home_win']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   随机划分: 训练集{len(X_train)}样本, 测试集{len(X_test)}样本")
        
        return X_train, X_test, y_train, y_test
    
    def _apply_scaling(self, X_train, X_test):
        """应用标准化"""
        # 确定需要标准化的列（数值列，排除标记列）
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        exclude_cols = [col for col in numeric_cols 
                       if any(keyword in col for keyword in 
                             ['exists', 'is_', 'games_played', 'streak', 'count'])]
        
        scale_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(scale_cols) == 0:
            print("   警告: 没有找到需要标准化的特征")
            return X_train, X_test, None
        
        scaler = StandardScaler()
        
        # 复制数据避免警告
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # 拟合并转换训练集
        X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
        # 转换测试集（使用训练集的统计量）
        X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])
        
        print(f"   标准化: 处理了{len(scale_cols)}个特征")
        
        return X_train_scaled, X_test_scaled, scaler
    
    def train_xgboost(self, X_train, y_train, params=None):
        """
        训练XGBoost模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            params: 模型参数，如果为None使用默认参数
            
        Returns:
            训练好的XGBoost模型
        """
        print(f"🌲 训练XGBoost模型...")
        
        if params is None:
            params = { 
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        print(f"   ✅ XGBoost训练完成")
        print(f"      参数: n_estimators={params['n_estimators']}, "
              f"max_depth={params['max_depth']}, lr={params['learning_rate']}")
        
        return model
    
    def train_random_forest(self, X_train, y_train, params=None):
        """
        训练随机森林模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            params: 模型参数，如果为None使用默认参数
            
        Returns:
            训练好的随机森林模型
        """
        print(f"🌳 训练随机森林模型...")
        
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': 'balanced'
            }
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        print(f"   ✅ 随机森林训练完成")
        print(f"      参数: n_estimators={params['n_estimators']}, "
              f"max_depth={params['max_depth']}")
        
        return model
    
    def train_gradient_boosting(self, X_train, y_train, params=None):
        """
        训练梯度提升模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            params: 模型参数，如果为None使用默认参数
            
        Returns:
            训练好的梯度提升模型
        """
        print(f"📈 训练梯度提升模型...")
        
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'random_state': 42
            }
        
        model = GradientBoostingClassifier(**params)
       

        model.fit(X_train, y_train)
        
        print(f"   ✅ 梯度提升训练完成")
        print(f"      参数: n_estimators={params['n_estimators']}, "
              f"max_depth={params['max_depth']}, lr={params['learning_rate']}")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """
        评估模型性能
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            评估指标字典
        """
        from sklearn.metrics import (accuracy_score, precision_score, 
                                   recall_score, f1_score, roc_auc_score)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        if y_prob is not None:
            metrics['auc'] = roc_auc_score(y_test, y_prob)
        
        return metrics
    
    def get_feature_importance(self, model, feature_names=None):
        """
        获取特征重要性
        
        Args:
            model: 训练好的模型
            feature_names: 特征名称列表，如果为None使用self.feature_names
            
        Returns:
            DataFrame包含特征和重要性分数
        """
        if feature_names is None:
            feature_names = self.feature_names
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            print("警告: 该模型没有feature_importances_属性")
            return None
    
    def save_scaler(self, filepath):
        """保存标准化器"""
        if self.scaler is not None:
            with open(filepath, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"✅ 标准化器已保存: {filepath}")
            return True
        else:
            print("警告: 没有标准化器可保存")
            return False
    
    def load_scaler(self, filepath):
        """加载标准化器"""
        try:
            with open(filepath, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ 标准化器已加载: {filepath}")
            return True
        except Exception as e:
            print(f"❌ 加载标准化器失败: {e}")
            return False
    
    def save_model(self, model, filepath):
        """保存模型"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ 模型已保存: {filepath}")
            return True
        except Exception as e:
            print(f"❌ 保存模型失败: {e}")
            return False
    
    def load_model(self, filepath):
        """加载模型"""
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ 模型已加载: {filepath}")
            return model
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            return None

