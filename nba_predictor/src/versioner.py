# src/multi_model_versioner.py
import os
import json
import pickle
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, 
                           confusion_matrix, roc_curve, auc)

class MultiModelVersioner:
    """支持多模型实验的版本管理器"""
    
    def __init__(self, base_dir="model_experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.save_molde_name = 'model.pkl'
        
        # 初始化索引文件
        self.index_file = self.base_dir / "experiments_index.csv"
        if not self.index_file.exists():
            pd.DataFrame(columns=[
                'experiment_id', 'version', 'model_type', 
                'accuracy', 'auc', 'best_model', 'created_at', 'notes'
            ]).to_csv(self.index_file, index=False)
    
    def create_experiment(self, experiment_name, data_info, feature_info):
        """
        创建一个新的实验
        
        Args:
            experiment_name: 实验名称，如 "playoff_feature_test"
            data_info: 数据信息字典
            feature_info: 特征信息字典
        """
        # 创建实验目录
        experiment_id = self._generate_experiment_id()
        exp_dir = self.base_dir / experiment_id
        exp_dir.mkdir(parents=True)
        
        # 创建子目录结构
        (exp_dir / "models").mkdir()
        (exp_dir / "results").mkdir()
        # (exp_dir / "features").mkdir()
        # (exp_dir / "configs").mkdir()
        # (exp_dir / "artifacts").mkdir()
        
        # 保存实验配置
        experiment_config = {
            'experiment_id': experiment_id,
            'experiment_name': experiment_name,
            'created_at': datetime.now().isoformat(),
            'data_info': data_info,
            'feature_info': feature_info,
            'models_tested': [],
            'best_model': None,
            'status': 'running'
        }
        
        with open(exp_dir / "experiment_config.json", 'w') as f:
            json.dump(experiment_config, f, indent=2)
        
        print(f"✅ 实验 {experiment_name} 已创建")
        print(f"   实验ID: {experiment_id}")
        print(f"   路径: {exp_dir}")
        
        return experiment_id, exp_dir
    
    def save_model_result(self, experiment_id, model_type, model, 
                         X_train, y_train, X_test, y_test, 
                         model_config, scaler=None, notes=""):
        """
        保存单个模型的训练结果
        
        Args:
            experiment_id: 实验ID
            model_type: 模型类型，如 'xgboost', 'random_forest'
            model: 训练好的模型对象
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据
            model_config: 模型配置参数
            notes: 模型说明
        """
        exp_dir = self.base_dir / experiment_id
        
        # 为这个模型创建版本
        model_version = self._get_next_model_version(exp_dir, model_type)
        model_dir = exp_dir / "models" / model_type / f"v{model_version}"
        model_dir.mkdir(parents=True)
        
        # 计算评估指标
        metrics = self._compute_model_metrics(model, X_test, y_test)
        
        # 保存模型
        # self._save_model_files(model, model_dir, model_type, model_version)
        self._save_model_files(model, model_dir, model_type, model_version, scaler)
        # 保存模型元数据
        metadata = {
            'experiment_id': experiment_id,
            'model_type': model_type,
            'model_version': model_version,
            'created_at': datetime.now().isoformat(),
            'metrics': metrics,
            'model_config': model_config,
            'notes': notes,
            'data_info': {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': X_train.shape[1],
                'train_pos_ratio': y_train.mean(),
                'test_pos_ratio': y_test.mean()
            }
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 保存特征重要性
        if hasattr(model, 'feature_importances_'):
            self._save_feature_importance(model, X_train.columns, model_dir)
        
        # 生成可视化结果
        self._generate_model_artifacts(model, X_test, y_test, model_dir)
        
        # 更新实验配置
        self._update_experiment_config(exp_dir, model_type, model_version, metrics)
        
        # 更新全局索引
        self._update_experiments_index(experiment_id, model_type, model_version, metrics, notes)
        
        print(f"  ✅ {model_type} v{model_version} 已保存")
        print(f"     准确率: {metrics['accuracy']:.2%}, AUC: {metrics.get('auc', 'N/A')}")
        
        return {
            'model_type': model_type,
            'model_version': model_version,
            'metrics': metrics,
            'model_dir': model_dir
        }
    
    def compare_models_in_experiment(self, experiment_id):
        """比较实验中的所有模型"""
        exp_dir = self.base_dir / experiment_id
        
        if not exp_dir.exists():
            raise ValueError(f"实验 {experiment_id} 不存在")
        
        # 加载所有模型的元数据
        models_data = []
        models_dir = exp_dir / "models"
        
        for model_type in models_dir.iterdir():
            if model_type.is_dir():
                for model_version in model_type.iterdir():
                    if model_version.is_dir():
                        meta_file = model_version / "metadata.json"
                        if meta_file.exists():
                            with open(meta_file, 'r') as f:
                                metadata = json.load(f)
                            models_data.append(metadata)
        
        if not models_data:
            print("实验中没有找到模型数据")
            return None
        
        # 创建比较表格
        comparison_df = pd.DataFrame([
            {
                'Model': f"{d['model_type']} v{d['model_version']}",
                'Accuracy': d['metrics']['accuracy'],
                'AUC': d['metrics'].get('auc', 0),
                'Precision': d['metrics']['precision'],
                'Recall': d['metrics']['recall'],
                'F1': d['metrics']['f1_score'],
                'Train Samples': d['data_info']['train_samples'],
                'Features': d['data_info']['feature_count'],
                'Config': str(d['model_config'])[:50] + '...'
            }
            for d in models_data
        ])
        
        # 排序并显示
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        print("\n" + "="*100)
        print(f"实验 {experiment_id} - 模型比较")
        print("="*100)
        print(comparison_df.to_string(index=False))
        
        # 生成比较图表
        self._plot_model_comparison(comparison_df, exp_dir)
        
        # 保存比较结果
        comparison_df.to_csv(exp_dir / "results" / "model_comparison.csv", index=False)
        
        # 识别最佳模型
        best_model = comparison_df.iloc[0]
        print(f"\n🏆 最佳模型: {best_model['Model']}")
        print(f"   准确率: {best_model['Accuracy']:.2%}")
        print(f"   AUC: {best_model['AUC']:.3f}")
        
        return comparison_df
    
    def get_best_model(self, experiment_id):
        """获取实验中的最佳模型"""
        exp_dir = self.base_dir / experiment_id
        
        # 加载实验配置
        config_file = exp_dir / "experiment_config.json"
        if not config_file.exists():
            raise ValueError(f"实验配置不存在: {config_file}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        if not config.get('best_model'):
            print("实验尚未完成或没有最佳模型标记")
            return None
        
        best_model_info = config['best_model']
        model_type = best_model_info['model_type']
        model_version = best_model_info['model_version']
        
        # 加载最佳模型
        model_dir = exp_dir / "models" / model_type / f"v{model_version}"
        model_path = model_dir / "model.pkl"
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # 加载元数据
        meta_path = model_dir / "metadata.json"
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"🎯 最佳模型: {model_type} v{model_version}")
        print(f"   准确率: {metadata['metrics']['accuracy']:.2%}")
        print(f"   路径: {model_dir}")
        
        return {
            'model': model,
            'metadata': metadata,
            'model_dir': model_dir
        }
    
    def _generate_experiment_id(self):
        """生成实验ID（时间戳）"""
        return datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    
    def _get_next_model_version(self, exp_dir, model_type):
        """获取模型的下一个版本号"""
        model_type_dir = exp_dir / "models" / model_type
        if not model_type_dir.exists():
            return 1
        
        versions = []
        for item in model_type_dir.iterdir():
            if item.is_dir() and item.name.startswith('v'):
                try:
                    version = int(item.name[1:])
                    versions.append(version)
                except:
                    pass
        
        return max(versions) + 1 if versions else 1
    
    def _compute_model_metrics(self, model, X_test, y_test):
        """计算模型评估指标"""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        if y_prob is not None:
            metrics['auc'] = roc_auc_score(y_test, y_prob)
        
        return metrics
    
    def _save_model_files(self, model, model_dir, model_type, model_version, scaler=None):
        """保存模型文件"""

        import pickle
    
        # 保存完整的模型包
        model_package = {
            'model': model,
            'scaler': scaler,  # 保存scaler
            'model_type': model_type,
            'model_version': model_version,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(model_dir / self.save_molde_name, 'wb') as f:
            pickle.dump(model_package, f)
        
        # # 可选：单独保存模型（如果需要）
        # with open(model_dir / "model_only.pkl", 'wb') as f:
        #     pickle.dump(model, f)
        
        # # 单独保存scaler
        # if scaler is not None:
        #     with open(model_dir / "scaler.pkl", 'wb') as f:
        #         pickle.dump(scaler, f)

        # 如果是XGBoost，额外保存原生格式
        if model_type == 'xgboost' and hasattr(model, 'save_model'):
            model.save_model(model_dir / "model.json")
    
    def _save_feature_importance(self, model, feature_names, model_dir):
        """保存特征重要性"""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv(model_dir / "feature_importance.csv", index=False)
        
        # 保存前20个特征的图表
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(20).sort_values('importance')
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig(model_dir / "feature_importance.png", dpi=150)
        plt.close()
    
    def _generate_model_artifacts(self, model, X_test, y_test, model_dir):
        """生成模型可视化结果"""
        artifacts_dir = model_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        try:
            # 混淆矩阵
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(artifacts_dir / "confusion_matrix.png", dpi=150)
            plt.close()
            
            # ROC曲线
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (AUC = {roc_auc:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(artifacts_dir / "roc_curve.png", dpi=150)
                plt.close()
        except Exception as e:
            print(f"警告: 生成可视化结果时出错: {e}")
    
    def _update_experiment_config(self, exp_dir, model_type, model_version, metrics):
        """更新实验配置"""
        config_file = exp_dir / "experiment_config.json"
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # 添加模型到已测试列表
        model_info = {
            'model_type': model_type,
            'model_version': model_version,
            'accuracy': metrics['accuracy'],
            'auc': metrics.get('auc', 0)
        }
        
        if 'models_tested' not in config:
            config['models_tested'] = []
        
        config['models_tested'].append(model_info)
        
        # 更新最佳模型
        if not config.get('best_model') or metrics['accuracy'] > config['best_model'].get('accuracy', 0):
            config['best_model'] = model_info
        
        # 如果所有模型都测试完成，更新状态
        if len(config['models_tested']) >= len(config.get('planned_models', [])):
            config['status'] = 'completed'
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _update_experiments_index(self, experiment_id, model_type, model_version, metrics, notes):
        """更新全局实验索引"""
        if self.index_file.exists():
            index_df = pd.read_csv(self.index_file)
        else:
            index_df = pd.DataFrame()
        
        new_entry = {
            'experiment_id': experiment_id,
            'version': model_version,
            'model_type': model_type,
            'accuracy': metrics['accuracy'],
            'auc': metrics.get('auc', 0),
            'best_model': False,  # 稍后更新
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'notes': notes[:100]
        }
        
        index_df = pd.concat([index_df, pd.DataFrame([new_entry])], ignore_index=True)
        index_df.to_csv(self.index_file, index=False)
    
    def _plot_model_comparison(self, comparison_df, exp_dir):
        """绘制模型比较图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 准确率比较
        axes[0, 0].barh(comparison_df['Model'], comparison_df['Accuracy'] * 100)
        axes[0, 0].set_xlabel('Accuracy (%)')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].axvline(x=50, color='r', linestyle='--', alpha=0.5)
        
        # AUC比较
        if 'AUC' in comparison_df.columns and comparison_df['AUC'].notna().any():
            axes[0, 1].barh(comparison_df['Model'], comparison_df['AUC'])
            axes[0, 1].set_xlabel('AUC')
            axes[0, 1].set_title('Model AUC Comparison')
            axes[0, 1].axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
        
        # F1-score比较
        axes[1, 0].barh(comparison_df['Model'], comparison_df['F1'])
        axes[1, 0].set_xlabel('F1 Score')
        axes[1, 0].set_title('Model F1-Score Comparison')
        
        # 精度-召回率散点图
        axes[1, 1].scatter(comparison_df['Precision'], comparison_df['Recall'], s=100)
        for idx, row in comparison_df.iterrows():
            axes[1, 1].annotate(row['Model'].split()[-1], 
                              (row['Precision'], row['Recall']),
                              fontsize=9, alpha=0.7)
        axes[1, 1].set_xlabel('Precision')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title('Precision-Recall Trade-off')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(exp_dir / "results" / "model_comparison_chart.png", dpi=150, bbox_inches='tight')
        plt.close()