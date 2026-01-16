# src/nba_model_pipeline.py
"""
NBA模型管道 - 协调训练器和版本管理器
"""
import pandas as pd
import pickle
import json
import os
from datetime import datetime
from src.trainer import MultiModelTrainer
from src.versioner import MultiModelVersioner
from src.reporter import generate_experiment_report


class NBAModelPipeline:
    """
    NBA模型管道 - 协调器
    
    职责：
    1. 协调MultiModelTrainer和MultiModelVersioner
    2. 管理完整的训练和版本管理流程
    3. 提供统一的预测接口
    """
    
    def __init__(self, use_time_split=True, use_scaler=True):
        """
        初始化管道
        
        Args:
            use_time_split: 是否使用时间感知划分
            use_scaler: 是否使用标准化
        """
        self.trainer = MultiModelTrainer(use_time_split, use_scaler)
        self.versioner = MultiModelVersioner("nba_experiments")
        self.current_experiment = None
        self.current_data = None  # 保存当前数据
        
        print(f"🚀 NBA模型管道已初始化")
        print(f"   时间划分: {use_time_split}")
        print(f"   标准化: {use_scaler}")
    

    def make_sklearn_compatible(self, model):
        """使模型兼容 scikit-learn"""
        if not hasattr(model, '_estimator_type'):
            model._estimator_type = "classifier"
        return model


    def run_experiment(self, experiment_name, df, test_seasons=3):
        """
        运行完整实验
        
        Args:
            experiment_name: 实验名称
            df: 包含所有数据的DataFrame
            test_seasons: 测试赛季数量
            
        Returns:
            实验结果字典
        """
        print(f"\n{'='*60}")
        print(f"开始实验: {experiment_name}")
        print(f"{'='*60}")
        
        # 1. 使用trainer准备特征
        print("\n1️⃣ 准备特征数据...")
        X_train, X_test, y_train, y_test, feature_names, scaler = \
            self.trainer.prepare_features(df, test_seasons)
        
        # 保存当前数据供后续使用
        self.current_data = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'feature_names': feature_names,
            'scaler': scaler
        }
        
        # 2. 创建实验（通过versioner）
        print("\n2️⃣ 创建实验记录...")
        data_info = {
            'experiment_name': experiment_name,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': len(feature_names),
            'train_pos_ratio': y_train.mean(),
            'test_pos_ratio': y_test.mean(),
            'use_time_split': self.trainer.use_time_split,
            'use_scaler': self.trainer.use_scaler,
            'test_seasons': test_seasons
        }
        
        feature_info = {
            'feature_names': feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names),
            'scaler_used': scaler is not None
        }
        
        self.current_experiment, exp_dir = self.versioner.create_experiment(
            experiment_name, data_info, feature_info
        )
        
        print(f"   实验ID: {self.current_experiment}")
        
        # 3. 训练并保存所有模型
        print("\n3️⃣ 训练所有模型...")
        models = ['xgboost', 'random_forest', 'gradient_boosting']
        results = {}
        
        for model_type in models:
            print(f"\n   🔄 训练{model_type}...")
            
            # 训练模型
            if model_type == 'xgboost':
                model = self.trainer.train_xgboost(X_train, y_train)
            elif model_type == 'random_forest':
                model = self.trainer.train_random_forest(X_train, y_train)
            else:  # gradient_boosting
                 # 需要填充NaN
                X_train = X_train.fillna(0)
                X_test = X_test.fillna(0)
                model = self.trainer.train_gradient_boosting(X_train, y_train)
            
            model = self.make_sklearn_compatible(model)

            # 评估模型
            metrics = self.trainer.evaluate_model(model, X_test, y_test)
            print(f"      准确率: {metrics['accuracy']:.2%}")
            
            # 保存到版本管理器
            model_config = self._get_default_model_config(model_type)
            
            result = self.versioner.save_model_result(
                experiment_id=self.current_experiment,
                model_type=model_type,
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model_config=model_config,
                scaler=scaler,
                notes=f"{model_type} with time_split={self.trainer.use_time_split}"
            )
            
            results[model_type] = result
        
        # 4. 比较模型
        print("\n4️⃣ 比较模型性能...")
        comparison = self.versioner.compare_models_in_experiment(self.current_experiment)
        
        # 5. 获取最佳模型
        print("\n5️⃣ 确定最佳模型...")
        best_model_info = self.versioner.get_best_model(self.current_experiment)
        
        if best_model_info:
            best_metrics = best_model_info['metadata']['metrics']
            print(f"   🏆 最佳模型: {best_model_info['metadata']['model_type']}")
            print(f"       准确率: {best_metrics['accuracy']:.2%}")
            if 'auc' in best_metrics:
                print(f"       AUC: {best_metrics['auc']:.3f}")
        
        # 6. 保存完整管道
        print("\n6️⃣ 保存完整预测管道...")
        pipeline_path = self._save_complete_pipeline(best_model_info, scaler, feature_names)
        
        # 7. 生成实验报告
        print("\n7️⃣ 生成实验报告...")
        self._generate_experiment_report(results, comparison, best_model_info)
        
        print(f"\n{'='*60}")
        print(f"实验完成!")
        print(f"实验ID: {self.current_experiment}")
        print(f"报告路径: {exp_dir}/experiment_report.md")
        print(f"管道文件: {pipeline_path}")
        print(f"{'='*60}")
        
        return {
            'experiment_id': self.current_experiment,
            'results': results,
            'comparison': comparison,
            'best_model': best_model_info,
            'pipeline_path': pipeline_path
        }
    
    def _get_default_model_config(self, model_type):
        """获取默认模型配置"""
        configs = {
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': 42,
                'n_jobs': -1
            },
            'gradient_boosting': {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'random_state': 42
            }
        }
        
        return configs.get(model_type, {})
    
    def _save_complete_pipeline(self, best_model_info, scaler, feature_names):
        """保存完整预测管道"""
        if not best_model_info:
            print("警告: 没有最佳模型信息，跳过管道保存")
            return None
        
        pipeline_package = {
            'model': best_model_info['model'],
            'scaler': scaler,
            'feature_names': feature_names,
            'experiment_id': self.current_experiment,
            'trainer_config': {
                'use_time_split': self.trainer.use_time_split,
                'use_scaler': self.trainer.use_scaler
            },
            'metadata': best_model_info['metadata'],
            'created_at': datetime.now().isoformat()
        }
        
        # 确保目录存在
        
        os.makedirs("pipelines", exist_ok=True)
        
        pipeline_path = f"pipelines/{self.current_experiment}_pipeline.pkl"
        
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline_package, f)
        
        print(f"   ✅ 管道已保存: {pipeline_path}")
        
        return pipeline_path
    
    def _generate_experiment_report(self, results, comparison, best_model_info):
        """生成实验报告"""
        try:
            
            
            # 准备报告数据
            report_data = {
                'experiment_id': self.current_experiment,
                'results': results,
                'best_model': best_model_info
            }
            
            # 调用报告生成器
            report_path = generate_experiment_report(report_data)
            return report_path
            
        except ImportError:
            print("警告: 报告生成器未找到，跳过报告生成")
            return None
    
    def load_pipeline(self, experiment_id=None, pipeline_path=None):
        """
        加载预测管道
        
        Args:
            experiment_id: 实验ID，如果提供则加载该实验的最佳模型
            pipeline_path: 直接指定管道文件路径
            
        Returns:
            加载的管道对象
        """
        if pipeline_path:
            # 直接加载指定路径
            try:
                with open(pipeline_path, 'rb') as f:
                    pipeline = pickle.load(f)
                
                print(f"✅ 管道已加载: {pipeline_path}")
                return pipeline
                
            except Exception as e:
                print(f"❌ 加载管道失败: {e}")
                return None
        
        elif experiment_id:
            # 加载指定实验的最佳模型
            best_model_info = self.versioner.get_best_model(experiment_id)
            
            if not best_model_info:
                print(f"❌ 实验 {experiment_id} 没有找到最佳模型")
                return None
            
            # 构建管道包
            model_dir = best_model_info['model_dir']
            
            # 尝试加载scaler
            scaler_path = model_dir / "scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            else:
                scaler = None
            
            # 加载特征名称（从实验配置）
            exp_dir = self.versioner.base_dir / experiment_id
            feature_file = exp_dir / "features" / "feature_list.json"
            
            if feature_file.exists():
                with open(feature_file, 'r') as f:
                    feature_names = json.load(f)
            else:
                feature_names = None
            
            pipeline = {
                'model': best_model_info['model'],
                'scaler': scaler,
                'feature_names': feature_names,
                'experiment_id': experiment_id,
                'metadata': best_model_info['metadata']
            }
            
            print(f"✅ 实验 {experiment_id} 的管道已构建")
            return pipeline
        
        else:
            print("❌ 必须提供experiment_id或pipeline_path")
            return None
    
    def predict(self, new_features, experiment_id=None, pipeline_path=None):
        """
        使用训练好的管道进行预测
        
        Args:
            new_features: 新数据的特征字典
            experiment_id: 实验ID，加载该实验的最佳模型
            pipeline_path: 直接指定管道文件路径
            
        Returns:
            预测结果
        """
        # 1. 加载管道
        pipeline = self.load_pipeline(experiment_id, pipeline_path)
        
        if not pipeline:
            return None
        
        # 2. 准备特征数据
        feature_names = pipeline['feature_names']
        
        if feature_names is None:
            print("❌ 管道中没有特征名称信息")
            return None
        
        # 创建特征DataFrame，确保特征顺序正确
        feature_df = pd.DataFrame([new_features])
        
        # 检查特征是否完整
        missing_features = set(feature_names) - set(feature_df.columns)
        if missing_features:
            print(f"警告: 缺少特征: {missing_features}")
            # 填充缺失特征为0
            for feature in missing_features:
                feature_df[feature] = 0
        
        # 确保特征顺序
        feature_df = feature_df[feature_names]
        
        # 3. 标准化处理
        scaler = pipeline['scaler']
        if scaler is not None:
            try:
                feature_df_scaled = scaler.transform(feature_df)
            except Exception as e:
                print(f"❌ 标准化失败: {e}")
                feature_df_scaled = feature_df.values
        else:
            feature_df_scaled = feature_df.values
        
        # 4. 预测
        model = pipeline['model']
        
        try:
            prediction = model.predict(feature_df_scaled)[0]
            
            result = {
                'prediction': int(prediction),
                'prediction_label': '主胜' if prediction == 1 else '客胜'
            }
            
            # 获取概率（如果可用）
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(feature_df_scaled)[0]
                result['probability'] = probability.tolist()
                result['win_probability'] = float(probability[1])
                result['confidence'] = self._get_confidence_level(probability[1])
            
            # 添加模型信息
            metadata = pipeline.get('metadata', {})
            result['model_info'] = {
                'experiment_id': pipeline.get('experiment_id'),
                'model_type': metadata.get('model_type', type(model).__name__),
                'accuracy': metadata.get('metrics', {}).get('accuracy', 0)
            }
            
            print(f"✅ 预测完成: {result['prediction_label']}")
            if 'win_probability' in result:
                print(f"   获胜概率: {result['win_probability']:.2%}")
            
            return result
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return None
    
    def _get_confidence_level(self, probability):
        """根据概率确定置信度"""
        if probability > 0.7 or probability < 0.3:
            return '高'
        elif probability > 0.6 or probability < 0.4:
            return '中'
        else:
            return '低'
    
    def get_experiment_summary(self, experiment_id=None):
        """获取实验摘要"""
        if experiment_id is None:
            experiment_id = self.current_experiment
        
        if experiment_id is None:
            print("❌ 没有指定实验ID")
            return None
        
        return self.versioner.compare_models_in_experiment(experiment_id)