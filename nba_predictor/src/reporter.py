# src/report_generator.py
"""
报告生成模块 - 为NBA模型实验生成详细报告
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path

def generate_experiment_report(experiment_results):
    """
    生成完整的实验报告
    
    参数:
        experiment_results: trainer.train_all_models() 返回的结果字典
    返回:
        报告文件路径
    """
    # 1. 获取实验ID
    exp_id = experiment_results['experiment_id']
    
    # 2. 获取实验配置
    from src.versioner import MultiModelVersioner
    versioner = MultiModelVersioner("nba_experiments")
    exp_dir = versioner.base_dir / exp_id
    
    # 3. 读取实验配置
    config_file = exp_dir / "experiment_config.json"
    with open(config_file, 'r') as f:
        experiment_config = json.load(f)
    
    # 4. 读取比较结果
    comparison_file = exp_dir / "results" / "model_comparison.csv"
    if comparison_file.exists():
        comparison_df = pd.read_csv(comparison_file)
    else:
        comparison_df = None
    
    # 5. 构建报告内容
    report_lines = []
    
    # 报告标题
    report_lines.append(f"# NBA胜负预测模型实验报告")
    report_lines.append("")
    report_lines.append(f"**实验ID**: {exp_id}")
    report_lines.append(f"**实验名称**: {experiment_config.get('experiment_name', '未命名实验')}")
    report_lines.append(f"**实验时间**: {experiment_config.get('created_at', '未知时间')}")
    report_lines.append(f"**实验状态**: {experiment_config.get('status', 'unknown')}")
    report_lines.append(f"**测试模型数量**: {len(experiment_results['results'])}")
    report_lines.append(f"**特征数量**: {experiment_config['data_info']['feature_count']}")
    report_lines.append("")
    
    # 数据概览
    report_lines.append("## 数据概览")
    report_lines.append("")
    report_lines.append("| 指标 | 训练集 | 测试集 |")
    report_lines.append("|------|--------|--------|")
    report_lines.append(f"| 样本数量 | {experiment_config['data_info']['train_samples']:,} | {experiment_config['data_info']['test_samples']:,} |")
    report_lines.append(f"| 主胜比例 | {experiment_config['data_info']['train_pos_ratio']:.2%} | {experiment_config['data_info']['test_pos_ratio']:.2%} |")
    report_lines.append("")
    
    # 最佳模型结果
    report_lines.append("## 最佳模型结果")
    report_lines.append("")
    
    if experiment_results['best_model']:
        metadata = experiment_results['best_model']['metadata']
        metrics = metadata['metrics']
        
        report_lines.append(f"**最佳模型**: {metadata['model_type']} v{metadata['model_version']}")
        report_lines.append(f"**创建时间**: {metadata['created_at']}")
        report_lines.append("")
        report_lines.append("**性能指标**")
        report_lines.append("| 指标 | 值 |")
        report_lines.append("|------|----|")
        report_lines.append(f"| 准确率 | {metrics['accuracy']:.2%} |")
        if 'auc' in metrics:
            report_lines.append(f"| AUC | {metrics['auc']:.3f} |")
        report_lines.append(f"| 精确率 | {metrics['precision']:.3f} |")
        report_lines.append(f"| 召回率 | {metrics['recall']:.3f} |")
        report_lines.append(f"| F1分数 | {metrics['f1_score']:.3f} |")
        report_lines.append("")
        
        # 模型配置
        report_lines.append("**模型配置**")
        report_lines.append("```python")
        report_lines.append(json.dumps(metadata['model_config'], indent=2))
        report_lines.append("```")
        report_lines.append("")
        
        # 数据信息
        report_lines.append("**数据信息**")
        report_lines.append(f"- 训练样本数: {metadata['data_info']['train_samples']:,}")
        report_lines.append(f"- 测试样本数: {metadata['data_info']['test_samples']:,}")
        report_lines.append(f"- 特征数量: {metadata['data_info']['feature_count']}")
        report_lines.append("")
    
    # 所有模型比较
    if comparison_df is not None:
        report_lines.append("## 所有模型性能比较")
        report_lines.append("")
        report_lines.append("| 模型 | 准确率 | AUC | F1分数 | 训练样本 | 特征数 |")
        report_lines.append("|------|--------|-----|--------|----------|--------|")
        
        for _, row in comparison_df.iterrows():
            auc_value = row.get('AUC', 'N/A')
            auc_str = f"{auc_value:.3f}" if isinstance(auc_value, (int, float)) else auc_value
            report_lines.append(f"| {row['Model']} | {row['Accuracy']:.2%} | {auc_str} | {row['F1']:.3f} | {row['Train Samples']:,} | {row['Features']} |")
        
        report_lines.append("")
    
    # 实验结果文件说明
    report_lines.append("## 实验结果文件")
    report_lines.append("")
    report_lines.append(f"实验的所有结果文件保存在以下目录：")
    report_lines.append(f"`model_experiments/{exp_id}/`")
    report_lines.append("")
    
    # 使用说明
    report_lines.append("## 如何使用最佳模型")
    report_lines.append("")
    report_lines.append("```python")
    report_lines.append("# 方法1：通过版本管理器加载")
    report_lines.append("from multi_model_versioner import MultiModelVersioner")
    report_lines.append("")
    report_lines.append(f"versioner = MultiModelVersioner('nba_experiments')")
    report_lines.append(f"best_model_data = versioner.get_best_model('{exp_id}')")
    report_lines.append("")
    report_lines.append("model = best_model_data['model']")
    report_lines.append("metadata = best_model_data['metadata']")
    report_lines.append("")
    report_lines.append("print(f\"加载模型: {metadata['model_type']} v{metadata['model_version']}\")")
    report_lines.append("print(f\"模型准确率: {metadata['metrics']['accuracy']:.2%}\")")
    report_lines.append("")
    report_lines.append("# 进行预测")
    report_lines.append("predictions = model.predict(new_data)")
    report_lines.append("probabilities = model.predict_proba(new_data)")
    report_lines.append("```")
    report_lines.append("")
    
    # 实验总结
    report_lines.append("## 实验总结")
    report_lines.append("")
    
    if experiment_results['best_model']:
        best_acc = experiment_results['best_model']['metadata']['metrics']['accuracy']
        test_pos_ratio = experiment_config['data_info']['test_pos_ratio']
        improvement = best_acc - test_pos_ratio
        
        report_lines.append(f"1. **最佳模型**: {experiment_results['best_model']['metadata']['model_type']}")
        report_lines.append(f"2. **最佳准确率**: {best_acc:.2%}")
        report_lines.append(f"3. **相对于基准提升**: {improvement:.2%} 百分点")
        report_lines.append(f"4. **相对提升率**: {improvement/test_pos_ratio*100:.1f}%")
    report_lines.append("")
    
    # 页脚
    report_lines.append("---")
    report_lines.append(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # 6. 保存报告
    report_content = "\n".join(report_lines)
    report_path = exp_dir / "experiment_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ 实验报告已生成: {report_path}")
    
    # 7. 在控制台输出关键信息
    print("\n" + "="*60)
    print("实验报告摘要")
    print("="*60)
    print(f"实验ID: {exp_id}")
    print(f"实验名称: {experiment_config.get('experiment_name', '未命名实验')}")
    
    if experiment_results['best_model']:
        best_model = experiment_results['best_model']['metadata']
        print(f"最佳模型: {best_model['model_type']} v{best_model['model_version']}")
        print(f"准确率: {best_model['metrics']['accuracy']:.2%}")
    
    print(f"报告路径: {report_path}")
    print("="*60)
    
    return report_path


def generate_model_detail_report(experiment_id, model_type, model_version):
    """
    生成单个模型的详细报告
    
    参数:
        experiment_id: 实验ID
        model_type: 模型类型
        model_version: 模型版本号
    返回:
        报告文件路径
    """
    # 1. 初始化版本管理器
    from src.versioner import MultiModelVersioner
    versioner = MultiModelVersioner("nba_experiments")
    
    # 2. 构建模型路径
    model_dir = versioner.base_dir / experiment_id / "models" / model_type / f"v{model_version}"
    
    if not model_dir.exists():
        print(f"❌ 模型目录不存在: {model_dir}")
        return None
    
    # 3. 读取模型元数据
    metadata_file = model_dir / "metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # 4. 读取特征重要性
    importance_file = model_dir / "feature_importance.csv"
    if importance_file.exists():
        importance_df = pd.read_csv(importance_file)
    else:
        importance_df = None
    
    # 5. 构建报告内容
    report_lines = []
    
    # 报告标题
    report_lines.append(f"# 模型详细报告 - {model_type} v{model_version}")
    report_lines.append("")
    report_lines.append(f"**所属实验**: {experiment_id}")
    report_lines.append(f"**创建时间**: {metadata['created_at']}")
    report_lines.append(f"**模型说明**: {metadata.get('notes', '无说明')}")
    report_lines.append("")
    
    # 性能指标
    report_lines.append("## 性能指标")
    report_lines.append("")
    report_lines.append("| 指标 | 值 |")
    report_lines.append("|------|----|")
    report_lines.append(f"| 准确率 | {metadata['metrics']['accuracy']:.2%} |")
    
    if 'auc' in metadata['metrics']:
        report_lines.append(f"| AUC | {metadata['metrics']['auc']:.3f} |")
    
    report_lines.append(f"| 精确率 | {metadata['metrics']['precision']:.3f} |")
    report_lines.append(f"| 召回率 | {metadata['metrics']['recall']:.3f} |")
    report_lines.append(f"| F1分数 | {metadata['metrics']['f1_score']:.3f} |")
    report_lines.append("")
    
    # 特征重要性
    if importance_df is not None and len(importance_df) > 0:
        report_lines.append("## 特征重要性（Top 10）")
        report_lines.append("")
        report_lines.append("| 排名 | 特征 | 重要性 |")
        report_lines.append("|------|------|--------|")
        
        for idx, row in importance_df.head(10).iterrows():
            report_lines.append(f"| {idx+1} | {row['feature']} | {row['importance']:.4f} |")
        
        report_lines.append("")
    
    # 使用示例
    report_lines.append("## 使用示例")
    report_lines.append("")
    report_lines.append("```python")
    report_lines.append("import pickle")
    report_lines.append("import pandas as pd")
    report_lines.append("")
    report_lines.append(f"# 加载模型")
    report_lines.append(f"model_path = 'model.pkl'  # 替换为实际路径")
    report_lines.append("with open(model_path, 'rb') as f:")
    report_lines.append("    model = pickle.load(f)")
    report_lines.append("")
    report_lines.append("# 进行预测")
    report_lines.append("predictions = model.predict(new_data)")
    report_lines.append("")
    report_lines.append("# 获取概率")
    report_lines.append("if hasattr(model, 'predict_proba'):")
    report_lines.append("    probabilities = model.predict_proba(new_data)")
    report_lines.append("    win_probabilities = probabilities[:, 1]")
    report_lines.append("```")
    report_lines.append("")
    
    # 页脚
    report_lines.append("---")
    report_lines.append(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # 6. 保存报告
    report_content = "\n".join(report_lines)
    report_path = model_dir / f"{model_type}_v{model_version}_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ 模型详细报告已生成: {report_path}")
    return report_path


# 主函数示例
if __name__ == "__main__":
    print("报告生成器模块已加载")
    print("使用方法:")
    print("1. generate_experiment_report(experiment_results)")
    print("2. generate_model_detail_report(experiment_id, model_type, model_version)")