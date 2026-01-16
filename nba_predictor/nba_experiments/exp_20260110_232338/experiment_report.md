# NBA胜负预测模型实验报告

**实验ID**: exp_20260110_232338
**实验名称**: Exp0001
**实验时间**: 2026-01-10T23:23:38.355918
**实验状态**: completed
**测试模型数量**: 3
**特征数量**: 88

## 数据概览

| 指标 | 训练集 | 测试集 |
|------|--------|--------|
| 样本数量 | 47,182 | 3,960 |
| 主胜比例 | 60.89% | 55.78% |

## 最佳模型结果

**最佳模型**: gradient_boosting v1
**创建时间**: 2026-01-10T23:24:37.525001

**性能指标**
| 指标 | 值 |
|------|----|
| 准确率 | 64.60% |
| AUC | 0.700 |
| 精确率 | 0.644 |
| 召回率 | 0.816 |
| F1分数 | 0.720 |

**模型配置**
```python
{
  "n_estimators": 200,
  "max_depth": 5,
  "learning_rate": 0.05,
  "subsample": 0.8,
  "random_state": 42
}
```

**数据信息**
- 训练样本数: 47,182
- 测试样本数: 3,960
- 特征数量: 88

## 所有模型性能比较

| 模型 | 准确率 | AUC | F1分数 | 训练样本 | 特征数 |
|------|--------|-----|--------|----------|--------|
| gradient_boosting v1 | 64.60% | 0.700 | 0.720 | 47,182 | 88 |
| xgboost v1 | 64.39% | 0.700 | 0.720 | 47,182 | 88 |
| random_forest v1 | 63.79% | 0.694 | 0.676 | 47,182 | 88 |

## 实验结果文件

实验的所有结果文件保存在以下目录：
`model_experiments/exp_20260110_232338/`

## 如何使用最佳模型

```python
# 方法1：通过版本管理器加载
from multi_model_versioner import MultiModelVersioner

versioner = MultiModelVersioner('nba_experiments')
best_model_data = versioner.get_best_model('exp_20260110_232338')

model = best_model_data['model']
metadata = best_model_data['metadata']

print(f"加载模型: {metadata['model_type']} v{metadata['model_version']}")
print(f"模型准确率: {metadata['metrics']['accuracy']:.2%}")

# 进行预测
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)
```

## 实验总结

1. **最佳模型**: gradient_boosting
2. **最佳准确率**: 64.60%
3. **相对于基准提升**: 8.81% 百分点
4. **相对提升率**: 15.8%

---
*报告生成时间: 2026-01-10 23:24:38*