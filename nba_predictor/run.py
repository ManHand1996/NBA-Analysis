"""
NBA胜负预测系统 - 主程序（简化版）
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
# 导入配置
sys.path.insert(0, str(Path(__file__).parent))
from config import setup_directories

# 导入工具
from src.utils.logger import logger
from src.utils.paths import PathManager
from src.nba_model_pipeline import NBAModelPipeline

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='NBA胜负预测系统')
    parser.add_argument('--mode', choices=['train', 'predict', 'evaluate', 'demo'], 
                       default='demo', help='运行模式')
    parser.add_argument('--experiment', help='实验ID')
    parser.add_argument('--data', help='数据文件路径')
    parser.add_argument('--env', default='development', help='运行环境')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 设置环境
    # if args.env != ENV:
    #     os.environ['NBA_ENV'] = args.env
    
    # 初始化
    setup_directories()
    path_manager = PathManager()
    path_manager.ensure_directories()
    
    # 设置日志
    # logger = setup_logger(
    #     name='nba_predictor',
    #     log_file=LOGS_DIR / 'app.log',
    #     level='DEBUG' if args.verbose else 'INFO'
    # )
    
    logger.info(f"启动NBA预测系统 - 模式: {args.mode}")
    pipeline = NBAModelPipeline()

    # 1.准备特征
    
    feature_df = pd.read_csv(path_manager.get_data_path('games_features.csv'))
    feature_df = feature_df.drop(['home_last_game_date', 'away_last_game_date'], axis=1)
    # 2.

    try:
        # 根据模式执行
        
        if args.mode == 'train':
            exper_result = pipeline.run_experiment("Exp0001", feature_df)
            
            
        # elif args.mode == 'predict':
        #     from src.models.predictor import make_prediction
        #     result = make_prediction(args.data, args.experiment)
        #     print(f"预测结果: {result}")
            
        # elif args.mode == 'evaluate':
        #     from src.evaluation.evaluator import evaluate_model
        #     evaluate_model(args.experiment, logger)
            
        # else:  # demo模式
        #     run_demo_mode(logger)
            
    except Exception as e:
        raise e
        logger.error(f"运行失败: {e}")
        sys.exit(1)
    
    logger.info("程序运行完成")

# def run_demo_mode(logger):
#     """演示模式"""
#     logger.info("运行演示模式...")
    
#     # 导入演示模块
#     from src.training.demo import run_demo_experiment
#     from src.evaluation.reporter import generate_report
    
#     # 运行演示实验
#     experiment_id = run_demo_experiment(logger)
    
#     if experiment_id:
#         logger.info(f"实验完成: {experiment_id}")
        
#         # 生成报告
#         report_path = generate_report(experiment_id)
#         logger.info(f"报告生成: {report_path}")
        
#         # 显示预测示例
#         from src.models.demo_predictor import show_prediction_example
#         show_prediction_example(experiment_id, logger)
#     else:
#         logger.error("演示实验失败")

if __name__ == "__main__":
    main()