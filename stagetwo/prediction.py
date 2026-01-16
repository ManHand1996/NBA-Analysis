"""
赛后的统计数据都是应用于描述性分析和诊断性分析，基于已发生的结果对事实的统计，
绝不能使用这些数据直接进行预测 eg. ORtg与DRtg直接预测Wins

step 1:
features_v1 = [
    'win_pct_last10',      # 简单胜率
    'win_pct_last5',       # 近期胜率  
    'home_win_pct_last20', # 主场表现
    'current_streak',      # 连胜连败
    'avg_margin_last10',   # 平均净胜分
    'rest_days'            # 休息天数
]
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import pickle
RAW_DATA_PATH = os.path.join(os.getcwd(), f'raw_data')
PROCESS_DATA_PATH = os.path.join(os.getcwd(), f'process_data')


class NBADataProcessor:
    def __init__(self, games_df, teams_df=None):
        """
        初始化处理器
        games_df(联盟赛程): 包含列 ['game_date', 'season', 'home', 'away', 
                          'home_pts', 'away_pts', 'home_id', 'away_id']
        teams_df: 球队信息（可选）
        """
        
        self.games_df = games_df.copy()
        self.games_df['game_date'] = pd.to_datetime(self.games_df['game_date'])
        self.games_df = self.games_df.sort_values(['season', 'game_date'])
        

    def calculate_team_features(self, team_id, current_date, season):
        """
        为特定球队在特定日期计算所有特征
        """
        # 获取该球队在当前日期前的所有比赛
        # 相当于SQL join： tb_game tg join tb_team_info ti on tg.home = ti.team or tg.visitor =ti.team

        team_games: pd.DataFrame = self.games_df[
                    ((self.games_df['home_id'] == team_id) | (self.games_df['away_id'] == team_id))
                        & (self.games_df['season'] == season)
                        & (self.games_df['game_date'] < current_date)
                        ].copy()

        if len(team_games) == 0:
            return self._get_empty_features(team_id, current_date, season)
        
        # 计算每场比赛对目标球队的结果
        team_games['is_home'] = team_games['home_id'] == team_id
        team_games['team_pts'] =  np.where(
            team_games['is_home'], 
            team_games['home_pts'], 
            team_games['away_pts']
        )
        team_games['opp_pts'] =  np.where(
            team_games['is_home'], 
            team_games['away_pts'],
            team_games['home_pts']
           
        )

        team_games['win'] = (team_games['team_pts'] > team_games['opp_pts']).astype(int)
        team_games['pts_diff'] = team_games['team_pts'] - team_games['opp_pts']
        # team_games['game_num'] = range(1, len(team_games) + 1)
        
        # 获取最后一场比赛日期
        last_game_date = team_games['game_date'].max()
        rest_days = (current_date - last_game_date).days
        
        # 基础特征
        features = {
            'team_id': team_id,
            'current_date': current_date,
            'season': season,
            'games_played': len(team_games),
            'total_wins': team_games['win'].sum(),
            'total_win_pct': team_games['win'].mean() if len(team_games) > 0 else 0.5,
            # 'last_game_date': last_game_date,
            'rest_days': rest_days,
            'current_streak': self._calculate_current_streak(team_games['win']),
            'avg_pts_diff_all': team_games['pts_diff'].mean() if len(team_games) > 0 else 0.0
        }
        
        # 动态窗口特征（last5, last10, last20）
        for window in [5, 10, 20]:
            window_key = f'last{window}'
            if len(team_games) >= window:
                window_games = team_games.tail(window)
                features.update({
                    f'win_pct_{window_key}': window_games['win'].mean(),
                    f'avg_pts_diff_{window_key}': window_games['pts_diff'].mean(),
                    f'home_wins_{window_key}': window_games[window_games['is_home']]['win'].sum(),
                    f'away_wins_{window_key}': window_games[~window_games['is_home']]['win'].sum(),
                    f'{window_key}_exists': 1
                })
            else:
                # 中性填充 + 存在标记
                features.update({
                    f'win_pct_{window_key}': 0.5,
                    f'avg_pts_diff_{window_key}': 0.0,
                    f'home_wins_{window_key}': 0,
                    f'away_wins_{window_key}': 0,
                    f'{window_key}_exists': 0
                })
        
        # 添加存在标记的交互特征
        features['games_played_x_last10_exists'] = features['games_played'] * features.get('last10_exists', 0)
        features['games_played_x_last20_exists'] = features['games_played'] * features.get('last20_exists', 0)
        
        return features
    

    def calculate_team_features_with_carryover(self, team_id, current_date, season, carryover_data=None):
        """
        计算球队特征，支持赛季结转数据
        """
        # 先计算常规特征
        base_features = self.calculate_team_features(team_id, current_date, season)
        
        # 如果是赛季早期且没有足够数据，使用结转数据
        current_games_played = base_features['games_played']
        
        if current_games_played < 10 and carryover_data:
            # 赛季早期，使用结转数据增强特征
            for window in [5, 10, 20]:
                window_key = f'last{window}'
                exists_key = f'{window_key}_exists'
                
                if base_features[exists_key] == 0:  # 当前赛季没有该窗口数据
                    # 使用结转数据填充
                    if carryover_data['carryover_games'] >= window:
                        # 结转数据足够
                        base_features[f'win_pct_{window_key}'] = carryover_data['carryover_win_pct']
                        base_features[f'avg_pts_diff_{window_key}'] = carryover_data['carryover_avg_pts_diff']
                        base_features[f'{window_key}_is_carryover'] = 1  # 标记为结转数据
                    elif carryover_data['carryover_games'] > 0:
                        # 结转数据部分可用
                        base_features[f'win_pct_{window_key}'] = carryover_data['carryover_win_pct']
                        base_features[f'avg_pts_diff_{window_key}'] = carryover_data['carryover_avg_pts_diff']
                        base_features[f'{window_key}_is_carryover'] = 1
                    else:
                        # 结转数据标记
                        base_features[f'{window_key}_is_carryover'] = 0
        
        # 添加上赛季总结特征
        if carryover_data:
            base_features['prev_season_win_pct'] = carryover_data.get('prev_season_final_win_pct', 0.5)
            base_features['prev_season_games'] = carryover_data.get('prev_season_total_games', 0)
        else:
            base_features['prev_season_win_pct'] = 0.5  # 默认值
            base_features['prev_season_games'] = 0
            base_features['is_expansion_team'] = 1  # 可能是扩张球队或第一个赛季
        
        return base_features

    def _calculate_current_streak(self, win_series):
        """计算当前连胜/连负"""
        streak = 0
        current_result = win_series.iloc[-1]
        for i in range(len(win_series) - 1, -1, -1):
            if win_series.iloc[i] == current_result:
                streak += (1 if current_result == 1 else -1)
            else:
                break
        return streak
    
    def _get_empty_features(self, team_id, current_date, season):
        """为新球队或赛季初期返回默认特征"""
        return {
            'team_id': team_id,
            'current_date': current_date,
            'season': season,
            'games_played': 0,
            'total_wins': 0,
            'total_win_pct': 0.5,
            # 'last_game_date': None,
            'rest_days': 100,  # 赛季开始前
            'current_streak': 0,
            'avg_pts_diff_all': 0.0,
            **{f'{k}_{w}': (0.5 if 'win_pct' in k else 0.0 if 'avg_pts_diff' in k else 0) 
               for w in [5, 10, 20] for k in ['win_pct', 'avg_pts_diff', 'home_wins', 'away_wins']},
            **{f'last{w}_exists': 0 for w in [5, 10, 20]},
            'games_played_x_last10_exists': 0,
            'games_played_x_last20_exists': 0
        }


class NBAWinPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None

        
    def prepare_features(self, df, test_seasons=3):
        """
        准备特征和标签，按时间划分训练测试集
        test_seasons: 用最后几个赛季作为测试集
        """
        identifier_columns = [
            'home_team_id', 'away_team_id',
            'home_win', 'game_date', 'season', 'pts_diff'
            ]
        # 分离特征和标签
        X = df.drop(identifier_columns, axis=1)
        y = df['home_win']
        
        # 按时间划分：用较早的赛季训练，较晚的赛季测试
        seasons = sorted(df['season'].unique())
        test_season_cutoff = seasons[-test_seasons]
        
        train_mask = df['season'] < test_season_cutoff
        test_mask = df['season'] >= test_season_cutoff
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        print(f"训练集: {len(X_train)} 场比赛 ({df[train_mask]['season'].min()}-{df[train_mask]['season'].max()})")
        print(f"测试集: {len(X_test)} 场比赛 ({df[test_mask]['season'].min()}-{df[test_mask]['season'].max()})")
        
        # 数值特征标准化（排除存在标记）
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        exclude_cols = [col for col in numeric_cols if 'exists' in col or 'games_played' in col]
        scale_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(scale_cols) > 0:
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            
            self.scaler.fit(X_train[scale_cols])
            X_train_scaled[scale_cols] = self.scaler.transform(X_train[scale_cols])
            X_test_scaled[scale_cols] = self.scaler.transform(X_test[scale_cols])
        else:
            X_train_scaled, X_test_scaled = X_train, X_test
        
        self.feature_columns = X.columns.tolist()
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train, model_type='xgboost'):
        """训练模型"""
        
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
        elif model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        else:
            raise ValueError(f"未知模型类型: {model_type}")
        
        print(f"训练 {model_type} 模型...")
        # X_train = X_train.fillna(0.5)
        # 填充NaN
        model.fit(X_train, y_train)
        
        self.model = model
        return model
    
    def evaluate_model(self, X_test, y_test):
        """评估模型性能"""
        if self.model is None:
            raise ValueError("请先训练模型")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n{'='*50}")
        print(f"模型评估结果")
        print(f"{'='*50}")
        print(f"准确率: {accuracy:.2%}")
        print(f"AUC分数: {auc:.3f}")
        print(f"基准准确率（预测主队全胜）: {y_test.mean():.2%}")
        
        # 按赛季查看表现
        print(f"\n按赛季表现:")
        
        # 详细分类报告
        print(f"\n详细报告:")
        print(classification_report(y_test, y_pred, target_names=['客胜', '主胜']))
        
        return accuracy, auc
    
    def get_feature_importance(self, top_n=20):
        """获取特征重要性"""
        if self.model is None:
            raise ValueError("请先训练模型")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop {top_n} 重要特征:")
            print(importance_df.head(top_n).to_string(index=False))
            
            return importance_df
        
        return None
    
    def save_model(self, filepath='nba_predictor.pkl'):
        """保存模型"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }, f)
        print(f"模型已保存到 {filepath}")
    
    @staticmethod
    def load_model(filepath='nba_predictor.pkl'):
        """加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        predictor = NBAWinPredictor()
        predictor.model = data['model']
        predictor.scaler = data['scaler']
        predictor.feature_columns = data['feature_columns']
        
        return predictor


def check_NaN(X):
    # 1. 查看哪个特征工程步骤产生了NaN
    print("检查各个特征列的NaN情况:")
    for col in X.columns:
        nan_count = X[col].isna().sum()
        if nan_count > 0:
            print(f"{col}: {nan_count}个NaN")
            # 查看前几行
            print(f"  示例值: {X[col].head(10).tolist()}")

    # 2. 可能是last_game_date转换rest_days时的问题
    # 检查rest_days相关列
    rest_cols = [col for col in X.columns if 'rest' in col.lower()]
    for col in rest_cols:
        if col in X.columns:
            print(f"{col}: NaN={X[col].isna().sum()}, 范围={X[col].min()}-{X[col].max()}")

# 完整的训练流程
def train_complete_pipeline(games_df=None):
    """完整的训练流程"""
    
    print("步骤1: 构建特征数据集...")
    # feature_df = build_complete_dataset(games_df)

    feature_df = pd.read_csv(os.path.join(PROCESS_DATA_PATH, 'games_features.csv'))
    feature_df = feature_df.drop(['home_last_game_date', 'away_last_game_date'], axis=1)
    
    print("\n步骤2: 准备训练数据...")
    predictor = NBAWinPredictor()
    X_train, X_test, y_train, y_test = predictor.prepare_features(feature_df, test_seasons=3)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    # check_NaN(X_s)
    print("\n步骤3: 训练模型...")
    # 尝试不同模型，选择最好的
    models = ['xgboost', 'random_forest', 'gradient_boosting']
    best_score = 0
    best_model = None
    
    for model_type in models:
        print(f"\n尝试 {model_type}...")
        predictor.train_model(X_train, y_train, model_type)
        accuracy, auc = predictor.evaluate_model(X_test, y_test)
        
        if auc > best_score:
            best_score = auc
            best_model = model_type
            best_predictor = predictor
    
    print(f"\n最佳模型: {best_model} (AUC: {best_score:.3f})")
    
    # 查看特征重要性
    best_predictor.get_feature_importance()
    
    # 保存模型
    best_predictor.save_model('nba_win_predictor.pkl')
    
    return best_predictor, feature_df


def calculate_season_carryover(prev_season_games, team_id):
    """
    计算上个赛季末段的表现，用于本赛季初期的特征
    """
    if len(prev_season_games) == 0:
        return None
    
    # 计算每场比赛的结果
    prev_season_games = prev_season_games.copy()
    prev_season_games['is_home'] = prev_season_games['home_id'] == team_id
    prev_season_games['team_pts'] = np.where(
        prev_season_games['is_home'], 
        prev_season_games['home_pts'], 
        prev_season_games['away_pts']
    )
    prev_season_games['opp_pts'] = np.where(
        prev_season_games['is_home'], 
        prev_season_games['away_pts'], 
        prev_season_games['home_pts']
    )
    prev_season_games['win'] = (prev_season_games['team_pts'] > prev_season_games['opp_pts']).astype(int)
    prev_season_games['pts_diff'] = prev_season_games['team_pts'] - prev_season_games['opp_pts']
    
    # 取最后10场比赛（如果不够10场，取所有）
    n_games = min(10, len(prev_season_games))
    last_games = prev_season_games.tail(n_games)
    
    return {
        'carryover_win_pct': last_games['win'].mean() if n_games > 0 else 0.5,
        'carryover_avg_pts_diff': last_games['pts_diff'].mean() if n_games > 0 else 0.0,
        'carryover_games': n_games,
        'prev_season_final_win_pct': prev_season_games['win'].mean(),
        'prev_season_total_games': len(prev_season_games)
    }

def build_complete_dataset(games_df, start_season=1984, end_season=2025):
    """
    为所有比赛构建特征数据集
    """
    processor = NBADataProcessor(games_df)
    
    all_records = []
    
    seasons = sorted(games_df['season'].unique())
    seasons = [s for s in seasons if s >= start_season and s <= end_season]
    
    print(f"正在处理 {len(seasons)} 个赛季的数据...")
    
    for season_idx, season in enumerate(seasons): 
        print(f"处理赛季 {season} ({season_idx+1}/{len(seasons)})")
        
        season_games = games_df[games_df['season'] == season].copy()
        
        for _, game in season_games.iterrows():
            # 为每场比赛构建特征
            game_date = game['game_date']
            home_team = game['home_id']
            away_team = game['away_id']
            
            # 如果是赛季早期比赛，可能需要上个赛季的数据
            if season_idx > 0:  # 不是第一个赛季
                prev_season = seasons[season_idx - 1]
                
                # 获取主队上个赛季的数据
                home_prev_games = games_df[
                    (games_df['season'] == prev_season) & 
                    ((games_df['home_id'] == home_team) | 
                     (games_df['away_id'] == home_team))
                ].copy()
                
                # 获取客队上个赛季的数据  
                away_prev_games = games_df[
                    (games_df['season'] == prev_season) & 
                    ((games_df['home_id'] == away_team) | 
                     (games_df['away_id'] == away_team))
                ].copy()
                
                # 计算上个赛季末段的表现（最后10-20场）
                home_carryover = calculate_season_carryover(home_prev_games, home_team)
                away_carryover = calculate_season_carryover(away_prev_games, away_team)
            else:
                # 第一个赛季，没有上个赛季数据
                home_carryover = None
                away_carryover = None

            # 获取主队特征（考虑赛季结转）
            home_features = processor.calculate_team_features_with_carryover(
                home_team, game_date, season, home_carryover
            )
            
            # 获取客队特征（考虑赛季结转）
            away_features = processor.calculate_team_features_with_carryover(
                away_team, game_date, season, away_carryover
            )
            
            # 合并特征并添加前缀
            features = {}
            features['is_playoff'] = game['is_playoff']

            # 添加主队特征（带前缀）
            for key, value in home_features.items():
                if key not in ['current_date', 'season']:
                    features[f'home_{key}'] = value
            
            # 添加客队特征（带前缀）
            for key, value in away_features.items():
                if key not in ['current_date', 'season']:
                    features[f'away_{key}'] = value
            
            # 添加比赛上下文特征
            features['game_date'] = game_date
            features['season'] = season
            
            # 添加差异特征
            features['win_pct_diff'] = features.get('home_total_win_pct', 0.5) - features.get('away_total_win_pct', 0.5)
            features['rest_days_diff'] = features.get('home_rest_days', 7) - features.get('away_rest_days', 7)
            features['streak_diff'] = features.get('home_current_streak', 0) - features.get('away_current_streak', 0)
            
            # 如果是第一个赛季，添加标记
            if season_idx == 0:
                features['is_first_season'] = 1
                features['has_prev_season_data'] = 0
            else:
                features['is_first_season'] = 0
                features['has_prev_season_data'] = 1


            # 添加比赛结果（标签）
            home_pts = game['home_pts']
            away_pts = game['away_pts']
            features['home_win'] = 1 if home_pts > away_pts else 0
            features['pts_diff'] = home_pts - away_pts
            
            all_records.append(features)
        
        # 每5个赛季保存一次进度
        if season_idx % 5 == 0:
            temp_df = pd.DataFrame(all_records)
            print(f"  已处理 {len(temp_df)} 场比赛")
    
    # 创建完整DataFrame
    final_df = pd.DataFrame(all_records)
    
    # 确保特征顺序一致
    feature_columns = [col for col in final_df.columns if col not in ['home_win', 'game_date', 'season', 'pts_diff']]
    final_df = final_df[feature_columns + ['home_win', 'game_date', 'season', 'pts_diff']]
    
    regular_home_win = final_df[final_df['is_playoff'] == 0]['home_win'].mean()
    post_home_win = final_df[final_df['is_playoff'] == 1]['home_win'].mean()
    print(f"\n数据集构建完成！")
    print(f"总比赛场次: {len(final_df)}")
    print(f"常规赛主队胜率: {regular_home_win:.2%}")
    print(f"季后赛主队胜率: {post_home_win:.2%}")
    
    return final_df




if __name__ == '__main__':
    # 需要当赛季所有球队的特征
    # team_stats = cal_team_stats_todate(1610612747, 2024, '2024-01-03')
    # print(result_stats)
    # df_raw = pd.read_csv(os.path.join(RAW_DATA_PATH, "tb_games.csv"))
    # df_raw['game_date'] = pd.to_datetime(df_raw['game_date'])
    # s:pd.DataFrame = build_complete_dataset(df_raw).round(2)
    # s.to_csv(os.path.join(PROCESS_DATA_PATH, 'games_features.csv'), index=False)

    train_complete_pipeline()
