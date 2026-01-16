"""特征构建，是核心

在对预测比赛或比较优劣(A VS B -> result[W/L, Good/Bad])构建特征时，我发现一些通用规则：
0.预测前置环境(如比赛中的赛程，休息天数)
1.对象(A,B) 各自的指标->作为特征之一(前提是按时间序列)
2.A,B的共有特征(例如比赛胜率，进攻效率) 需要比较分项优劣
    2.1: 使用同期的所有对象平均值
    2.2: 使用同期的各自对手平均值
3.最后使用综合优劣
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
    # 第一层: 结果类
    # 第二层：质量类
    # 第三层: 趋势
    

    def __init__(self, team_schedules:pd.DataFrame, games_schedules:pd.DataFrame):
        """
        初始化处理器
        
        team_schedules(以球队为主的赛程):
            包含列 ['game_id','game_date', 'season', 'team_id', 'opp_id', 'is_home',
                          'team_pts', 'opp_pts', 'team_id', 'opp_id',
                          'team_four_factors', opp_four_fators, clutch_game, streak, pace]
        games_schedules:联盟赛程 game_date, season, game_id, home_team, away_team,
        """
        
        self.team_schedules = team_schedules.copy()
        self.games_schedules = games_schedules.copy()
        self.window_feature_keys = ['avg_pts_diff', 'avg_eFG_pct', 'avg_TOV_pct', 'avg_ORB_pct', 'avg_ORB_pct'
                           'avg_opp_eFG_pct', 'avg_opp_TOV_pct', 'avg_DRB_pct', 'avg_opp_FT_FGA', 'avg_pace', 'wins_pct',
                           'home_wins_pct', 'away_wins_pct'
                           ]
        self.windows_name = {
            5: 'last5',
            10: 'last10'
        }
        self.team_schedules['game_date'] = pd.to_datetime(self.team_schedules['game_date'])
        self.team_schedules = self.team_schedules.sort_values(['season', 'game_date'])

    def _get_default_features(self):
        """所有特征默认值
        特征分层(分类)
        """
        team_features = {
            # 球队目前状态
            'games_played': 0, # 已比赛场次
            'wins_pct': 0.5, # 当前胜率
            'current_streak': 0, # 当前连胜
            'vs_strong_team_win_pct': 0.3, # 对阵胜率50%或以上的球队胜率
            'home_streak': 0, # 连续客场
            'away_streak': 0, # 连续主场
            'back_to_back': 0, # 背靠背比赛
            'rest_days': 100, # 休息日期

            # 主客场:
            'home_wins_pct': 0, # 主场胜率
            'away_wins_pct': 0, # 客场胜率

            # 进攻四要素
            'avg_eFG_pct': 0.5, # 有效命中率
            'avg_TOV_pct': 5, # 每百回合失误数
            'avg_ORB_pct': 0.1, # 进攻篮板率
            'avg_FT_FGA': 0.1, # 罚球率(每次出手获得罚球数)
            
            # 防守四要素
            'avg_opp_eFG_pct': 0.5, # 有效命中率
            'avg_opp_TOV_pct': 5, # 每百回合失误数
            'avg_DRB_pct': 0.1, # 防守篮板率
            'avg_opp_FT_FGA': 0.1, # 罚球率(每次出手获得罚球数)
            
            # 关键时刻
            'clutch_wins_pct': 0, # 关键时刻比赛胜率
            'clutch_pts_diff': 0, # 关键时刻净胜分

            # 节奏
            'avg_pace': 100, # 48分钟回合数
            
            # 趋势(近5场 vs 近5场之前):
        }
        # 滚动窗口特征
        window_feature = team_features.copy()
        window_feature = {f'{k}_{window}':v for window in self.windows_name.values() for k, v in window_feature.items() if k in self.window_feature_keys}
        
        # 若是新球队第一个赛季, 则使用默认值填充
        # 当样本不足够 可以使用上赛季联盟平均值填充 (后续添加)
        team_features.update(window_feature)

        

        return team_features


    def build_all_features(self, home_id, away_id, season, current_date):
        teams_games_recent: pd.DataFrame = self.team_schedules[
                        (self.team_schedules['season'] == season)
                        & (self.team_schedules['game_date'] < current_date)
                        ].copy()
        
        # 各球队在current_date前的胜率
        all_features = {}
        teams_games_recent['win'] = (teams_games_recent['team_pts'] > teams_games_recent['opp_pts']).astype(int)
        teams_win_pct = teams_games_recent.groupby(['team_id'])['win'].mean()

        home_features = self.get_team_features(home_id, season, current_date, teams_win_pct)
        away_features = self.get_team_features(away_id, season, current_date, teams_win_pct)

        # 衍生特征
        drop_features = ['avg_eFG_pct', 'avg_TOV_pct', 'avg_ORB_pct', 'avg_ORB_pct'
                           'avg_opp_eFG_pct', 'avg_opp_TOV_pct', 'avg_DRB_pct', 'avg_opp_FT_FGA' ]
        off_advantage = home_features['avg_eFG_pct'] - away_features['avg_opp_eFG_pct'] # +:A进攻好
        def_advantage = away_features['avg_eFG_pct'] - home_features['avg_opp_eFG_pct'] # +:A防守好
        off_def_diff = off_advantage - def_advantage  # 进攻防守综合差异
        orb_advantage = home_features['avg_ORB_pct'] - away_features['avg_ORB_pct'] 
        drb_advantage = home_features['avg_DRB_pct'] -away_features['avg_DRB_pct']
        trb_diff = (home_features['avg_ORB_pct'] + home_features['avg_DRB_pct']) - \
                (away_features['avg_ORB_pct'] + away_features['avg_DRB_pct'])
        
        tov_diff = away_features['avg_TOV_pct'] - home_features['avg_TOV_pct']  # +:A失误少，球权控制好
        draw_tov_diff = home_features['avg_opp_TOV_pct'] - away_features['avg_opp_TOV_pct'] #  +:A制造对手失误比B好
        net_tov_impact = (home_features['avg_TOV_pct'] - home_features['avg_opp_TOV_pct']) - \
            (away_features['avg_TOV_pct'] - away_features['avg_opp_TOV_pct'])  # 净回合优势(A净回合优势-B净回合优势)
        
        draw_foul_diff = home_features['avg_FT_FGA'] -  away_features['avg_FT_FGA'] # 造杀伤能力(每次出手获得罚球次数)
        opp_draw_foul_diff =  away_features['avg_opp_FT_FGA'] - home_features['avg_opp_FT_FGA']  # 对手造杀伤能力(每次出手获得罚球次数)
        net_foul_advantage = (home_features['avg_FT_FGA'] - home_features['avg_opp_FT_FGA']) - \
             (away_features['avg_FT_FGA'] - away_features['avg_opp_FT_FGA'])

        pace_diff = home_features['avg_pace'] - away_features['avg_pace']


        for k, v in home_features.items():
            if k not in drop_features:
                all_features[f'home_{k}'] = v

        for k, v in away_features.items():
            if k not in drop_features:
                all_features[f'away_{k}'] = v

        all_features.update({
            'off_advantage': off_advantage,
            'def_advantage': def_advantage,
            'off_def_diff': off_def_diff,
            'orb_advantage': orb_advantage,
            'drb_advantage': drb_advantage,
            'trb_diff': trb_diff,
            'tov_diff': tov_diff,
            'draw_tov_diff': draw_tov_diff,
            'net_tov_impact': net_tov_impact,
            'draw_foul_diff': draw_foul_diff,
            'opp_draw_foul_diff': opp_draw_foul_diff,
            'net_foul_advantage': net_foul_advantage,
            'pace_diff': pace_diff
        })
        return all_features
        # 待优化
        # 1.使用同期联盟基准值对比
        # 2.使用同期各自对手平均值对比

    def get_team_features(self, team_id, season ,current_date, teams_win_pct, current_game_id):
        
        # 初始化特征默认值
        features = self._get_default_features()

        team_games: pd.DataFrame = self.team_schedules[
                    (self.team_schedules['team_id'] == team_id) 
                        & (self.team_schedules['season'] == season)
                        & (self.team_schedules['game_date'] < current_date)
                        ].copy()
        
        if len(team_games) == 0:
            return features
        
        features.update(self._build_team_features(team_games))
        
    

        # 对阵强队胜率
        strong_teams = teams_win_pct[
            (teams_win_pct >= 0.5) & 
            (teams_win_pct.index.isin(team_games['opp_id']))
        ].index
        
        vs_strong_games = team_games[team_games['opp_id'].isin(strong_teams)]
        if not vs_strong_games.empty:
            vs_strong_team_win_pct = vs_strong_games['win'].mean()
        else:
            vs_strong_team_win_pct = 0.3

        # 获取最后一场比赛日期
        last_game_date = team_games['game_date'].max()
        rest_days =  (current_date - last_game_date).days - 1 # 连续休息天数

        # 连续客场或主场 
        current_home_or_away = self.games_schedules[(self.games_schedules['game_id'] == current_game_id) & (self.games_schedules['home_id'] == team_id)].count()
        cnt_streak_home_away = 0
        desc_games = team_games.sort_values(by='game_id',ascending=False)
        for idx, row in desc_games.iterrows():
            if row['is_home'] == current_home_or_away:
                cnt_streak_home_away += 1
            else:
                break
                
        home_away_streak = 'home_streak' if current_home_or_away == 1 else 'away_streak'

        # 背靠背
        back_to_back = 0 if rest_days else 1
        

        features.update({
            'games_played':  len(team_games),
            'rest_days': rest_days,
            'back_to_back': back_to_back,
            'vs_strong_team_win_pct': vs_strong_team_win_pct,
            f'{home_away_streak}': cnt_streak_home_away

        })

        
        # 动态窗口特征（last5, last10）
        for window, window_key in self.windows_name:
           
            if len(team_games) >= window:
                window_games = team_games.tail(window)
                features.update(self._build_team_features(window_games, f'_{window_key}'))
                features.update({f'exits_{window_key}': 1})
            else:
                # 使用默认
                features.update({
                    f'exists_{window_key}': 0
                })
        return features


    def _build_team_features(self, team_games, window=''):
        """构建(计算)球队特征,
            与窗口特征共用, 减少计算代码
        """
        team_games['wins'] = (team_games['team_pts'] > team_games['opp_pts']).astype(int)
        
        wins_pct = team_games['wins'].mean()
        avg_pts_diff = (team_games['team_pts'] - team_games['opp_pts']).mean()

        # 关键时刻
        clutch_games = team_games[team_games['clutch_game'] == 1]

        clutch_wins_pct = clutch_games['wins'].mean() if len(clutch_games) > 0 else 0
        clutch_pts_diff =  (clutch_games['team_pts'] - clutch_games['opp_pts']).mean() if len(clutch_games) > 0 else 0
        
        # 主客场
        home_wins_df = team_games[team_games['is_home'] == 1]
        away_wins_df = team_games[team_games['is_home'] == 0]
        home_wins_pct = home_wins_df['wins'].mean() if len(home_wins_df) > 0 else 0
        away_wins_pct = away_wins_df['wins'].mean() if len(away_wins_df) > 0 else 0
        
        
        # 节奏
        avg_pace = team_games['pace'].mean()
        
      

        features = {

            f'wins_pct{window}': wins_pct,
            f'current_streak{window}': team_games.loc[len(team_games),'streak'],
            f'avg_pts_diff{window}': avg_pts_diff,
            # 对对手平均进攻
            f'avg_eFG_pct{window}': team_games['eFG_pct'].mean(), # 有效命中率
            f'avg_TOV_pct{window}': team_games['TOV_pct'].mean(), # 每百回合失误数
            f'avg_ORB_pct{window}': team_games['ORB_pct'].mean(), # 进攻篮板率
            f'avg_FT_FGA{window}': team_games['FT_FGA'].mean(), # 罚球率(每次出手获得罚球数)
            # 对手平均防守
            f'avg_opp_eFG_pct{window}': team_games['opp_eFG_pct'].mean(), # 有效命中率
            f'avg_opp_TOV_pct{window}': team_games['opp_TOV_pct'].mean(), # 每百回合失误数
            f'avg_DRB_pct{window}': team_games['DRB_pct'].mean(), # 防守篮板率
            f'avg_opp_FT_FGA{window}': team_games['opp_FT_FGA'].mean(), # 罚球率(每次出手获得罚球数)
            
            f'clutch_wins_pct{window}': clutch_wins_pct,
            f'clutch_pts_diff{window}': clutch_pts_diff,
            f'home_wins_pct{window}': home_wins_pct,
            f'away_wins_pct{window}': away_wins_pct,
            f'avg_pace{window}': avg_pace
        }
        
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
            for window in [5, 10]:
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
        return  {
            'team_id': team_id,
            'current_date': current_date,
            'season': season,
        }.update(self.team_features)



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

def build_complete_dataset(team_schedule_df, games_schedule_df, start_season=1984, end_season=2025):
    """
    为所有比赛构建特征数据集
    """
    processor = NBADataProcessor(team_schedule_df, games_schedule_df)
    
    all_records = []
    
    seasons = sorted(games_schedule_df['season'].unique())
    seasons = [s for s in seasons if s >= start_season and s <= end_season]
    
    print(f"正在处理 {len(seasons)} 个赛季的数据...")
    
    for season_idx, season in enumerate(seasons): 
        print(f"处理赛季 {season} ({season_idx+1}/{len(seasons)})")
        
        season_games = games_schedule_df[games_schedule_df['season'] == season].copy()
        
        for _, game in season_games.iterrows():
            # 为每场比赛构建特征
            game_date = game['game_date']
            home_team = game['home_id']
            away_team = game['away_id']
            
            # 暂不考虑赛季初或新球队，优化时使用联盟平均值填充

            # 
            features = processor.build_all_features(home_team, away_team, season, game_date)
            features['is_playoff'] = game['is_playoff']

            # 添加比赛上下文特征
            features['game_date'] = game_date
            features['season'] = season


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


# 球队特征：(要按本赛季表现， 上赛季的数据没有太大参考价值[球队内部变动太大]) 近5,10场
#   基础特征: games_played,total_wins,total_win_pct,rest_days,current_streak,avg_pts_diff_all
#   近5,10场表现: win_pct, avg_pts_diff, home_wins(主场胜场), away_wins(做客胜场)
#   赛程强度：
#       近几个赛季赛程变得紧密，背靠背比赛更多，出现连续做客与连续主场: is_backToback
#       对主场球队：is_home_streaks
#       对客场球队：is_away_streaks
#       

# 球员特征：（本赛季与上赛季，球员生涯连续性较强， 球员表现是球队表现重要因素）