"""探索胜率相关性(常规赛)
胜率 vs:
场均得分，进攻效率(每百回合)，进攻篮板率，助攻率
防守效率(每百回合)，防守篮板率，

总篮板率 = 篮板数(己方总篮板数)/总篮板机会(对方+己方)
进攻篮板率 = 进攻篮板数/出手数(己方)
防守篮板率 = 防守篮板数/出手数(对方)

totoal RB 3672
total chances 7600-3660 + 7130-3109

"""
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import seaborn as sns
import pymysql
from pymysql.cursors import DictCursor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

DB_INFO = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': 'root',
    'database': 'new_nba_data',
    'charset': 'utf8'
}

RAW_DATA_PATH = os.path.join(os.getcwd(), f'raw_data')
PROCESS_DATA_PATH = os.path.join(os.getcwd(), f'process_data')




def load_data():
    connect = pymysql.connect(**DB_INFO,cursorclass=DictCursor)
    cursor = connect.cursor()

    query_sql = """
    select tltt.season, tltt.team, tltt.team_short, tltt.is_opponent, tltt.is_playoff,tltt.FG,
    tltt.FGA,tltt.FT,tltt.FTA,tltt.ORB,tltt.DRB,tltt.TRB,tltt.AST,tltt.STL,tltt.BLK,tltt.TOV,tltt.PF, tltg.PTS as PTSAvg
    from tb_leagues_team_total tltt left join tb_leagues_team_pergame tltg on tltt.season = tltg.season and tltt.team = tltg.team and 
     tltt.is_playoff = tltg.is_playoff and tltt.is_opponent = tltg.is_opponent
    """

    cursor.execute(query_sql)
    raw_data = cursor.fetchall()
    dtype_spec = {
    'FG': 'int64',
    'FGA': 'int64',
    'FT': 'int64',
    'FTA': 'int64',
    'ORB': 'int64',
    'DRB': 'int64',
    'TRB': 'int64',
    'AST': 'int64',
    'STL': 'int64',
    'BLK': 'int64',
    'TOV': 'int64',
    'PF': 'int64',
    # 'PTS': 'int64',
    'PTSAvg': 'float64',
    'season': 'int64'  # 或者其他合适的类型
    }


    df = pd.DataFrame(raw_data)
    df = df.astype(dtype_spec)
    
    df['is_playoff'] = df['is_playoff'].apply(lambda x: 'Yes' if x==1 else 'No')
    df['is_opponent'] = df['is_opponent'].apply(lambda x: 'Yes' if x==1 else 'No')

    file_path = os.path.join(RAW_DATA_PATH,'teams pers.csv')
    save_df = df.round(2)
    save_df.to_csv(file_path, sep=',', index=False)
    return save_df


def cal_corr():
    """
    球队进攻，防守篮板率，助攻率，场均得分
    """
    df_total = load_data()
    df_team = df_total[df_total['is_opponent'] == 'No'].copy().drop(columns=['is_opponent'],axis=1)
    df_opp = df_total[df_total['is_opponent'] == 'Yes'].copy().drop(columns=['is_opponent'],axis=1)
    df_opp = df_opp.rename(columns={"FG":"opp_FG","FGA":"opp_FGA","FT":"opp_FT","FTA":"opp_FTA","ORB":"opp_ORB","DRB":"opp_DRB","TRB":"opp_TRB",
    "AST":"opp_AST","STL":"opp_STL","BLK":"opp_BLK","TOV":"opp_TOV","PF":"opp_PF","PTSAvg":"opp_PTSAvg"})

    df_merge = df_team.merge(df_opp, on=['season', 'team', 'is_playoff'], how='left')
    
    # df_merge['PTSAvg'] = df_merge['PTS'] / 82
    df_merge['TORBPct'] = df_merge['ORB']/(df_merge['FGA'] - df_merge['FG'] + df_merge['FTA'] - df_merge['FT'])
    df_merge['TDRBPct'] = df_merge['DRB']/(df_merge['opp_FGA'] - df_merge['opp_FG'] + df_merge['opp_FTA'] - df_merge['opp_FT'])
    df_merge['ASTPct'] = df_merge['AST']/df_merge['FG']
    
    df_merge = df_merge[['season','team', 'is_playoff','PTSAvg','TORBPct', 'TDRBPct', 'ASTPct']]

    df_advs = pd.read_csv(os.path.join(RAW_DATA_PATH, 'teams advanced.csv'), usecols=['season', 'team','is_playoff','Wins','Losses',
                                                                                      'ORtg', 'DRtg', 'eFGPct', 'TOVPct', 'ORBPct', 'FT_Rate',
                                                                                      'opp_eFGPct', 'opp_TOVPct', 'DRBPct', 'opp_FT_Rate'])
    df_res = df_merge.merge(df_advs, on=['season', 'team', 'is_playoff'], how='left').reset_index()
    

    # matrix_corr = df_res[['Wins','ORtg', 'DRtg','PTSAvg','ORBPct', 'DRBPct', 'ASTPct']].corr() # 两两计算相关系（包含自身与自身），一个矩阵
    
    df_grouped = df_res.groupby(by='is_playoff')[['Wins','ORtg', 'DRtg','PTSAvg','TORBPct', 'TDRBPct', 'ASTPct', 'eFGPct', 'TOVPct', 'ORBPct', 'FT_Rate',
                                                                                      'opp_eFGPct', 'opp_TOVPct', 'DRBPct', 'opp_FT_Rate']].corr().reset_index()
    # df_grouped = df_res.groupby(by='is_playoff').apply(lambda x: pd.Series({
    #     'Corr Wins ORBPct': x['Wins'].corr(x['ORBPct']),
    #     'Corr Wins DRBPct': x['Wins'].corr(x['DRBPct']),
    #     'Corr Wins ASTPct': x['Wins'].corr(x['ASTPct']),
    #     'Corr Wins PTSAvg': x['Wins'].corr(x['PTSAvg']),
    #     'Corr Wins ORtg': x['Wins'].corr(x['ORtg']),
    #     'Corr Wins DRtg': x['Wins'].corr(x['DRtg']),
    # }), include_groups=False).reset_index()
    # print('CORR2:', df_grouped)
    df_grouped.round(2).to_csv(os.path.join(PROCESS_DATA_PATH, 'Wins Corr.csv'), index=False)
    return df_res, df_grouped


def cal_corr2():
    """综合指标
    NRtg, OFF/DEF, 
    """
    df_total = load_data()
    df_team = df_total[df_total['is_opponent'] == 'No'].copy().drop(columns=['is_opponent'],axis=1)
    df_opp = df_total[df_total['is_opponent'] == 'Yes'].copy().drop(columns=['is_opponent'],axis=1)
    df_opp = df_opp.rename(columns={"FG":"opp_FG","FGA":"opp_FGA","FT":"opp_FT","FTA":"opp_FTA","ORB":"opp_ORB","DRB":"opp_DRB","TRB":"opp_TRB",
    "AST":"opp_AST","STL":"opp_STL","BLK":"opp_BLK","TOV":"opp_TOV","PF":"opp_PF","PTSAvg":"opp_PTSAvg"})

    df_merge = df_team.merge(df_opp, on=['season', 'team', 'is_playoff'], how='left')
    
    # df_merge['PTSAvg'] = df_merge['PTS'] / 82
    df_merge['TORBPct'] = df_merge['ORB']/(df_merge['FGA'] - df_merge['FG'] + df_merge['FTA'] - df_merge['FT'])
    df_merge['TDRBPct'] = df_merge['DRB']/(df_merge['opp_FGA'] - df_merge['opp_FG'] + df_merge['opp_FTA'] - df_merge['opp_FT'])
    df_merge['ASTPct'] = df_merge['AST']/df_merge['FG']
    
    df_merge = df_merge[['season','team', 'is_playoff','PTSAvg','TORBPct', 'TDRBPct', 'ASTPct']]

    df_advs = pd.read_csv(os.path.join(RAW_DATA_PATH, 'teams advanced.csv'), usecols=['season', 'team','is_playoff','Wins','Losses',
                                                                                      'ORtg', 'DRtg', 'eFGPct', 'TOVPct', 'ORBPct', 'FT_Rate',
                                                                                      'opp_eFGPct', 'opp_TOVPct', 'DRBPct', 'opp_FT_Rate'])
    df_res = df_merge.merge(df_advs, on=['season', 'team', 'is_playoff'], how='left').reset_index()
    
    df_res['NRtg'] = df_res['ORtg'] - df_res['DRtg']
    df_res['OffDefBalance'] = df_res['ORtg'] / df_res['DRtg']
    # df_grouped['PointDiff'] = df_grouped['ORtg'] - df_grouped['DRtg']
    weights = {'ORtg': 0.4, 'DRtg': -0.3, 'DRBPct': 0.1, 'ORBPct': 0.1, 'TOVPct': -0.1}  # DRtg负权重
    df_res['CompositeScore'] = sum(df_res[col] * weight for col, weight in weights.items())
    # matrix_corr = df_res[['Wins','ORtg', 'DRtg','PTSAvg','ORBPct', 'DRBPct', 'ASTPct']].corr() # 两两计算相关系（包含自身与自身），一个矩阵
    
    df_grouped = df_res.groupby(by='is_playoff')[['Wins','ORtg', 'DRtg','NRtg','OffDefBalance','CompositeScore', 'PTSAvg','TORBPct', 'TDRBPct', 'ASTPct', 'eFGPct', 'TOVPct', 'ORBPct', 'FT_Rate',
                                                                                      'opp_eFGPct', 'opp_TOVPct', 'DRBPct', 'opp_FT_Rate']].corr().reset_index()
    
    # df_grouped = df_res.groupby(by='is_playoff').apply(lambda x: pd.Series({
    #     'Corr Wins ORBPct': x['Wins'].corr(x['ORBPct']),
    #     'Corr Wins DRBPct': x['Wins'].corr(x['DRBPct']),
    #     'Corr Wins ASTPct': x['Wins'].corr(x['ASTPct']),
    #     'Corr Wins PTSAvg': x['Wins'].corr(x['PTSAvg']),
    #     'Corr Wins ORtg': x['Wins'].corr(x['ORtg']),
    #     'Corr Wins DRtg': x['Wins'].corr(x['DRtg']),
    # }), include_groups=False).reset_index()
    # print('CORR2:', df_grouped)
    df_grouped.round(2).to_csv(os.path.join(PROCESS_DATA_PATH, 'Wins Corr.csv'), index=False)
    return df_res, df_grouped

def draw_scatter1(df, features):
    for idx, feature in enumerate(features, 1):
        plt.subplot(4,4 , idx)
        
        sns.scatterplot(data=df, x=feature, y='Wins', alpha=0.6)
        # sns.scatterplot( x=df[feature].values, y=df['Wins'].values, alpha=0.6)
        sns.regplot(x=df[feature].values, y=df['Wins'].values, scatter=False, 
                    line_kws={'color': 'red'})
        
        corr_val = df[feature].corr(df['Wins'])

        # 3. 添加相关性说明（关键代码）
        corr_text = f'r = {corr_val:.3f}'

        # 根据相关系数值添加解释
        if abs(corr_val) >= 0.7:
            strength = "强相关"
        elif abs(corr_val) >= 0.4:
            strength = "中等相关"
        elif abs(corr_val) >= 0.2:
            strength = "弱相关"
        else:
            strength = "极弱相关"

        direction = "正" if corr_val > 0 else "负"

        # 在图上显示
        plt.text(0.05, 0.95, f'{corr_text}\n{direction}{strength}', 
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.title(f'{feature}\nr = {corr_val:.3f}')
        plt.grid(True, alpha=0.3)

def multi_regression(df: pd.DataFrame):
    # dimensions = ['ORtg', 'DRtg', 'ASTPct', 'ORBPct']
    dimensions = ['NRtg']
    # 1. 准备数据
    X = df[dimensions]
    y = df['Wins']
    print(y.std())

    # # 2. 标准化X和y
    # scaler_X = StandardScaler()
    # scaler_y = StandardScaler()

    # X_scaled = scaler_X.fit_transform(X)
    # y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    # 3. 建模
    model_raw = LinearRegression()
    model_raw.fit(X, y)


    print("多元回归结果：")
    print(f"R² = {model_raw.score(X, y):.3f}")
    for i, col in enumerate(dimensions):
        print(f"  {col}: 系数 = {model_raw.coef_[i]:.3f}")
   # 计算标准化系数用于比较重要性
    std_coeffs = model_raw.coef_ * X.std().values / y.std()
    print("\n标准化系数（比较变量重要性）：")
    for i, col in enumerate(dimensions):
        print(f"  {col}: β_std = {std_coeffs[i]:.3f}")

    # 重要性排序
    print("\n变量重要性排序（|β_std|从大到小）：")
    importance = sorted(zip(dimensions, 
                        np.abs(std_coeffs)), 
                        key=lambda x: x[1], reverse=True)
    for col, importance_score in importance:
        print(f"  {col}: |β_std| = {importance_score:.3f}")


"""
目标: (球队层面找出与胜场较相关的指标(corr > 0.3)
1.从进阶数据找
2.从每百回找

"""

def get_data():
    df_100poss = pd.read_csv(os.path.join(RAW_DATA_PATH, 'teams 100poss.csv'))
    df_advanced = pd.read_csv(os.path.join(RAW_DATA_PATH, 'teams advanced.csv'))
    df_advanced = df_advanced.drop([ 'champion', 'Losses'], axis=1)
    df_100possTeam = df_100poss[df_100poss['is_opponent'] == 0]
    df_100possTeam = df_100possTeam.drop(['is_opponent'], axis=1)
    df_100possTeam['is_playoff'] = df_100possTeam['is_playoff'].apply(lambda x: 'Yes' if x == 1 else 'No')

    df_wins = df_advanced[['season', 'team', 'is_playoff', 'Wins']]
    
    df_100possTeam = df_100possTeam.merge(df_wins, on=['team', 'season', 'is_playoff'], how='left')
    exclude_cols = ['season', 'team', 'team_short']
    # cols_100poss = [col for col in df_100possTeam.columns.to_list() if col not in exclude_cols ]
    # cols_adv = [col for col in df_advanced.columns.to_list() if col not in exclude_cols ]

    df_advanced = df_advanced.drop(exclude_cols, axis=1)
    df_100possTeam = df_100possTeam.drop(exclude_cols, axis=1)

    return df_advanced, df_100possTeam
    # df_advanced_reg = df_advanced[df_advanced['is_playoff'] == 'No']
    # df_advanced_post = df_advanced[df_advanced['is_playoff'] == 'Yes']
    # df_100possTeam_reg = df_100possTeam[df_100possTeam['is_playoff'] == 'No']
    # df_100possTeam_post = df_100possTeam[df_100possTeam['is_playoff'] == 'Yes']

    # df_corr100reg = df_100possTeam_reg[cols_100poss].corr()
    # df_corr100post = df_100possTeam_post[cols_100poss].corr()
    
    # df_corradvreg = df_advanced_reg[cols_100adv].corr()
    # df_corradvpost = df_advanced_post[cols_100adv].corr()
    
    
    # # cols = [for col in df_merge.columns.to_list() if col not in ('team', 'season', 'is_playoff', 'is_opponent', 'team_short',)]
    
    # print('常规赛每百回合相关系数:\n',np.abs(df_corr100reg['Wins']).sort_values(ascending=False))
    # print('季后赛每百回合相关系数:\n',np.abs(df_corr100post['Wins']).sort_values(ascending=False))
    # print('常规赛进阶数据相关系数:\n',np.abs(df_corradvreg['Wins']).sort_values(ascending=False))
    # print('季后赛进阶数据相关系数:\n',np.abs(df_corradvpost['Wins']).sort_values(ascending=False))



    # print(df_merge.info())
    # print(df_100possTeam.info())

def get_corr_series(df_corr: pd.DataFrame, target='Wins'):
    corr_series:pd.Series = df_corr[target]
    if target and target in corr_series.index:
        corr_series = corr_series.drop(target, axis=0)

    return corr_series

def filter_strong_metrics(df_corr, threshold=0.2):

    
    corr_series = get_corr_series(df_corr)
    corr_series = corr_series[corr_series.abs() > threshold]
    return corr_series

def compare_regular_post(df_corr_reg, df_corr_post, threshold=0.2):

    strong_regular = filter_strong_metrics(df_corr_reg, threshold)
    strong_playoff = filter_strong_metrics(df_corr_post, threshold)
    
    common_metrics = set(strong_regular.index.to_list()) & set(strong_playoff.index.to_list())
    regular_only = set(strong_regular.index.to_list()) - set(strong_playoff.index.to_list())
    playoff_only = set(strong_playoff.index.to_list()) - set(strong_regular.index.to_list())

    print(f"\n共同强指标 ({len(common_metrics)}个):")
    for metric in sorted(common_metrics):
        r_reg = strong_regular[metric]
        r_po = strong_playoff[metric]
        diff = r_po - r_reg
        arrow = "↑" if diff > 0 else "↓"
        print(f"  {metric:15}: 常规赛={r_reg:.3f}, 季后赛={r_po:.3f} ({arrow}{abs(diff):.3f})")
    

    print(f"\n仅常规赛强 ({len(regular_only)}个):")
    for metric in sorted(regular_only):
        print(f"  {metric:15}: r={strong_regular[metric]:.3f} (常规赛重要)")
    
    print(f"\n仅季后赛强 ({len(playoff_only)}个):")
    for metric in sorted(playoff_only):
        print(f"  {metric:15}: r={strong_playoff[metric]:.3f} (季后赛重要)")
    
    return list(common_metrics), list(regular_only), list(playoff_only)


# 筛选>0.3 0.5
# 季后赛与常规赛共同强指标和对应的独立强指标
# 分类指标
# 每百回合："FG","FGA","FGPct","ThreeP","ThreePA","ThreePPct","TwoP","TwoPA","TwoPPct","FT","FTA","FTPct","ORB","DRB","TRB","AST","STL","BLK","TOV","PF","PTS"
# 进阶数据：ORtg,DRtg,NRtg,Pace,FTAr,TSPct,eFGPct,TOVPct,ORBPct,FT_Rate,opp_eFGPct,opp_TOVPct,DRBPct,opp_FT_Rate,ThreePPct,TwoPPct,eASTPct
# 百回合: ["TRB", "TwoPPct", "AST", "BLK", "STL"] (R:0.717) 恰好就是基础数据 得分，篮板，助攻，抢断，盖帽
# 进阶： [ORtg,DRtg], ["eFGPct", "TOVPct", "ORBPct", "FT_Rate"]
categories = {
        '得分效率': ['PTS', 'FGPct', 'TwoPPct', 'ThreePPct', 'eFGPct', 'TSPct', 'ORtg'],
        '篮板控制': ['DRB', 'TRB', 'DRBPct', 'ORBPct'],
        '组织助攻': ['AST', 'ASTPct'],
        '防守能力': ['DRtg', 'STL', 'BLK', 'opp_eFGPct'],
        '失误控制': ['TOV', 'TOVPct', 'opp_TOVPct'],
        '罚球相关': ['FT', 'FTPct', 'FTA', 'FT_Rate'],
        '进攻节奏': ['Pace'],
        '球队经验': ['age'],
        '综合效率': ['NRtg']
    }
# 创建复合指标(回归系数)

def regressions(df, dimensions=[], y_col = 'Wins'):
    if not dimensions:
        return
    model = LinearRegression()

    x = df[dimensions]
    y = df[y_col]
    model.fit(x, y)

    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # 计算VIF
    if len(x.columns) > 1:
        vif_data = pd.DataFrame()
        vif_data["feature"] = x.columns
        vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]
        print("VIF(共线性):",vif_data)


    print('多元线性回归系数')
    print(f'R²={model.score(x,y):.3f} ({y_col})')
    for i, col in enumerate(dimensions):
        print(f'{col}: 系数= {model.coef_[i]:0.3f}')


def pca_check(df, offense_vars, defense_vars):
    # 对高度相关的变量组做PCA
    # offense_vars = ['TwoPPct', 'AST', 'PTS', 'ThreePPct']
    # defense_vars = ['BLK', 'STL', 'DRB']

    # 进攻维度PCA
    scaler = StandardScaler()
    offense_scaled = scaler.fit_transform(df[offense_vars])
    pca_offense = PCA(n_components=1)
    df['Offense_PC1'] = pca_offense.fit_transform(offense_scaled)

    # 防守维度PCA  
    defense_scaled = scaler.fit_transform(df[defense_vars])
    pca_defense = PCA(n_components=1)
    df['Defense_PC1'] = pca_defense.fit_transform(defense_scaled)



    print(f"进攻主成分解释方差: {pca_offense.explained_variance_ratio_[0]:.1%}")
    print(f"防守主成分解释方差: {pca_defense.explained_variance_ratio_[0]:.1%}")


# 


if __name__ == '__main__':

   
    df_advanced, df_100possTeam = get_data()
    df_advanced_reg = df_advanced[df_advanced['is_playoff'] == 'No'].drop(['is_playoff'],axis=1)
    df_advanced_post = df_advanced[df_advanced['is_playoff'] == 'Yes'].drop(['is_playoff'],axis=1)
    df_100possTeam_reg = df_100possTeam[df_100possTeam['is_playoff'] == 'No'].drop(['is_playoff'],axis=1)
    df_100possTeam_post = df_100possTeam[df_100possTeam['is_playoff'] == 'Yes'].drop(['is_playoff'],axis=1)

    df_corr100reg = df_100possTeam_reg.corr()
    df_corr100post = df_100possTeam_post.corr()
    
    df_corradvreg = df_advanced_reg.corr()
    df_corradvpost = df_advanced_post.corr()
    # print(df_100possTeam)
    print('每百回合数据：')
    common_metrics, regular_only, playoff_only = compare_regular_post(df_corr100reg, df_corr100post)
    # print(common_metrics)

    regressions(df_corr100reg, ["TRB", "TwoPPct", "AST", "BLK", "STL"])
    
    print('进阶数据：') # 属于结果描述性指标的描述性分析() 进阶数据是赛后统计计算得出。
    common_metrics, regular_only, playoff_only = compare_regular_post(df_corradvreg, df_corradvpost)
    # df_adv = df_advanced.drop(['is_playoff'], axis=1)
    # adv_corr = df_adv.corr()
    regressions(df_corradvreg, ["eFGPct","TOVPct","ORBPct","FT_Rate"])

    print("PCA:")
    off_vars = ["ORtg"]
    def_vars = ["DRtg"]
    pca_check(df_corradvreg, off_vars, def_vars)

    # complete_metric_analysis(df_advanced, df_corradvreg, df_corradvpost)
    # df, matrix_corr = cal_corr2()
    # print(df[(df['season'] == 2024) & (df['team'] == 'Dallas Mavericks')])
    # df = df[df['is_playoff'] == 'No']
    
    

    # corr_cols =['Wins','ORtg', 'DRtg','PTSAvg',
    #             'TORBPct', 'TDRBPct', 'ASTPct','eFGPct', 
    #             'TOVPct', 'ORBPct', 'FT_Rate', 'opp_eFGPct', 
    #             'opp_TOVPct', 'DRBPct', 'opp_FT_Rate']
    
    # # corr_cols2= ['Wins','NRtg', 'OffDefBalance', 'CompositeScore']
    # corr_cols2 = ['Wins', 'ORtg', 'DRtg', 'NRtg','ASTPct', 'ORBPct']

    # correlation_matrix = matrix_corr[matrix_corr['is_playoff'] == 'Yes'][corr_cols2]
    # plt.figure(figsize=(12, 10))
    # # df['Wins']
    # # print(df.dtypes)
    # # 创建热力图
    # # mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # # ax = sns.heatmap(correlation_matrix, 
    # #             mask=mask,
    # #             annot=True, 
    # #             fmt='.2f', 
    # #             cmap='coolwarm',
    # #             center=0,
    # #             square=True,
    # #             linewidths=1,
    # #             cbar_kws={"shrink": 0.8})
    # # ax.set_xticklabels(correlation_matrix.columns, 
    # #                rotation=45,  # 旋转45度防止重叠
    # #                ha='right',   # 水平对齐方式
    # #                fontsize=12)

    # # ax.set_yticklabels(correlation_matrix.columns, 
    # #                 rotation=0,   # Y轴标签不旋转
    # #                 fontsize=12)
    # # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # # ax.set_yticklabels(ax.get_xticklabels(), rotation=0)

    # # plt.title('球队表现指标相关系数矩阵热力图', fontsize=16, fontweight='bold')
    # # plt.tight_layout()
    # # plt.show()

    # # 散点图

    # # 选择前3个相关指标
    # features = correlation_matrix.columns[1:].to_list()
    # # print(df[df['Wins'] == 0])
    
    # multi_regression(df)
    # draw_scatter1(df, features)
    # plt.suptitle('胜场数相关性分析', fontsize=14, fontweight='bold')
    # plt.tight_layout()
    # plt.show()
    
        
        
