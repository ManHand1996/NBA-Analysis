import pymysql
from pymysql.cursors import DictCursor
import pandas as pd
import numpy as np
import os

DB_INFO = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': 'root',
    'database': 'new_nba_data',
    'charset': 'utf8'
}

RAW_DATA_PATH = os.path.join(os.getcwd(), 'raw_data')
PROCESS_DATA_PATH = os.path.join(os.getcwd(), 'process_data')


def get_leageues_team_adv(db_info):
    connect = pymysql.connect(**db_info,cursorclass=DictCursor)
    cursor = connect.cursor()

    query_sql = """select tlta.season,tlta.team,tlta.team_short,tlta.is_playoff,(case 
	when (tlif.season = tlta.season ) then 'Yes'
	 else 'No'
	end ) as champion,
	tlta.age,tlta.Wins, tlta.Losses,tlta.ORtg, tlta.DRtg, tlta.NRtg, tlta.Pace, tlta.FTAr, tlta.TSPct,
	tlta.eFGPct, tlta.TOVPct,tlta.ORBPct,tlta.FT_Rate,
	tlta.opp_eFGPct, tlta.opp_TOVPct, tlta.DRBPct, tlta.opp_FT_Rate,
	tltpp.ThreePPct, tltpp.TwoPPct,
    round((tltpp.AST / tltpp.FGA),3) as eASTPct
    from tb_leagues_team_advanced tlta left join tb_leagues_team_per100poss tltpp on tltpp.is_opponent =0 and tlta.is_playoff = tltpp.is_playoff and tlta.season = tltpp.season and tlta.team =tltpp.team  
    left join tb_leagues_team_total tltt on tltt.is_opponent =0 and tltt.is_playoff = tltpp.is_playoff and tltpp.season = tltt.season and tltpp.team =tltt.team 
    left join tb_leagues_info tlif on tlif.season = tltpp.season and tltpp.team = tlif.champion_team 
    """

    cursor.execute(query_sql)
    raw_data = cursor.fetchall()
    dtype_spec = {
    'ORtg': 'float64',
    'DRtg': 'float64', 
    'NRtg': 'float64',
    'FTAr': 'float64',
    'TSPct': 'float64',
    'eFGPct': 'float64',
    'ThreePPct': 'float64',
    'TwoPPct': 'float64',
    'eASTPct': 'float64',
    'TOVPct': 'float64',
    'ORBPct': 'float64',
    'FT_Rate': 'float64',
    'opp_eFGPct': 'float64',
    'opp_TOVPct': 'float64',
    'DRBPct': 'float64',
    'opp_FT_Rate': 'float64',
    'Wins': 'int64',
    'Losses': 'int64',
    'season': 'int64'  # 或者其他合适的类型
    }


    df = pd.DataFrame(raw_data)
    df = df.astype(dtype_spec)
    
    df['is_playoff'] = df['is_playoff'].apply(lambda x: 'Yes' if x==1 else 'No')

    file_path = os.path.join(RAW_DATA_PATH,'teams advanced.csv')
    save_df = df.round(2)
    save_df.to_csv(file_path, sep=',', index=False)
    return save_df




def get_league_standard(raw_df: pd.DataFrame):
    """计算
    各赛季联盟的平均值得出标杆表作比较
    """
    # 联盟标杆表
    raw_copy = raw_df.copy(deep=True)
    raw_copy['WinPct'] = raw_copy['Wins'] / (raw_copy['Wins'] + raw_copy['Losses'])
    league_std_df = raw_copy.groupby(['season','is_playoff']).apply(lambda g: pd.Series({
        'Corr ORtg Wins': g['ORtg'].corr(g['Wins']),
        'Corr DRtg Wins': g['DRtg'].corr(g['Wins']),
        'Corr NRtg Wins': g['NRtg'].corr(g['Wins']),
        'ORtg Avg': g['ORtg'].mean(),
        'DRtg Avg': g['DRtg'].mean(),
        'NRtg Avg': g['NRtg'].mean(),
        'ThreePPct Avg': g['ThreePPct'].mean(),
        'TwoPPct Avg': g['TwoPPct'].mean(),
        'eASTPct Avg': g['eASTPct'].mean(),
        'TOVPct Avg': g['TOVPct'].mean(),
        'WinPct Avg': g['WinPct'].mean()
        
    }), include_groups=False).reset_index()
    
    

    playoff = league_std_df[league_std_df['is_playoff'] == 'Yes'].set_index('season')
    regular = league_std_df[league_std_df['is_playoff'] == 'No'].set_index('season')
    delta_corr = playoff.copy()
    delta_corr['delta Offense'] = delta_corr['Corr ORtg Wins'] - regular['Corr ORtg Wins']
    delta_corr['delta Defense'] = np.abs(delta_corr['Corr DRtg Wins']) - np.abs(regular['Corr DRtg Wins'])
    delta_corr = delta_corr[['delta Offense', 'delta Defense']].reset_index()
    
    file_path = os.path.join(RAW_DATA_PATH,'league standard.csv')
    
    league_std_df.round(2).to_csv(file_path,sep=",", index=False)
    
    delta_corr.round(2).to_csv(os.path.join(RAW_DATA_PATH,'league delta corr.csv'), index=False)

    

    return league_std_df


def calculate_correlations(group):
    wins_corrs = group[['ORtg', 'DRtg', 'NRtg']].corrwith(group['Wins'])
    return wins_corrs



def output_team_league(team_adv_df: pd.DataFrame, league_std_df:pd.DataFrame):
    # 将各球队指标与联盟平均值对比, 得出差值
    # merger_df = team_adv_df.merge(league_std_df, on=['season', 'is_playoff'], how='left')
    team_grouped = team_adv_df.groupby(['season', 'is_playoff'])

    

    # 合并联盟均值与相关系数排名(得出所有赛季所有球队)
    merger_df = (team_adv_df.reset_index(drop=True).merge(league_std_df.reset_index(drop=True), on=['season', 'is_playoff'], how='left'))
    
    # 排名(只与当赛季有关) 对齐行
    merger_df['Rank ORtg'] = team_grouped['ORtg'].rank(method='dense', ascending=False,).astype('int')
    merger_df['Rank DRtg'] = team_grouped['DRtg'].rank(method='dense',).astype('int')
    merger_df['Rank NRtg'] = team_grouped['NRtg'].rank(method='dense', ascending=False).astype('int')
    merger_df['Rank ThreePPct'] = team_grouped['ThreePPct'].rank(method='dense', ascending=False).astype('int')
    merger_df['Rank TwoPPct'] = team_grouped['TwoPPct'].rank(method='dense', ascending=False).astype('int')
    merger_df['WinPct'] = merger_df['Wins'] / (merger_df['Wins'] + merger_df['Losses'])
   

    # 与联盟平均差值，能够消除跨赛季强度不一
    merger_df['ORtg vs League'] = merger_df['ORtg'].astype('float64') - merger_df['ORtg Avg'].astype('float64')
    merger_df['DRtg vs League'] = merger_df['DRtg'].astype('float64') - merger_df['DRtg Avg'].astype('float64') # 失分：越少越好
    merger_df['NRtg vs League'] = merger_df['NRtg'].astype('float64') - merger_df['NRtg Avg'].astype('float64')
    merger_df['ThreePPct vs League'] = merger_df['ThreePPct'].astype('float64') - merger_df['ThreePPct Avg'].astype('float64')
    merger_df['TwoPPct vs League'] = merger_df['TwoPPct'].astype('float64') - merger_df['TwoPPct Avg'].astype('float64')
    merger_df['TOVPct vs League'] = merger_df['TOVPct'].astype('float64') - merger_df['TOVPct Avg'].astype('float64')
    merger_df['WinPct vs League'] = merger_df['WinPct'].astype('float64') - merger_df['WinPct Avg'].astype('float64')
    
    # 计算Z-SCORE(WinPct，NRtg) Z-SCORE = xi - x_avg / x.std(标准差)
    # 计算统治力Dom_Z = (WinPct_Z + NRtg_Z) / 2
    # 使用NRtg计算Z-SCORE会不会造成由于不同赛季的强度，导致偏差，这样对比较统治力球队有偏颇(统治力球队是当赛季的统治力，还是整个nba历史中比较)
    # 由于找极具统治力的球队需要跨赛季比较：使用相对联盟胜率与相对联盟净效率 消除跨赛季强度影响偏差
    merger_df['NRtg_Z'] = (merger_df['NRtg vs League'] - merger_df['NRtg vs League'].mean()) / merger_df['NRtg vs League'].std()
    merger_df['WinPct_Z'] = (merger_df['WinPct vs League'] - merger_df['WinPct vs League'].mean()) / merger_df['WinPct vs League'].std()
    # 方法1 根据条件赋值新列.loc
    # champions['Dom_Z'] = (champions['WinPct_Z'] + champions['NRtg_Z'])/2
    # champions.loc[champions['is_playoff'] == 'Yes', 'Dom_Z'] =  champions['NRtg_Z']

    merger_df['Dom_Z'] = np.where(merger_df['is_playoff'] == 'Yes', merger_df['NRtg_Z'], (merger_df['WinPct_Z'] + merger_df['NRtg_Z'])/2)

    merger_df = merger_df.round(2)
    champions = merger_df[merger_df['champion'] == 'Yes'].copy()

    # df_summary = pd.DataFrame()

    # 计算冠军类型比例(季后赛)
    total = (champions['is_playoff'] == 'No').sum()
    ortg_better = ((champions['Rank ORtg'] < champions['Rank DRtg']) & (champions['is_playoff'] == 'No')).sum()
    drtg_better = ((champions['Rank ORtg'] > champions['Rank DRtg']) & (champions['is_playoff'] == 'No')).sum()
    nrtg_no1 = ((champions['Rank NRtg'] <= 3) & (champions['is_playoff'] == 'No')).sum()
    diff3 = ((pd.Series(champions['Rank ORtg'] - champions['Rank DRtg']).abs() <= 3) & (champions['is_playoff'] == 'No')).sum()
    diff5 = ((pd.Series(champions['Rank ORtg'] - champions['Rank DRtg']).abs() <= 5) & (champions['is_playoff'] == 'No')).sum()
    # 计算冠军类型比例(常规赛)
    
    
    # df_result = champions[diff3].copy()
    # df_result['Playoff2'] = 'Rank ORtg DRtg diff 3'
    print(f"NRtg No.1- No.3 {nrtg_no1} {round(nrtg_no1*100/total, 2)}%")
    print(f"ortg better: {ortg_better} {round(ortg_better*100/total, 2)}%")
    print(f"drtg better: {drtg_better} {round(drtg_better*100/total, 2)}%")
    print(f"drtg diff ortg in 3: {diff3} {round(diff3*100/total, 2)}%")
    print(f"drtg diff ortg in 5: {diff5} {round(diff5*100/total, 2)}%")
    
    df_summary = champions.groupby('is_playoff').apply(lambda g: pd.Series({
        'ORtg Better': round(100*( g['Rank ORtg'] < g['Rank DRtg']).sum()/total,2),
        'DRtg Better': round(100*( g['Rank ORtg'] > g['Rank DRtg']).sum()/total,2),
        'ORtf diff DRtg <= 3': round( 100*( (g['Rank ORtg'] - g['Rank DRtg'] ) <= 3).sum()/total,2),
        'NRtg No.1~3': round(100*( g['Rank NRtg'] <= 3).sum()/total,2),
        'DRtg or ORtg No.1~3': round(100*( (g['Rank ORtg'] <= 3) | (g['Rank DRtg'] <= 3) ).sum()/total,2),
    }), include_groups=False).reset_index()
    print(df_summary)

    file_path = os.path.join(PROCESS_DATA_PATH, 'League Champions.csv')
    file_path_all = os.path.join(PROCESS_DATA_PATH, 'League All Teams.csv')
    champions.to_csv(file_path, index=False)
    merger_df.to_csv(file_path_all, index=False)

    


def main():
    team_adv_df = get_leageues_team_adv(DB_INFO)
    # print(team_adv_df.dtypes)
    # print(team_adv_df.head(10))
    league_std_df = get_league_standard(team_adv_df)
    output_team_league(team_adv_df, league_std_df)
    # # df = pd.DataFrame(data=np.array([[1.2324,2,3,4],[15,6,7,89]]), columns=['A', 'B', 'C', 'D'],)
    # # df['X'] = df['D'] - df['D'].mean()
    # # print(df)
    # # df2 = df.round(2)
    # # print(df2)
    # # df.to_csv('A.csv')
    # # df2.to_csv('B.csv')


def demonstrate_simpsons_paradox():
    # 组内负相关，但整体正相关的极端例子
    group1 = pd.DataFrame({
        'DRtg': np.linspace(100, 110, 20),
        'Wins': np.linspace(60, 30, 20)  # 强队：负相关
    })
    group1['Group'] = '强队'
    
    group2 = pd.DataFrame({
        'DRtg': np.linspace(110, 100, 20), 
        'Wins': np.linspace(35, 5, 20)    # 弱队：负相关
    })
    group2['Group'] = '弱队'
    
    combined = pd.concat([group1, group2])
    
    print("组内相关系数:")
    print(f"强队: {group1['DRtg'].corr(group1['Wins']):.3f}")
    print(f"弱队: {group2['DRtg'].corr(group2['Wins']):.3f}")
    print(f"整体: {combined['DRtg'].corr(combined['Wins']):.3f}")



if __name__ == '__main__':
    main()
    # demonstrate_simpsons_paradox()