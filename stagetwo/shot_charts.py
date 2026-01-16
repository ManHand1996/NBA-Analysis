"""球员投篮对比图
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nba_api.stats.endpoints import ShotChartDetail
from nba_api.stats.static import players
from nba_api.stats.static import teams

from matplotlib.patches import Circle, Rectangle, Arc

# NBA_ZONES = {
#     'Restricted Area': (0,4),
#     'Paint (Non-RA)': (4,8),
#     'Mid-Range': (8,22),
#     'Corner 3': (22,23.75),
#     'Three': (23.75, 99)
# }
BASIC_ZONES = ['Above the Break 3','Mid-Range', 'In The Paint (Non-RA)', 
               'Restricted Area', 'Right Corner 3', 'Left Corner 3','Backcourt']

CN_BASIC_ZONES = {
    'Above the Break 3': '三分',
      'Mid-Range': '中距离',
    'In The Paint (Non-RA)': '油漆区',
     'Restricted Area': '进攻有理区',
    'Right Corner 3': '三分',
    'Left Corner 3': '三分',
    'Backcourt': '三分'
}

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def get_shot_df(player_name = 'Stephen Curry', full_team = 'Golden State Warriors', season='2022-23'):
    # 1. 获取球员ID
    player_dict = players.get_players()
    stephen = [player for player in player_dict if player['full_name'] == player_name][0]
    player_id = stephen['id']

    # 2. 获取球队ID（可选，但推荐）
    warriors = [team for team in teams.get_teams() if team['full_name'] == full_team][0]
    team_id = warriors['id']

    # 3. 获取投篮图数据（示例：2022-23赛季常规赛）
    shotchart = ShotChartDetail(
        team_id=team_id,
        player_id=player_id,
        context_measure_simple='FGA',  # 投篮出手
        season_nullable=season,
        season_type_all_star='Regular Season'
    )

    return shotchart.get_data_frames()[0]

# 4. 转换为DataFrame
# shot_df = shotchart.get_data_frames()[0]

# made = shot_df[shot_df['SHOT_MADE_FLAG'] == 1]
# missed = shot_df[shot_df['SHOT_MADE_FLAG'] == 0]



def draw_court(ax=None, color='black', lw=2, outer_lines=False):

    """
    参考, 调整了底线三分长度140->137, 更接近实际
    http://savvastjortjoglou.com/nba-shot-sharts.html
    """

    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color='GREEN', fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 137, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 137, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    # get rid of axis labels and tick marks
    

    return ax


def plot_shot_comparison(shots_data_list, players_names, colors=None):
    """
    绘制多个球员的投篮对比图
    
    参数:
    shots_data_list: 投篮数据列表（每个元素是一个球员的投篮数据）
    players_names: 球员名称列表
    colors: 每个球员的颜色列表（可选）
    """
    if colors is None:
        # 默认颜色方案
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 创建子图
    n_players = len(shots_data_list)
    fig, axes = plt.subplots(1, n_players, figsize=(6*n_players, 9))
    
    if n_players == 1:
        axes = [axes]
    
    # 为每个球员绘制投篮图
    for idx, (shots_data, player_name, ax) in enumerate(zip(shots_data_list, players_names, axes)):
        # 绘制半场
        ax.set_xlim(-300,300)
        ax.set_ylim(-100,500)
        ax.axis('off') 
        
        draw_court(ax,outer_lines=True)
        
        # 分离命中和未命中
        made_shots = shots_data[shots_data['SHOT_MADE_FLAG'] == 1]
        missed_shots = shots_data[shots_data['SHOT_MADE_FLAG'] == 0]

        # 绘制未命中的投篮（空心圆）
        
        ax.scatter(missed_shots['LOC_X'],
                     missed_shots['LOC_Y'],
                    #   facecolors='none', edgecolors=colors[idx],
                    color='red',
                      s=50, alpha=0.6, label='Missed', linewidths=.5, marker='x')
        
        # 绘制命中的投篮（实心圆）
        
        ax.scatter(made_shots['LOC_X'],
                      made_shots['LOC_Y'],
                      color=colors[idx], s=80, alpha=0.8, 
                      label='Made', edgecolors='white', linewidths=1)
        
        # 计算并显示命中率
        total_shots = len(shots_data)
        made_count = len(made_shots)
        fg_percentage = made_count / total_shots * 100 if total_shots > 0 else 0
        
        
        detail_str = ''
        threes = ('Above the Break 3', 'Right Corner 3', 'Left Corner 3', 'Backcourt')
        
        threes_all =  shots_data[shots_data['SHOT_ZONE_BASIC'].isin(threes)]
        thress_total = len(threes_all)
        threes_made =  len(threes_all[threes_all['SHOT_MADE_FLAG'] == 1])
        threes_pct = threes_made/thress_total * 100 if thress_total > 0 else 0

        detail_str += f'三分球:({threes_made}/{thress_total}){threes_pct:.1f}%\n'
        for zone in BASIC_ZONES:
            if zone not in threes:
                zone_all =  shots_data[shots_data['SHOT_ZONE_BASIC'] == zone]
                zone_total = len(zone_all)
                zone_made = len(zone_all[zone_all['SHOT_MADE_FLAG'] == 1])
                zone_pct = zone_made/zone_total * 100 if zone_total > 0 else 0
                
                detail_str += f'{CN_BASIC_ZONES[zone]}:({zone_made}/{zone_total}){zone_pct:.1f}%\n'

        # 添加标题和统计信息
        ax.set_title(f'{player_name}\n{made_count}/{total_shots} ({fg_percentage:.1f}%)\n{detail_str}',
                    fontsize=14, fontweight='bold', pad=20)
        
        # 添加图例
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                 ncol=2, fontsize=10, frameon=False)
    
    plt.tight_layout()
    return fig, axes


def quick_shot_chart(shot_df):
    """快速创建投篮图"""
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # 使用seaborn的联合分布图
    g = sns.jointplot(
        data=shot_df,
        x='LOC_X',
        y='LOC_Y',
        kind='hex',  # 六边形热图
        cmap='Blues',
        height=9,
        ratio=5,
        space=0
    )
    
    # 简单绘制篮筐
    g.ax_joint.add_patch(plt.Circle((0, 0), radius=0.625, color='red', fill=False, lw=2))
    
    # 绘制简单三分线
    theta = np.linspace(0, np.pi, 100)
    x_three = 23.75 * np.cos(theta)
    y_three = 23.75 * np.sin(theta)
    g.ax_joint.plot(x_three, y_three, 'black', lw=2)
    
    return g


if __name__ == '__main__':
    # data
    stephencurry: pd.DataFrame = get_shot_df('Stephen Curry', full_team='Golden State Warriors', season='2023-24')
    # print(stephencurry['SHOT_ZONE_BASIC'].unique())
    # print(stephencurry['SHOT_ZONE_AREA'].unique())

    kyrieirving = get_shot_df('Kyrie Irving', full_team='Dallas Mavericks', season='2023-24')
    fig, axs =plot_shot_comparison(
        shots_data_list=[stephencurry, kyrieirving],
        players_names=["Stephen Curry", "Kyrie Irving"],
        colors=['#FFC107', '#1E88E5']  # 金色和蓝色
    )
    fig.suptitle('2023-24 赛季投篮对比', 
                 fontsize=16, fontweight='bold', y=0.95)
    # quick_shot_chart(stephencurry)
    plt.savefig('shot_compare curry vs kyrie.png')
    plt.show()