import pandas as pd

p = "D:\\Download\\archive (1)\\csv\\play_by_play.csv"
p2 = "D:\\Download\\archive (1)\\csv\\play_by_play2.csv"

df = pd.read_csv(p)
df['scoremargin'] = df['scoremargin'].apply(lambda x: '' if x=='TIE' else x)
df.to_csv(p2)