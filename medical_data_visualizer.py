import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv') #  index_col=['id'] 

# 2
df['overweight'] = 0 # All values to zero by defualt
df.loc[df['weight'] / np.square(df['height']/100) > 25, 'overweight'] = 1 # set 1 on those who match the overweight condition

# 3
df['cholesterol'] = [0 if chol == 1 else 1 for chol in df['cholesterol']]
df['gluc'] = [0 if gluc == 1 else 1 for gluc in df['gluc']]

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol','gluc','smoke','alco','active','overweight'])
    # 'cardio' as id_var will be useful for grouping data

    # 6
    df_cat = df_cat.groupby('cardio')[['variable','value']].value_counts(sort=False).to_frame()
    # Grouping by 'cardio', remaining variable and value columns, then value_counts and finally converting to a df
    # Not sorting for getting identical charts
    # Resulting count column is named 0
    df_cat.rename(columns={0: 'total'}, inplace=True) # Rename resulting column and modifying the original df 

    # 7
    bar_plot = sns.catplot(x='variable', y='total', data=df_cat,
            kind='bar',
            col='cardio',
            hue='value') # output of catplot

    # 8
    fig = bar_plot.figure # Getting the figure of the output
    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    mask1 = df['ap_lo'] <= df['ap_hi']
    mask2 = df['height'] >= df['height'].quantile(0.025)
    mask3 = df['height'] <= df['height'].quantile(0.975)
    mask4 = df['weight'] >= df['weight'].quantile(0.025)
    mask5 = df['weight'] <= df['weight'].quantile(0.975)
    correct_data_mask = (mask1) & (mask2) & (mask3) & (mask4) & (mask5)
    df_heat = df.loc[correct_data_mask]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.full(corr.shape, False) # full boolean matrix of False values.
    rows = mask.shape[0]
    for i in range(rows):
        mask[i,i:] = True # In each row, from diagonal position to end


    # 14
    fig, ax = plt.subplots(figsize=(12,9), dpi=300)
    ax.set_title('Correlation Heatmap')

    # 15
    sns.heatmap(corr, mask=mask, vmin=-0.1, vmax=0.25,
            cmap='RdBu', # From Red to Blue
            linewidths=.5,
            linecolor='white',
            annot=True, # Show cell values
            fmt=".1f", # Round cell values to 1 decimal
            annot_kws={
                'fontsize': 8 
            })


    # 16
    fig.savefig('heatmap.png')
    return fig
