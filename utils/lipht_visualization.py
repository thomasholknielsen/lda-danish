def word_frequency_barplot(df, nr_top_words=50, title=None, directory=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    cur_dir = os.getcwd()
    
    """ df should have a column named count.
    """
    fig, ax = plt.subplots(1,1,figsize=(20,10))

    sns.barplot(list(range(nr_top_words)), df['count'].values[:nr_top_words], palette='hls', ax=ax)

    ax.set_xticks(list(range(nr_top_words)))
    ax.set_xticklabels(df.index[:nr_top_words], fontsize=14, rotation=90)
    if not title:
        title="Word Frequencies"
    ax.set_title(title, fontsize=16)
    fig.savefig(fname=os.path.join(cur_dir,'data','{}_wordfrequency.png'.format(directory)),transparent=True, bbox_inches="tight")
    return ax


def topic_distribution_barplot(lda_model, df, n_topbars=5, title=None, fname=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from utils.lipht_lda_utils import df_lda_features
    import os
    cur_dir = os.getcwd()

    # df_lda_features(lda_model, df)
    topic_distribution = df['lda_features'].mean()
    
    if not n_topbars:
        n_topbars = 5
    if not title:
        title = 'Topic distributions showing top {} bars of {} topics'.format(n_topbars, len(topic_distribution))
    if not fname:
        fname = title
    
    fig, ax1 = plt.subplots(1,1,figsize=(20,10))
    ax1.set_title(title, fontsize=16)

    for ax, distribution, color in zip([ax1], [topic_distribution], ['r']):
        # Individual distribution barplots
        ax.bar(range(len(distribution)), distribution, alpha=0.7)
        rects = ax.patches
        for i in np.argsort(distribution)[-n_topbars:]:
            rects[i].set_color(color)
            rects[i].set_alpha(1)

    fig.tight_layout(h_pad=3.)
    fig.savefig(fname=os.path.join(cur_dir,'data','{}.png'.format(fname)),transparent=True, bbox_inches="tight")
    return