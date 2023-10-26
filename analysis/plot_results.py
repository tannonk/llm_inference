from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import seaborn as sns
from seaborn.objects import Text

sns.set_theme(context='paper', style='white', palette='colorblind')
# no ground_truth
MODEL_ORDER = ['openai-gpt-3.5-turbo', 'openai-text-davinci-003', 'openai-text-davinci-002', 'flan-ul2',
               'flan-t5-large', 'opt-iml-max-30b', 'flan-t5-xxl', 'flan-t5-base', 'flan-t5-xl', 'llama-7b',
               'llama-13b', 'openai-text-curie-001', 'bloom', 'opt-66b', 'llama-30b', 'gpt-neox-20b', 'gpt-j-6b',
               'opt-13b', 'opt-6.7b', 'llama-65b', 'flan-t5-small', 'openai-text-babbage-001', 'opt-30b', 'bloomz',
               'bloom-3b', 'bloomz-7b1', 'bloom-7b1', 't0', 'bloom-560m', 'opt-iml-max-1.3b', 'bloomz-3b', 'ul2',
               't0-3b', 'bloomz-560m', 't0pp', 'bloomz-1b1', 't5-base-lm-adapt', 'bloom-1b1', 'openai-text-ada-001',
               't5-small-lm-adapt', 't5-xxl-lm-adapt', 'opt-1.3b', 't5-large-lm-adapt', 't5-xl-lm-adapt']
                #'muss_en_wikilarge_mined', 'muss_en_mined']


def scatter_3d(df, metric_1, metric_2, metric_3, label_name, save_name=None, dataset='asset-test'):
    """
    :param df: dataframe
    :param metric_1: col name
    :param metric_2: col name
    :param metric_3: col name
    :param label_name: col name
    :param save_name:
    :param dataset: test dataset
    example: scatter_3d(df=res_df, metric_1='fkgl', metric_2='LENS', metric_3='F1_bert_ref',
                        label_name='Model', save_name='fkgl_LENS_F1_bert_ref_3d_all.png')
    """
    df = df[df['Test'] == dataset]
    sns.set_style('white')
    sns.set_palette('colorblind')
    cdict = {'p0': 0, 'p1': 1, 'p2': 2}
    color_vals = [cdict[c] for c in df['Prompt']]
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    cmap = ListedColormap(['#0173b2', '#de8f05', '#029e73'])

    scat = ax.scatter3D(df[metric_1], df[metric_2], df[metric_3], c=color_vals, cmap=cmap)
    ax.set_xlabel(metric_1)
    ax.set_ylabel(metric_2)
    ax.set_zlabel(metric_3)

    if metric_2 == 'fkgl':
        ax.invert_yaxis()
    if metric_1 == 'fkgl':
        ax.invert_xaxis()
    if metric_3 == 'fkgl':
        ax.invert_zaxis()
    for i, txt in enumerate(df[label_name]):
        ax.text(df[metric_1].tolist()[i], df[metric_2].tolist()[i], df[metric_3].tolist()[i], txt, fontsize=7)
    plt.legend(*scat.legend_elements(), title='Prompt')
    if save_name:
        plt.savefig(f'analysis/visualizations/{save_name}')
    else:
        plt.show()


def scatter_two_metrics_by_label(df, metric_1, metric_2, label_name, save_name=None, dataset='asset-test'):
    """
    Create 2d scatterplot with two metrics. Add prompt legend
    :param df: dataframe
    :param metric_1: col name
    :param metric_2: col name
    :param label_name: col name
    :param save_name:
    :param dataset: test dataset
    example: scatter_two_metrics_by_label(df=res_df, metric_1='LENS', metric_2='F1_bert_ref', label_name='Model')
    """
    plt.rcParams.update({'axes.titlesize': 'large',
                         'axes.labelsize':'large',
                         'ytick.labelsize': 'large',
                         'xtick.labelsize': 'large',
                         'legend.title_fontsize': 'large',
                         'legend.fontsize': 'large'})
    df = df[df['Test'] == dataset]
    ax = sns.scatterplot(data=df, x=metric_1, y=metric_2, hue='Prompt',
                         hue_order=['p0', 'p1', 'p2'])
    plt.xlabel(metric_1)
    plt.ylabel(metric_2)
    ax.set_xlim(right=50)
    plt.legend(loc='upper left')

    if metric_2 == 'fkgl':
        ax.invert_yaxis()
    if metric_1 == 'fkgl':
        ax.invert_xaxis()
    for i, txt in enumerate(df[label_name]):
        ax.text(df[metric_1].tolist()[i], df[metric_2].tolist()[i], txt, fontsize=9)
    if save_name:
        plt.savefig(f'analysis/visualizations/{save_name}')
    else:
        plt.show()


def scatter_all_pairs(df, save_name=None, annotate=False, name_cutoff=False, dataset='asset-test'):
    """
    Create 2d scatterplots. Add prompt legend
    :param df: dataframe
    :param save_name:
    :param annotate:
    :param name_cutoff: whether to display full model name
    :param dataset: test dataset
    """
    plt.rcParams.update({'axes.titlesize': 'large',
                         'axes.labelsize':'large',
                         'ytick.labelsize': 'large',
                         'xtick.labelsize': 'large',
                         'legend.title_fontsize': 'large',
                         'legend.fontsize': 'large'})
    # sns.set(font_scale=1.5)

    df = df[(df['Test'] == dataset) & (df['Model'] != 'cohere-command-light') & ('muss' not in df['Model'])
            & (df['Model'] != 'ground_truth')]
    grid = sns.pairplot(data=df, hue='Prompt', hue_order=['p0', 'p1', 'p2'],
                        vars=['sari', 'fkgl', 'F1_bert_ref', 'LENS'], kind='scatter')
    # grid.add_legend(fontsize='large')
    # plt.legend(fontsize='large')
    for ax in grid.axes.flatten():
        x, y = ax.get_xlabel(), ax.get_ylabel()
        if x or y:
            if not y:
                y = x
            if not x:
                x = y
            m1_max, m1_min = max(df[x]), min(df[x])  # because annotations take space
            fkgl_pad = 0.5
            other_pad = 2
            if x != 'fkgl':
                x_min = m1_min - other_pad if m1_min >= other_pad else 0
                x_max = m1_max + other_pad
            else:
                x_min = m1_min - fkgl_pad if m1_min >= fkgl_pad else 0
                x_max = m1_max + fkgl_pad
            ax.set_xlim(left=x_min, right=x_max)
            if annotate and x != y:
                for i, txt in enumerate(df['Model']):
                    if name_cutoff:
                        txt = txt.split('-')[0]

                    ax.text(df[x].tolist()[i], df[y].tolist()[i], txt, fontsize=9)
    grid.tight_layout(pad=2)
    if save_name:
        plt.savefig(f'analysis/visualizations/{save_name}')
    else:
        plt.show()


def prepare_facet_grid(df, sharex=True, sharey=True, height=4, use_model_prefix=False, metrics=None):
    """
    Utility function for creating a grid that has a plot per metric.
    :param df:
    :param sharex:
    :param sharey:
    :param height:
    :param use_model_prefix: to decrease number of models shown
    :param metrics: metrics to plot
    :return:
    """
    rel_data = {'Model': [],
                'Metric': [],
                'Value': [],
                'Prompt': []}
    metrics_caps = [met.upper() for met in metrics]
    for i in range(len(df)):
        for j in range(len(metrics)):
            if not use_model_prefix:
                rel_data['Model'].append(df['Model'].tolist()[i])
            else:
                # arbitrary categories, think about splitting by size instead
                rel_data['Model'].append(df['Model'].tolist()[i].split('-')[0])
            rel_data['Prompt'].append(df['Prompt'].tolist()[i])
        rel_data['Metric'].extend(metrics_caps)
        for met in metrics:
            rel_data['Value'].append(df[met].tolist()[i])
    rel_data = pd.DataFrame(rel_data)
    c_wrap = len(metrics) // 2
    g = sns.FacetGrid(rel_data, col="Metric", col_wrap=c_wrap, aspect=2, sharex=sharex, sharey=sharey, height=height)
    return g, rel_data


def box_facet_grid(df, yval='Model', save_name=None, sharex=False, sharey=True, plot_type=sns.boxplot,
                   use_model_prefix=False, dataset='asset-test', model_y=True, metrics=None):
    """
    This function visualizes [model OR prompt] (yval) with each metric (xval)
    :param df: data
    :param yval: 'Model' or 'Prompt'
    :param save_name:
    :param sharex:
    :param sharey:
    :param plot_type: can also work with violinplot (if yval is 'Prompt')
    :param use_model_prefix: if so, we can display a boxplot per plot per model prefix (otherwise it's too much data)
    :param dataset: test dataset
    example: box_facet_grid(df=res_df, yval='Prompt', plot_type=sns.violinplot)
    """
    plt.rcParams.update({'axes.titlesize': 20,
                         'axes.labelsize': 20,
                         'ytick.labelsize': 20,
                         'xtick.labelsize': 20})
    df = df[df['Test'] == dataset]
    if yval == 'Model':
        # NOTE: added the below later, the old plots average all prompts
        df = df[(df['Prompt'] == 'p2') & (df['Model'] != 'cohere-command-light')]
        height = 6
    else:
        df = df[(df['Model'] != 'cohere-command-light') & (df['Model'] != 'ground_truth')]
        height = 4
    order = MODEL_ORDER if yval == 'Model' else ['p0', 'p1', 'p2']
    grid, rel_data = prepare_facet_grid(df=df, sharex=sharex, sharey=sharey, height=height,
                                        use_model_prefix=use_model_prefix, metrics=metrics)
    grid.set_axis_labels("", yval)
    if use_model_prefix:  # use hues
        grid.map(plot_type, 'Value', yval, 'Prompt', order=order, hue_order=['p0', 'p1', 'p2'],
                 palette="colorblind")
    else:  # don't use hues
        if model_y:
            grid.map(plot_type, 'Value', yval, order=order, palette="colorblind")
        else:
            grid.map(plot_type, yval, 'Value', order=order, palette="colorblind")
            grid.set_axis_labels(yval, '')
    grid.set_titles("{col_name}")
    if yval == 'Model' and not use_model_prefix:
        if model_y:
            grid.set_yticklabels(MODEL_ORDER, fontsize=14,
                                 rotation_mode='anchor')
        else:
            grid.set_xticklabels(MODEL_ORDER, rotation=45, fontsize=14, horizontalalignment='right',
                                 rotation_mode='anchor')
    grid.add_legend()
    grid.tight_layout(pad=2)
    if save_name:
        plt.savefig(f'analysis/visualizations/{save_name}', bbox_inches='tight')
    else:
        plt.show()


def barplot_facet_grid(df, metrics, save_name=None, sharex=True, sharey=True, ave_prompts=True,
                       dataset='asset-test', use_ground_truth=False):
    """
    Create barplot per metric with prompt legend and model as x
    :param df: dataframe
    :param metrics: list of metrics to plot
    :param save_name:
    :param sharex:
    :param sharey:
    :param ave_prompts:
    :param dataset: test dataset
    :param use_ground_truth: whether we are using the ground truth value
    example: barplot_facet_grid(df=res_df, sharex=True, sharey=False)
    """
    df = df[df['Test'] == dataset]
    if not use_ground_truth:
        df = df[df['Model'] != 'ground_truth']
    grid, rel_data = prepare_facet_grid(df=df, metrics=metrics, sharex=sharex, sharey=sharey)
    if not ave_prompts:
        grid.map(sns.barplot, "Model", "Value", "Prompt", hue_order=['p0', 'p1', 'p2'], palette="colorblind", alpha=0.7)
        grid.add_legend()
    else:
        grid.map(sns.barplot, "Model", "Value", palette="colorblind", alpha=0.7)
    grid.set_titles("{col_name}")
    grid.set_axis_labels("", "")
    grid.tight_layout(pad=2)
    one_prompt_models = rel_data['Model'].unique()   # so we don't have repeat models
    grid.set_xticklabels(one_prompt_models, rotation=45, horizontalalignment='right', rotation_mode='anchor')

    if use_ground_truth:
        xticklabels = grid.axes[-1].get_xticklabels()
        ground_truth_ind = None
        for j in range(len(xticklabels)):
            if xticklabels[j].get_text() == 'ground_truth':
                ground_truth_ind = j

        for i in range(len(grid.axes)):
            ax = grid.axes[i]
            ax.patches[ground_truth_ind].set_color('black')
            ax.patches[ground_truth_ind].set_hatch('x')
            ax.patches[ground_truth_ind].set_fill(False)
    grid.fig.subplots_adjust(bottom=0.1)

    if save_name:
        plt.savefig(f'analysis/visualizations/{save_name}')
    else:
        plt.show()


if __name__ == '__main__':
    main_metrics = ['sari', 'fkgl', 'F1_bert_ref', 'LENS']
    all_quality_metrics = ['copies', 'adds', 'deletes', 'splits', 'c_ratio', 'lex_complexity', 'lev_sim']
    all_referenceless_metrics = ['copies', 'adds', 'deletes', 'splits', 'c_ratio', 'lex_complexity', 'lev_sim',
                                 'ppl_mean', 'fkgl']
    non_qe_referenceless_metrics = ['ppl_mean', 'fkgl']
    av_seed_full_path = "reports/full/full_results_by_sari.csv"  # can use this for anything
    raw_path = "reports/raw/raw_results.csv"  # 3 seeds, can use for boxplots/violins
    av_seed_full_df = pd.read_csv(av_seed_full_path)
    raw_df = pd.read_csv(raw_path)
    # sorted_sari_inds = np.argsort(a=-saris)
    # scatter_two_metrics_by_label(df=av_seed_full_df, metric_1='sari',
    #                              metric_2='F1_bert_ref', label_name='Model', save_name=None, dataset='asset-test')
    # print(sns.color_palette("colorblind").as_hex())
    # scatter_all_pairs(av_seed_full_df, annotate=False, name_cutoff=False, dataset='news-manual-all-test',
    #                   save_name='news-manual-all-test/scatter_all_metric_pairs_no_annotation.png')

    # scatter_3d(df=av_seed_full_df, metric_1='sari', metric_2='fkgl', metric_3='F1_bert_ref',
    #            dataset='news-manual-all-test',
    #            label_name='Model',
    #            save_name='news-manual-all-test/sari_fkgl_F1_bert_ref_3d_all.png')
    #
    # barplot_facet_grid(df=raw_df, metrics=non_qe_referenceless_metrics, sharex=True, sharey=False,
    #                    ave_prompts=True, dataset='news-manual-all-test', use_ground_truth=True,
    #                    save_name='news-manual-all-test/model_barplots_allseeds_allprompts_nonqe_referenceless.png')

    # box_facet_grid(df=raw_df, yval='Prompt', sharex=False, sharey=True, plot_type=sns.violinplot,
    #                use_model_prefix=False, dataset='news-manual-all-test', model_y=True, metrics=main_metrics)
                   # save_name='news-manual-all-test/prompt_violinplot_per_metric_diffscale.png')

    box_facet_grid(df=raw_df, yval='Model', sharex=True, sharey=False, plot_type=sns.boxplot,
                   use_model_prefix=False, dataset='med-easi-test', model_y=False, metrics=main_metrics,
                   save_name='med-easi-test/model_boxplot_per_metric_p2_diffscale.png')

