import os
from transquest.algo.my_transformers.run_model import QuestModel
import torch
import argparse
import time
from scipy.stats import pearsonr, spearmanr
from multiprocessing import cpu_count
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

import sys
sys.path.append('/home/u2/proj/lexitra/hu_TransQuest/')

from transquest.algo.my_transformers.evaluation import pearson_corr, spearman_corr, rmse
from universal_utils import filter_sents_tsv, normalise_score_dist, stratify_continuous

min_max_scaler = preprocessing.MinMaxScaler()


def draw_scatterplot(data_frame, real_column, prediction_column, path, topic):
    data_frame = data_frame.sort_values(real_column)
    sort_id = list(range(0, len(data_frame.index)))
    data_frame['id'] = pd.Series(sort_id).values

    data_frame = fit(data_frame, real_column)
    data_frame = fit(data_frame, prediction_column)

    pearson_res = pearson_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    spearman_res = spearman_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    rmse_value = rmse(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    mae_value = mean_absolute_error(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())

    textstr = 'RMSE=%.4f\nMAE=%.4f\nPearson Correlation=%.4f\nSpearman Correlation=%.4f' % (rmse_value, mae_value,
                                                                                            pearson_res, spearman_res)

    plt.figure()
    ax = data_frame.plot(kind='scatter', x='segid', y=real_column, color='DarkBlue', label='true TQ score (mean)',
                         title=topic)
    ax = data_frame.plot(kind='scatter', x='segid', y=prediction_column, color='orange',
                         label='predicted TQ score (mean)', ax=ax)
    ax.text(0.5 * data_frame.shape[0],
            min(min(data_frame[real_column].tolist()), min(data_frame[prediction_column].tolist())), textstr,
            fontsize=10)
    plt.legend(loc='lower left')
    plt.xticks(rotation=45, ha='right')
    fig = ax.get_figure()
    fig.savefig(path)


def print_stat(data_frame, real_column, prediction_column):
    data_frame = data_frame.sort_values(real_column)

    pearson_res = pearson_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    spearman_res = spearman_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    rmse_value = rmse(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    mae_value = mean_absolute_error(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())

    textstr = 'RMSE=%.4f\nMAE=%.4f\nPearson Correlation=%.4f\nSpearman Correlation=%.4f' % (rmse_value, mae_value,
                                                                                            pearson_res, spearman_res)
    print(textstr)


def fit(df0, label):
    x = df0[[label]].values.astype(float)
    x_scaled = min_max_scaler.fit_transform(x)
    df_out = df0.copy()
    df_out.loc[:, label] = x_scaled
    return df_out


def un_fit(df0, label):
    x = df0[[label]].values.astype(float)
    x_unscaled = min_max_scaler.inverse_transform(x)
    df_out = df0.copy()
    df_out.loc[:, label] = x_unscaled
    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--data", help="Sents-scores mapping", default='tables/test_data.tsv')  # segid, text_a, text_b, label
    arg("--target", help="Choose: fluency, accuracy, tq, mean, scaled_tq", default='mean')
    arg("--total", default=9000, help="How many items to retain at filtering time")
    arg("--sentlen", default=60, help="Sentence length limit")
    arg("--nodes", help="Number of neurons aka memory cells", default=32)
    arg('--epochs', default=1, help="How many passes over the data?")
    arg('--stop', default=5, help='Do you want earlystopping? Pass an interger for the patience parameter or None')
    arg('--batch', default=8, help="How many instances to process at a time?")
    arg('--loss', default="mae", help="Options: mae, mse, mape")
    arg("--device", type=str, default='cpu', help='Choose the device. Options: cpu, gpu')
    arg("--out", type=str, default='hu_TransQuest/temp/outputs/', help='Path where to create an output dir')
    arg("--random", default=42, help='Fix train-test splitting')

    args = parser.parse_args()
    start = time.time()
    random = args.random
    target = args.target

    outdir = args.out
    os.makedirs(outdir, exist_ok=True)
    bestmodel = outdir + 'best_model'
    os.makedirs(bestmodel, exist_ok=True)
    cache = 'temp/cache_dir/'
    os.makedirs(cache, exist_ok=True)

    testmode = True

    # experimenting with using both cpu and 1 gpu available
    # if args.device == 'cpu':
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    #
    # if args.device == 'gpu':
    #     print('==GPUs available:', torch.cuda.device_count())

    df = pd.read_csv(args.data, sep='\t')

    if not testmode:
        # renaming my_table columns to meet the hu_TransQuest conventions
        df = df.rename({'sent_id': 'segid', 'ssent': 'text_a', 'tsent': 'text_b', target: 'labels'}, axis='columns')

        df1 = filter_sents_tsv(data=df, maxsent=60, score_col='labels', src_col='text_a', tgt_col='text_b',
                               lose_nan=True)
        print('==After filtering, inc. for sent length:', df1.shape)

        df2 = normalise_score_dist(data=df1, norming_col='tq', val=100.0, want_total=9000, save_column='da', random=42)

        ytarget = df2['labels'].values

        train_df0, eval_df0, test_df0, _, _, _ = stratify_continuous(data=df2, y=ytarget, bins_num=35, seed=random)

    else:
        # test dataframe has the columns with the right names
        train_df0, temp_df = train_test_split(df, test_size=0.2, random_state=random)
        eval_df0, test_df0 = train_test_split(temp_df, test_size=0.5, random_state=random)

    train_df = fit(train_df0, 'labels')
    eval_df = fit(eval_df0, 'labels')

    print('Train %s mean: %.2f, std: %.2f' % (train_df.shape, train_df['labels'].mean(), train_df['labels'].std()))
    print('Eval %s mean: %.2f, std: %.2f' % (eval_df.shape, eval_df['labels'].mean(), eval_df['labels'].std()))
    print('Test %s mean: %.2f, std: %.2f' % (test_df0.shape, test_df0['labels'].mean(), test_df0['labels'].std()))

    SEED = 42

    if args.device == 'cpu':
        transformer_config = {
            'output_dir': outdir,
            "best_model_dir": bestmodel,
            'cache_dir': cache,

            'fp16': False,
            'fp16_opt_level': 'O1',
            'max_seq_length': args.sentlen,
            'train_batch_size': args.batch,
            'gradient_accumulation_steps': 1,
            'eval_batch_size': args.batch,
            'num_train_epochs': args.epochs,
            'weight_decay': 0,
            'learning_rate': 2e-5,
            'adam_epsilon': 1e-8,
            'warmup_ratio': 0.1,
            'warmup_steps': 0,
            'max_grad_norm': 1.0,
            'do_lower_case': False,

            'logging_steps': 300,
            'save_steps': 300,
            "no_cache": False,
            "no_save": False,
            "save_recent_only": True,
            'save_model_every_epoch': False,
            'n_fold': 3,
            'evaluate_during_training': True,
            "evaluate_during_training_silent": False,
            'evaluate_during_training_steps': 300,
            "evaluate_during_training_verbose": True,
            'use_cached_eval_features': False,
            "save_best_model": True,
            'save_eval_checkpoints': False,  # changed this
            'tensorboard_dir': None,
            "save_optimizer_and_scheduler": True,

            'regression': True,

            'overwrite_output_dir': True,
            'reprocess_input_data': True,

            'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
            'n_gpu': 1,
            'use_multiprocessing': True,
            "multiprocessing_chunksize": 500,
            'silent': False,

            'wandb_project': None,
            'wandb_kwargs': {},

            "use_early_stopping": True,
            "early_stopping_patience": args.stop,
            "early_stopping_delta": 0,
            "early_stopping_metric": "eval_loss",
            "early_stopping_metric_minimize": True,
            "early_stopping_consider_epochs": False,

            "manual_seed": SEED,

            "config": {},
            "local_rank": -1,
            "encoding": None,
        }
        # experimenting with using bothcpu and 1 gpu available
        model = QuestModel("bert", "bert-base-cased", num_labels=1, use_cuda=torch.cuda.is_available(),
                           args=transformer_config)

    elif args.device == 'gpu':
        transformer_config = {
            "fp16": True,  # I added this
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "best_model_dir": "temp/outputs/best_model",
            "lazy_text_a_column": 0,
            "lazy_text_b_column": 1,
            "lazy_labels_column": 2,
            "lazy_header_row": True,
            "regression": True,
            "train_batch_size": args.batch,
            "eval_batch_size": args.batch,
            "num_train_epochs": args.epochs,
            "n_gpu": 5,
        }

        model = QuestModel("bert", "bert-base-cased", num_labels=1, use_cuda=torch.cuda.is_available(),
                           args=transformer_config)

    else:
        model = None
        transformer_config = None

    model.train_model(train_df, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                      mae=mean_absolute_error)

    # loading best model
    model = QuestModel("bert", transformer_config["best_model_dir"], num_labels=1,
                       use_cuda=torch.cuda.is_available(), args=transformer_config)

    test_sentence_pairs = list(map(list, zip(test_df0['text_a'].to_list(), test_df0['text_b'].to_list())))
    print(test_sentence_pairs[:3])

    preds, raw_out = model.predict(test_sentence_pairs)
    test_df01 = test_df0.copy()
    test_df01.loc[:, 'predictions'] = preds
    test_df = un_fit(test_df01, 'predictions')

    test_df.to_csv(os.path.join(outdir, 'test_results_%depochs_%s.tsv' % (args.epochs, args.target)), header=True,
                   sep='\t', index=False, encoding='utf-8')
    draw_scatterplot(test_df, 'labels', 'predictions', os.path.join(outdir, 'test_%depochs_%s.png' %
                                                                    (args.epochs, args.target)),
                     "English-Russian human")
    print_stat(test_df, 'labels', 'predictions')

    print('Segids:', test_df['segid'].to_list()[:5])
    print('Gold scores:', test_df['labels'].to_list()[:5])
    print('Predicted:', test_df['predictions'].to_list()[:5])

    # pearsonr and spearmanr return (correlation, p_value)
    pearson = pearsonr(test_df['labels'].to_list(), preds.flatten())
    # a nonparametric measure of the monotonicity of the relationship between two datasets
    spearman = spearmanr(test_df['labels'].to_list(), preds.flatten())

    print('Pearson correlation: rho = %.4f (p < %.3f)' % (pearson[0], pearson[1]))
    print('Spearman correlation: rho = %.4f (p < %.3f)' % (spearman[0], spearman[1]))

    end = time.time()
    print('Training a regressor with hu_TransQuest (%d epochs, %s) took %d minites' % (args.epochs, args.target,
                                                                                    (end - start) / 60))
