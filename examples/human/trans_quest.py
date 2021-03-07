# run it from hu_TransQuest: python3 -m examples.human.trans_quest

import os
import shutil
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from examples.wmt_2020.common.util.download import download_from_google_drive
from examples.wmt_2020.common.util.draw import draw_scatterplot, print_stat
from examples.wmt_2020.common.util.normalizer import fit, un_fit
from examples.wmt_2020.common.util.postprocess import format_submission
from examples.wmt_2020.common.util.reader import read_annotated_file, read_test_file
from examples.human.transformer_config import TEMP_DIRECTORY, DRIVE_FILE_ID, GOOGLE_DRIVE, MODEL_NAME, \
    transformer_config, MODEL_TYPE, SEED, RESULT_FILE, RESULT_IMAGE, TARGET, SUBMISSION_FILE
from transquest.algo.my_transformers.evaluation import pearson_corr, spearman_corr
from transquest.algo.my_transformers.run_model import QuestModel

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

if GOOGLE_DRIVE:
    download_from_google_drive(DRIVE_FILE_ID, MODEL_NAME)

import sys
sys.path.append('//')

target = TARGET
# real datasets
TRAIN_FILE = "examples/human/data/train.enru.df.len60.tsv"
DEV_FILE = "examples/human/data/dev.enru.df.len60.tsv"
TEST_FILE = "examples/human/data/test.enru.df.len60.tsv"

train = read_annotated_file(TRAIN_FILE, index="segid", score=target)
dev = read_annotated_file(DEV_FILE, index="segid", score=target)
test = read_annotated_file(TEST_FILE, index="segid", score=target)

train = train[['original', 'translation', target]]
dev = dev[['original', 'translation', target]]
test = test[['index', 'original', 'translation', target]]

index = test['index'].to_list()
gold = test[target].to_list()

train = train.rename(columns={'original': 'text_a', 'translation': 'text_b', target: 'labels'}).dropna()
dev = dev.rename(columns={'original': 'text_a', 'translation': 'text_b', target: 'labels'}).dropna()
test = test.rename(columns={'original': 'text_a', 'translation': 'text_b', target: 'labels'}).dropna()

test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

train = fit(train, 'labels')
dev = fit(dev, 'labels')
test = fit(test, 'labels')  # NB! testing on real test? not on dev as in WMT

if transformer_config["evaluate_during_training"]:
    if transformer_config["n_fold"] > 1:
        dev_preds = np.zeros((len(dev), transformer_config["n_fold"]))
        test_preds = np.zeros((len(test), transformer_config["n_fold"]))
        for i in range(transformer_config["n_fold"]):

            if os.path.exists(transformer_config['output_dir']) and os.path.isdir(transformer_config['output_dir']):
                shutil.rmtree(transformer_config['output_dir'])

            model = QuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                               args=transformer_config)
            train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
            model.train_model(train_df, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                              mae=mean_absolute_error)
            model = QuestModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=1,
                               use_cuda=torch.cuda.is_available(), args=transformer_config)
            result, model_outputs, wrong_predictions = model.eval_model(dev, pearson_corr=pearson_corr,
                                                                        spearman_corr=spearman_corr,
                                                                        mae=mean_absolute_error)
            predictions, raw_outputs = model.predict(test_sentence_pairs)
            dev_preds[:, i] = model_outputs
            test_preds[:, i] = predictions

        dev['predictions'] = dev_preds.mean(axis=1)
        test['predictions'] = test_preds.mean(axis=1)

    else:
        model = QuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                           args=transformer_config)
        train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
        model.train_model(train_df, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                          mae=mean_absolute_error)
        model = QuestModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=1,
                           use_cuda=torch.cuda.is_available(), args=transformer_config)
        result, model_outputs, wrong_predictions = model.eval_model(dev, pearson_corr=pearson_corr,
                                                                    spearman_corr=spearman_corr,
                                                                    mae=mean_absolute_error)
        predictions, raw_outputs = model.predict(test_sentence_pairs)
        dev['predictions'] = model_outputs
        test['predictions'] = predictions

else:
    model = QuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                       args=transformer_config)
    model.train_model(train, pearson_corr=pearson_corr, spearman_corr=spearman_corr, mae=mean_absolute_error)
    result, model_outputs, wrong_predictions = model.eval_model(dev, pearson_corr=pearson_corr,
                                                                spearman_corr=spearman_corr, mae=mean_absolute_error)
    predictions, raw_outputs = model.predict(test_sentence_pairs)
    dev['predictions'] = model_outputs
    test['predictions'] = predictions

dev = un_fit(dev, 'labels')
dev = un_fit(dev, 'predictions')
test = un_fit(test, 'predictions')

outname = 'dev_' + MODEL_NAME + '_' + TARGET + '_' + str(transformer_config['num_train_epochs']) + 'epochs'
dev.to_csv(os.path.join(TEMP_DIRECTORY, outname + '.tsv'), header=True, sep='\t', index=False, encoding='utf-8')
draw_scatterplot(dev, 'labels', 'predictions', os.path.join(TEMP_DIRECTORY, outname + '.jpg'),
                 "English-Russian human dev")
print('===Results on dev===')
print_stat(dev, 'labels', 'predictions')

# in my setting there is no difference btw dev and test: use 8000 for training with inbuilt 3-fold evaluation!
outname = 'test_' + MODEL_NAME + '_' + TARGET + '_' + str(transformer_config['num_train_epochs']) + 'epochs'
test.to_csv(os.path.join(TEMP_DIRECTORY, outname + '.tsv'), header=True, sep='\t', index=False, encoding='utf-8')
draw_scatterplot(test, 'labels', 'predictions', os.path.join(TEMP_DIRECTORY, outname + '.jpg'),
                 "English-Russian human test")
print('===Results on test===')
print_stat(test, 'labels', 'predictions')

# format_submission(df=test, index=index, language_pair="ru-en", method="hu_TransQuest",
#                   path=os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE), index_type="Auto")
