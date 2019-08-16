# General imports
import numpy as np
import pandas as pd
import os, sys, gc, warnings, random

from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
import time

from tqdm import tqdm

import math
warnings.filterwarnings('ignore')


########################### Helpers
#################################################################################
## -------------------
## Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
## -------------------

########################### Vars
#################################################################################
SEED = 42
seed_everything(SEED)
LOCAL_TEST = False
TARGET = 'isFraud'


## learning time check
start = time.time()

########################### DATA LOAD
#################################################################################
print('Load Data')

train_df = pd.read_pickle('../input/ieee-data-minification/train_transaction.pkl')

if LOCAL_TEST:
    test_df = train_df.iloc[-100000:, ].reset_index(drop=True)
    train_df = train_df.iloc[:400000, ].reset_index(drop=True)

    train_identity = pd.read_pickle('../input/ieee-data-minification/train_identity.pkl')
    test_identity = train_identity[train_identity['TransactionID'].isin(test_df['TransactionID'])].reset_index(
        drop=True)
    train_identity = train_identity[train_identity['TransactionID'].isin(train_df['TransactionID'])].reset_index(
        drop=True)
else:
    test_df = pd.read_pickle('../input/ieee-data-minification/test_transaction.pkl')
    test_identity = pd.read_pickle('../input/ieee-data-minification/test_identity.pkl')

########################### Reset values for "noise" card1
valid_card = train_df['card1'].value_counts()

# 10개 이하의 데이터는 지워줌..
valid_card = valid_card[valid_card > 10]
valid_card = list(valid_card.index)

train_df['card1'] = np.where(train_df['card1'].isin(valid_card), train_df['card1'], np.nan)
test_df['card1'] = np.where(test_df['card1'].isin(valid_card), test_df['card1'], np.nan)


########################### Freq encoding
i_cols = ['card1','card2','card3','card5',
          'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
          'D1','D2','D3','D4','D5','D6','D7','D8','D9',
          'addr1','addr2',
          'dist1','dist2',
          'P_emaildomain', 'R_emaildomain'
         ]

for col in i_cols:
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    fq_encode = temp_df[col].value_counts().to_dict()
    train_df[col+'_fq_enc'] = train_df[col].map(fq_encode)
    test_df[col+'_fq_enc']  = test_df[col].map(fq_encode)


########################### ProductCD and M4 Target mean
for col in ['ProductCD','M4']:
    temp_dict = train_df.groupby([col])[TARGET].agg(['mean']).reset_index().rename(
                                                        columns={'mean': col+'_target_mean'})
    temp_dict.index = temp_dict[col].values
    temp_dict = temp_dict[col+'_target_mean'].to_dict()

    train_df[col+'_target_mean'] = train_df[col].map(temp_dict)
    test_df[col+'_target_mean']  = test_df[col].map(temp_dict)

########################### Encode Str columns
for col in list(train_df):
    if train_df[col].dtype == 'O':
        print(col)
        train_df[col] = train_df[col].fillna('unseen_before_label')
        test_df[col] = test_df[col].fillna('unseen_before_label')

        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

        le = LabelEncoder()
        le.fit(list(train_df[col]) + list(test_df[col]))
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')

########################### TransactionAmt

# Let's add some kind of client uID based on cardID ad addr columns
# The value will be very specific for each client so we need to remove it
# from final feature. But we can use it for aggregations.
train_df['uid'] = train_df['card1'].astype(str) + '_' + train_df['card2'].astype(str) + '_' + train_df['card3'].astype(
    str) + '_' + train_df['card4'].astype(str)
test_df['uid'] = test_df['card1'].astype(str) + '_' + test_df['card2'].astype(str) + '_' + test_df['card3'].astype(
    str) + '_' + test_df['card4'].astype(str)

train_df['uid2'] = train_df['uid'].astype(str) + '_' + train_df['addr1'].astype(str) + '_' + train_df['addr2'].astype(
    str)
test_df['uid2'] = test_df['uid'].astype(str) + '_' + test_df['addr1'].astype(str) + '_' + test_df['addr2'].astype(str)

# Check if Transaction Amount is common or not (we can use freq encoding here)
# In our dialog with model we are telling to trust or not to these values
valid_card = train_df['TransactionAmt'].value_counts()
valid_card = valid_card[valid_card > 10]
valid_card = list(valid_card.index)

train_df['TransactionAmt_check'] = np.where(train_df['TransactionAmt'].isin(test_df['TransactionAmt']), 1, 0)
test_df['TransactionAmt_check'] = np.where(test_df['TransactionAmt'].isin(train_df['TransactionAmt']), 1, 0)

# For our model current TransactionAmt is a noise (even when features importances are telling contrariwise)
# There are many unique values and model doesn't generalize well
# Lets do some aggregations
i_cols = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2']

for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col + '_TransactionAmt_' + agg_type
        temp_df = pd.concat([train_df[[col, 'TransactionAmt']], test_df[[col, 'TransactionAmt']]])
        temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(
            columns={agg_type: new_col_name})

        temp_df.index = list(temp_df[col])
        temp_df = temp_df[new_col_name].to_dict()

        train_df[new_col_name] = train_df[col].map(temp_df)
        test_df[new_col_name] = test_df[col].map(temp_df)



########################### Anomaly Search in geo information

# Let's look on bank addres and client addres matching
# card3/card5 bank country and name?
# Addr2 -> Clients geo position (country)
# Most common entries -> normal transactions
# Less common etries -> some anonaly
train_df['bank_type'] = train_df['card3'].astype(str)+'_'+train_df['card5'].astype(str)
test_df['bank_type']  = test_df['card3'].astype(str)+'_'+test_df['card5'].astype(str)

train_df['address_match'] = train_df['bank_type'].astype(str)+'_'+train_df['addr2'].astype(str)
test_df['address_match']  = test_df['bank_type'].astype(str)+'_'+test_df['addr2'].astype(str)

for col in ['address_match','bank_type']:
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    temp_df[col] = np.where(temp_df[col].str.contains('nan'), np.nan, temp_df[col])
    temp_df = temp_df.dropna()
    fq_encode = temp_df[col].value_counts().to_dict()
    train_df[col] = train_df[col].map(fq_encode)
    test_df[col]  = test_df[col].map(fq_encode)

train_df['address_match'] = train_df['address_match']/train_df['bank_type']
test_df['address_match']  = test_df['address_match']/test_df['bank_type']



########################### Model Features
## We can use set().difference() but order matters
rm_cols = [
    'TransactionID','TransactionDT', # These columns are pure noise right now
    TARGET,                          # Not target in features))
    'uid','uid2',                    # Our new clien uID -> very noisy data
    'bank_type',                     # Victims bank could differ by time
]
features_columns = list(train_df)
for col in rm_cols:
    if col in features_columns:
        features_columns.remove(col)



########################### Model params
lgb_params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.01,
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.7,
                    'subsample_freq':1,
                    'subsample':1,
                    'n_estimators':800,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': SEED,
                    'early_stopping_rounds':100,
                }

########################### Model
import lightgbm as lgb


def make_predictions(tr_df, tt_df, features_columns, target, lgb_params, NFOLDS=2):
    folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

    X, y = tr_df[features_columns], tr_df[target]
    P, P_y = tt_df[features_columns], tt_df[target]

    tt_df = tt_df[['TransactionID', target]]
    predictions = np.zeros(len(tt_df))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print('Fold:', fold_)
        tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
        vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]

        print(len(tr_x), len(vl_x))
        tr_data = lgb.Dataset(tr_x, label=tr_y)

        if LOCAL_TEST:
            vl_data = lgb.Dataset(P, label=P_y)
        else:
            vl_data = lgb.Dataset(vl_x, label=vl_y)

        estimator = lgb.train(
            lgb_params,
            tr_data,
            valid_sets=[tr_data, vl_data],
            verbose_eval=200,
        )

        pp_p = estimator.predict(P)
        predictions += pp_p / NFOLDS

        if LOCAL_TEST:
            feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(), X.columns)),
                                       columns=['Value', 'Feature'])
            print(feature_imp)

        del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
        gc.collect()

    tt_df['prediction'] = predictions

    return tt_df
## -------------------


########################### Model Train
if LOCAL_TEST:
    test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params)
    print(metrics.roc_auc_score(test_predictions[TARGET], test_predictions['prediction']))
else:
    lgb_params['learning_rate'] = 0.005
    lgb_params['n_estimators'] = 2000
    lgb_params['early_stopping_rounds'] = 100
    test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params, NFOLDS=10)



########################### Export
if not LOCAL_TEST:
    test_predictions['isFraud'] = test_predictions['prediction']
    test_predictions[['TransactionID','isFraud']].to_csv('submission.csv', index=False)


end = time.time()

print('elasped time : {}'.format (end - start))




