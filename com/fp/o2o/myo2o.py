import numpy as np
import pandas as pd
import seaborn as sns
import os,pickle,sys
import matplotlib.pyplot as plt
from datetime import date

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import StandardScaler


def show_data_info():
    dfoff = pd.read_csv("/Users/dongsheng/Documents/me/Machinelearning/tianchi/o2o/ccf_offline_stage1_train_1.csv")
    dftest = pd.read_csv("/Users/dongsheng/Documents/me/Machinelearning/tianchi/o2o/ccf_offline_stage1_test_revised_3.csv")

    dfon = pd.read_csv("/Users/dongsheng/Documents/me/Machinelearning/tianchi/o2o/ccf_online_stage1_train_2.csv")
    print(dfoff.head(5))
    print('有优惠券,购买商品条数',dfoff[(dfoff['Date_received'] != 'null') & (dfoff['Date'] != 'null')].shape[0])
    print('无优惠券,购买商品条数',dfoff[(dfoff['Date_received'] == 'null') & (dfoff['Date'] != 'null')].shape[0])
    print('有优惠券,不购买商品条数',dfoff[(dfoff['Date_received'] != 'null') & (dfoff['Date'] == 'null')].shape[0])

    print('1. User_id in training set but not in test set', set(dftest['User_id']) - set(dfoff['User_id']))
    # 在测试集中出现的商户但训练集没有出现
    print('2. Merchant_id in training set but not in test set', set(dftest['Merchant_id']) - set(dfoff['Merchant_id']))

    print('Discount_rate 类型:' , dfoff['Discount_rate'].unique())
    print('Discount 类型:' , dfoff['Distance'].unique())
    return dfoff,dftest,dfon

def convertRate(date):
    """Convert discount to rate """
    if date == 'null':
        return 1.0
    elif ':' in date:
        dates = date.split(':')
        return 1.0 - float(dates[1]) / float(dates[0])
    else:
        return float(date)

#折扣限制
def getDiscountMan(row):
    if ':' in row:
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0

def getDiscountJian(row):
    if ':' in row:
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0

def getDiscountType(row):
    if row == 'null':
        return 'null'
    elif ':' in row:
        return 1
    else:
        return 0


def processData(df):
    # convert discount_rate
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    print(df['discount_rate'].unique())

    df['distance'] = df['Distance'].replace('null',-1).astype(int)
    print(df['distance'].unique())
    return df

def date_info(dfoff):
    date_received = dfoff['Date_received'].unique()
    date_received = sorted(date_received[date_received != 'null'])

    date_buy = dfoff['Date'].unique()
    date_buy = sorted(date_buy[date_buy != 'null'])

    date_buy = sorted(dfoff[dfoff['Date'] != 'null']['Date'])
    print('优惠券收到日期从', date_received[0], '到', date_received[-1])
    print('消费日期从', date_buy[0], '到', date_buy[-1])
    return date_received,date_buy

def show_coupon_used(couponbydate, buybydate,date_received):
    sns.set_style('ticks')
    sns.set_context("notebook",font_scale=1.4)
    plt.figure(figsize= (12,8))
    date_received_dt = pd.to_datetime(date_received,format= '%Y%m%d')

    # 2行一列
    plt.subplot(211)
    # bar柱状图 ， plot线性
    plt.bar(date_received_dt,couponbydate['count'],label = 'number of coupon received')
    plt.bar(date_received_dt,buybydate['count'],label = 'number of coupon used')
    # y轴刻度函数，linear 线性，symlog，logit
    plt.yscale('log')
    plt.ylabel('Count')
    # 标签位置
    plt.legend()

    plt.subplot(212)
    plt.bar(date_received_dt, buybydate['count'] / couponbydate['count'])
    plt.ylabel('Ratio(coupon used/coupon received)')
    plt.tight_layout()
    # plt.show()


def coupon_info(dfoff):
    couponbydate = dfoff[dfoff['Date_received'] != 'null'][['Date_received', 'Date']].groupby(['Date_received'],
                                                                                              as_index=False).count()
    couponbydate.columns = ['Date_received', 'count']
    buybydate = dfoff[(dfoff['Date']!= 'null') & (dfoff['Date_received'] != 'null')][['Date_received','Date']].groupby(['Date_received'],as_index=False).count()
    buybydate.columns = ['Date_received','count']
    date_received, date_buy = date_info(dfoff)
    show_coupon_used(couponbydate, buybydate,date_received)

    # return couponbydate,buybydate

def getWeekday(row):
    if row == 'null':
        return row
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1

def transformWeekday(dfoff,dftest):
    dfoff['weekday'] = dfoff['Date_received'].astype(str).apply(getWeekday)
    dftest['weekday'] = dftest['Date_received'].astype(str).apply(getWeekday)

    # weekday_type :  周六和周日为1，其他为0
    dfoff['weekday_type'] = dfoff['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )
    dftest['weekday_type'] = dftest['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )

    # change weekday to one-hot encoding
    weekdaycols = ['weekday_' + str(i) for i in range(1, 8)]
    print(weekdaycols)

    tmpdf = pd.get_dummies(dfoff['weekday'].replace('null', np.nan))
    tmpdf.columns = weekdaycols
    dfoff[weekdaycols] = tmpdf

    tmpdf = pd.get_dummies(dftest['weekday'].replace('null', np.nan))
    tmpdf.columns = weekdaycols
    dftest[weekdaycols] = tmpdf
    return dfoff,dftest,weekdaycols


def label(row):
    if row['Date_received'] == 'null':
        return -1
    if row['Date'] != 'null':
        td = pd.to_datetime(row['Date'], format='%Y%m%d') -  pd.to_datetime(row['Date_received'], format='%Y%m%d')
        if td <= pd.Timedelta(15,'D'):
            return 1
    return 0





def check_model(data, predictors):
    classifier = lambda: SGDClassifier(
        loss='log',
        penalty='elasticnet',
        fit_intercept=True,
        max_iter=100,
        shuffle=True,
        n_jobs=1,
        class_weight=None)

    model = Pipeline(steps=[
        ('ss', StandardScaler()),
        ('en', classifier())
    ])

    parameters = {
        'en__alpha': [0.001, 0.01, 0.1],
        'en__l1_ratio': [0.001, 0.01, 0.1]
    }

    folder = StratifiedKFold(n_splits=3, shuffle=True)

    grid_search = GridSearchCV(
        model,
        parameters,
        cv=folder,
        n_jobs=-1,
        verbose=1)
    grid_search = grid_search.fit(data[predictors],
                                  data['label'])

    return grid_search



if __name__ == '__main__':
    dfoff, dftest, dfon = show_data_info()
    processData(dfoff)
    processData(dftest)

    # 0
    # couponbydate,buybydate = coupon_info(dfoff)

    dfoff, dftest,weekdaycols = transformWeekday(dfoff,dftest)
    dfoff['label'] = dfoff.apply(label, axis=1)
    print(dfoff['label'].value_counts())
    print('已有columns：',dfoff.columns.tolist())
    print(dfoff.head(2))

    # data split
    df = dfoff[dfoff['label'] != -1].copy()
    train = df[(df['Date_received'] < '20160516')].copy()
    valid = df[(df['Date_received'] >= '20160516') & (df['Date_received'] <= '20160615')].copy()
    print(train['label'].value_counts())
    print(valid['label'].value_counts())

    # model1
    original_feature = ['discount_rate','discount_type','discount_man', 'discount_jian','distance', 'weekday', 'weekday_type'] + weekdaycols
    predictors = original_feature
    print(predictors)

    if not os.path.isfile('1_model.pkl'):
        model = check_model(train, predictors)
        print(model.best_score_)
        print(model.best_params_)
        with open('1_model.pkl', 'wb') as f:
            pickle.dump(model, f)
    else:
        with open('1_model.pkl', 'rb') as f:
            model = np.pickle.load(f)

    # valid predict
    y_valid_pred = model.predict_proba(valid[predictors])
    valid1 = valid.copy()
    valid1['pred_prob'] = y_valid_pred[:, 1]
    valid1.head(2)

    vg = valid1.groupby(['Coupon_id'])
    aucs = []
    for i in vg:
        tmpdf = i[1]
        if len(tmpdf['label'].unique()) != 2:
            continue
        fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
        aucs.append(auc(fpr, tpr))
    print(np.average(aucs))

    y_test_pred = model.predict_proba(dftest[predictors])
    dftest1 = dftest[['User_id', 'Coupon_id', 'Date_received']].copy()
    dftest1['label'] = y_test_pred[:, 1]
    dftest1.to_csv('submit1.csv', index=False, header=False)
    dftest1.head(2)