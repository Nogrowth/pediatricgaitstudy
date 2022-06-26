import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
from pandas import DataFrame as df
from scipy.stats import ttest_rel
from sklearn.metrics import roc_curve, auc
from scipy.integrate import simps
from tableone import TableOne
plt.rcParams["font.family"] = "arial"
plt.rc('font', size=8)

data = pd.read_excel("./data/Data.xlsx")
data = data[:26]
data.rename(columns={'SL_Long':'SLl', 'SL_Short':'SLs',
                     'stance_Long':'StPDl', 'stance_Short':'StPDs',
                     'swing_Long':'SwPDl', 'swing_Short':'SwPDs',
                     }, inplace=True)

"""temporospatial parameter 시각화"""
# plt.subplot(1, 3, 1)
# plt.boxplot([data.SLl, data.SLs])
# plt.title("step length")
# plt.xticks([1, 2], ["Longer", "Shorter"])
#
# plt.subplot(1, 3, 2)
# plt.boxplot([data.StPDl, data.StPDs])
# plt.title("stance phase\nduration")
# plt.xticks([1, 2], ["Longer", "Shorter"])

# plt.subplot(1, 3, 3)
# plt.boxplot([data.SwPDl, data.SwPDs])
# plt.title("swing phase\nduration")
# plt.xticks([1, 2], ["Longer", "Shorter"])

# plt.subplots_adjust(wspace=0.5)
#
# plt.show()


"""temporospatial parameter paired-t-test"""
# statistic, p_value = ttest_rel(
#     data.SLl, data.SLs)
# print(f"statistic for step length : {statistic:.5f}")
# print(f"p_value for step length = {p_value:.5f}")
#
# statistic, p_value = ttest_rel(
#     data.StPDl, data.StPDs)
# print(f"statistic for stance phase duration : {statistic:.5f}")
# print(f"p_value for stance phase duration = {p_value:.5f}")
#
# statistic, p_value = ttest_rel(
#     data.SwPDl, data.SwPDs)
# print(f"statistic for swing phase duration : {statistic:.5f}")
# print(f"p_value for swing phase duration = {p_value:.5f}")


"""ROC curve 그리기"""
"""
True Dx.를 3가지 기준으로 각각 나눈 것이 SLDx., StPDDx., SwPDDx.
ROC 커브 및 AUC 구하기
"""
# y_true_SL = data['SLDx.'].values
# y_true_StPD = data['StPDDx.'].values
# y_true_SwPD = data['SwPDDx.'].values
# y_pred_abs = data['LLD(ABS)'].values
# y_pred_ratio = data['LLD(ratio, %)'].values
#
#
# def get_tpr(true, pred, threshold):
#     # TPR : positive를 positive로 예측한 수 / 실제 positive
#     ground_truth = len(true[true == 1]) # 실제 positive
#     tp = 0
#     for idx, v in enumerate(pred):
#         if (v >= threshold) and (true[idx] == 1):
#             tp += 1
#     tpr = tp / ground_truth
#     return tpr
#
# def get_fpr(true, pred, threshold):
#     # FPR : negative를 positive로 예측한 수 / 실제 negative
#     ground_false = len(true[true == 0]) # 실제 negative
#     fp = 0
#     for idx, v in enumerate(pred):
#         if (v >= threshold) and (true[idx] == 0):
#             fp += 1
#     fpr = fp / ground_false
#     return fpr
#
# def roc_plot(true, pred):
#     tpr, fpr = [] , []
#     for _ in np.arange(pred.min(), pred.max(), 0.01): # pred 를 thresholds 처럼 사용했음
#         tpr.append(get_tpr(true,pred,_ ))
#         fpr.append(get_fpr(true,pred,_ ))
#     fig = plt.figure(figsize=(9, 6))
#     plt.plot(fpr, tpr)
#     plt.scatter(fpr, tpr)
#     plt.plot([0, 1], [0, 1])
#     plt.xlabel('False-Positive-Rate')
#     plt.ylabel('True-Positive-Rate')
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     plt.title('ROC Curve 2D')
#     plt.show()
#
# plt.figure(figsize=(10, 8))
#
# plt.subplot(2, 3, 1)
# tpr, fpr = [] , []
# for _ in np.arange(y_pred_abs.min(), y_pred_abs.max(), 0.01): # pred 를 thresholds 처럼 사용했음
#     tpr.append(get_tpr(y_true_SL,y_pred_abs,_ ))
#     fpr.append(get_fpr(y_true_SL,y_pred_abs,_ ))
# plt.plot(fpr, tpr, color='black')
# plt.fill_between(fpr[0:len(fpr)], tpr, 0, facecolor='gray', alpha=.2)
# auc = np.trapz(fpr, tpr, dx=0.01, axis=-1)
# auc = round(1+auc, 3)
# plt.text(x=0.5, y=0.2, s= f"AUC : {auc}")
# plt.scatter(fpr, tpr, s=1)
# plt.plot([0, 1], [0, 1])
# plt.xlabel('False-Positive-Rate')
# plt.ylabel('True-Positive-Rate')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.title('Step length &\nabsolute LLD')
#
# plt.subplot(2, 3, 4)
# tpr, fpr = [] , []
# for _ in np.arange(y_pred_ratio.min(), y_pred_ratio.max(), 0.01): # pred 를 thresholds 처럼 사용했음
#     tpr.append(get_tpr(y_true_SL,y_pred_ratio,_ ))
#     fpr.append(get_fpr(y_true_SL,y_pred_ratio,_ ))
# plt.plot(fpr, tpr, color='black')
# plt.fill_between(fpr[0:len(fpr)], tpr, 0, facecolor='gray', alpha=.2)
# auc = np.trapz(fpr, tpr, dx=0.01, axis=-1)
# auc = round(1+auc, 3)
# plt.text(x=0.5, y=0.2, s= f"AUC : {auc}")
# plt.scatter(fpr, tpr, s=1)
# plt.plot([0, 1], [0, 1])
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.title('Step length &\nLLD ratio')
#
# plt.subplot(2, 3, 2)
# tpr, fpr = [] , []
# for _ in np.arange(y_pred_abs.min(), y_pred_abs.max(), 0.01): # pred 를 thresholds 처럼 사용했음
#     tpr.append(get_tpr(y_true_StPD,y_pred_abs,_ ))
#     fpr.append(get_fpr(y_true_StPD,y_pred_abs,_ ))
# plt.plot(fpr, tpr, color='black')
# plt.fill_between(fpr[0:len(fpr)], tpr, 0, facecolor='gray', alpha=.2)
# auc = np.trapz(fpr, tpr, dx=0.01, axis=-1)
# auc = round(1+auc, 3)
# plt.text(x=0.5, y=0.2, s= f"AUC : {auc}")
# plt.scatter(fpr, tpr, s=1)
# plt.plot([0, 1], [0, 1])
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.title('Stance phase duration &\nabsolute LLD')
#
# plt.subplot(2, 3, 5)
# tpr, fpr = [] , []
# for _ in np.arange(y_pred_ratio.min(), y_pred_ratio.max(), 0.01): # pred 를 thresholds 처럼 사용했음
#     tpr.append(get_tpr(y_true_StPD,y_pred_ratio,_ ))
#     fpr.append(get_fpr(y_true_StPD,y_pred_ratio,_ ))
# plt.plot(fpr, tpr, color='black')
# plt.fill_between(fpr[0:len(fpr)], tpr, 0, facecolor='gray', alpha=.2)
# auc = np.trapz(fpr, tpr, dx=0.01, axis=-1)
# auc = round(1+auc, 3)
# plt.text(x=0.5, y=0.2, s= f"AUC : {auc}")
# plt.scatter(fpr, tpr, s=1)
# plt.plot([0, 1], [0, 1])
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.title('Stance phase duration &\nLLD ratio')
#
# plt.subplot(2, 3, 3)
# tpr, fpr = [] , []
# for _ in np.arange(y_pred_abs.min(), y_pred_abs.max(), 0.01): # pred 를 thresholds 처럼 사용했음
#     tpr.append(get_tpr(y_true_SwPD,y_pred_abs,_ ))
#     fpr.append(get_fpr(y_true_SwPD,y_pred_abs,_ ))
# plt.plot(fpr, tpr, color='black')
# plt.fill_between(fpr[0:len(fpr)], tpr, 0, facecolor='gray', alpha=.2)
# auc = np.trapz(fpr, tpr, dx=0.01, axis=-1)
# auc = round(1+auc, 3)
# plt.text(x=0.5, y=0.2, s= f"AUC : {auc}")
# plt.scatter(fpr, tpr, s=1)
# plt.plot([0, 1], [0, 1])
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.title('Swing phase duration &\nabsolute LLD')
#
# plt.subplot(2, 3, 6)
# tpr, fpr = [] , []
# for _ in np.arange(y_pred_ratio.min(), y_pred_ratio.max(), 0.01): # pred 를 thresholds 처럼 사용했음
#     tpr.append(get_tpr(y_true_SwPD,y_pred_ratio,_ ))
#     fpr.append(get_fpr(y_true_SwPD,y_pred_ratio,_ ))
# plt.plot(fpr, tpr, color='black')
# plt.fill_between(fpr[0:len(fpr)], tpr, 0, facecolor='gray', alpha=.2)
# auc = np.trapz(fpr, tpr, dx=0.01, axis=-1)
# auc = round(1+auc, 3)
# plt.text(x=0.5, y=0.2, s= f"AUC : {auc}")
# plt.scatter(fpr, tpr, s=1)
# plt.plot([0, 1], [0, 1])
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.title('Swing phase duration &\nLLD ratio')
#
# plt.subplots_adjust(hspace=0.5, wspace=0.5)
# plt.show()

"""
Threshold 구하기 : 해당 threshold에서 민감도/특이도 어떻게 되는지 기능 추가
"""
# y_true_SL = data['SLDx.'].values
# y_true_StPD = data['StPDDx.'].values
# y_true_SwPD = data['SwPDDx.'].values
# y_pred_abs = data['LLD(ABS)'].values
# y_pred_ratio = data['LLD(ratio, %)'].values
#
# def get_sen(true, pred, threshold):
#     # TPR : positive를 positive로 예측한 수 / 실제 positive
#     ground_truth = len(true[true == 1]) # 실제 positive
#     tp = 0
#     for idx, v in enumerate(pred):
#         if (v >= threshold) and (true[idx] == 1):
#             tp += 1
#     tpr = tp / ground_truth
#     return threshold, tpr
#
# def get_spe(true, pred, threshold):
#     # FPR : negative를 positive로 예측한 수 / 실제 negative
#     ground_false = len(true[true == 0]) # 실제 negative
#     fp = 0
#     for idx, v in enumerate(pred):
#         if (v >= threshold) and (true[idx] == 0):
#             fp += 1
#     fpr = fp / ground_false
#     return threshold, 1-fpr
#
# plt.figure(figsize=(16, 8))
#
# plt.subplot(2, 3, 1)
# sen, spe = [] , []
# for _ in np.arange(y_pred_abs.min(), y_pred_abs.max(), 0.1): # pred 를 thresholds 처럼 사용했음
#     sen.append(get_sen(y_true_SL,y_pred_abs,_ ))
#     spe.append(get_spe(y_true_SL,y_pred_abs,_ ))
# th1, sensi = zip(*sen)
# plt1 = plt.plot(th1, sensi, label="sensitivity")
# th2, speci = zip(*spe)
# plt2 = plt.plot(th1, speci, label="specificity")
# plt.title('Absolute LLD &\nStep length')
# sensi = np.asarray(sensi)
# speci = np.asarray(speci)
# idx = np.argwhere(np.diff(np.sign(sensi-speci))).flatten()[0]
# cutoff = th1[idx]
# plt.legend(loc='best', title=f'cut off value: {cutoff:.3f}')
#
# plt.subplot(2, 3, 4)
# sen, spe = [] , []
# for _ in np.arange(y_pred_ratio.min(), y_pred_ratio.max(), 0.1): # pred 를 thresholds 처럼 사용했음
#     sen.append(get_sen(y_true_SL,y_pred_ratio,_ ))
#     spe.append(get_spe(y_true_SL,y_pred_ratio,_ ))
# th1, sensi = zip(*sen)
# plt1 = plt.plot(th1, sensi, label="sensitivity")
# th2, speci = zip(*spe)
# plt2 = plt.plot(th1, speci, label="specificity")
# plt.title('LLD ratio &\nStep length')
# sensi = np.asarray(sensi)
# speci = np.asarray(speci)
# idx = np.argwhere(np.diff(np.sign(sensi-speci))).flatten()[0]
# cutoff = th1[idx]
# plt.legend(loc='best', title=f'cut off value: {cutoff:.3f}')
#
# plt.subplot(2, 3, 2)
# sen, spe = [] , []
# for _ in np.arange(y_pred_abs.min(), y_pred_abs.max(), 0.1): # pred 를 thresholds 처럼 사용했음
#     sen.append(get_sen(y_true_StPD,y_pred_abs,_ ))
#     spe.append(get_spe(y_true_StPD,y_pred_abs,_ ))
# th1, sensi = zip(*sen)
# plt1 = plt.plot(th1, sensi, label="sensitivity")
# th2, speci = zip(*spe)
# plt2 = plt.plot(th1, speci, label="specificity")
# plt.title('Absolute LLD &\nStance phase duration')
# sensi = np.asarray(sensi)
# speci = np.asarray(speci)
# idx = np.argwhere(np.diff(np.sign(sensi-speci))).flatten()[0]
# cutoff = th1[idx]
# plt.legend(loc='best', title=f'cut off value: {cutoff:.3f}')
#
# plt.subplot(2, 3, 5)
# sen, spe = [] , []
# for _ in np.arange(y_pred_ratio.min(), y_pred_ratio.max(), 0.1): # pred 를 thresholds 처럼 사용했음
#     sen.append(get_sen(y_true_StPD,y_pred_ratio,_ ))
#     spe.append(get_spe(y_true_StPD,y_pred_ratio,_ ))
# th1, sensi = zip(*sen)
# plt1 = plt.plot(th1, sensi, label="sensitivity")
# th2, speci = zip(*spe)
# plt2 = plt.plot(th1, speci, label="specificity")
# plt.title('LLD ratio &\nStance phase duration')
# sensi = np.asarray(sensi)
# speci = np.asarray(speci)
# idx = np.argwhere(np.diff(np.sign(sensi-speci))).flatten()[0]
# cutoff = th1[idx]
# plt.legend(loc='best', title=f'cut off value: {cutoff:.3f}')
#
# plt.subplot(2, 3, 3)
# sen, spe = [] , []
# for _ in np.arange(y_pred_abs.min(), y_pred_abs.max(), 0.1): # pred 를 thresholds 처럼 사용했음
#     sen.append(get_sen(y_true_SwPD,y_pred_abs,_ ))
#     spe.append(get_spe(y_true_SwPD,y_pred_abs,_ ))
# th1, sensi = zip(*sen)
# plt1 = plt.plot(th1, sensi, label="sensitivity")
# th2, speci = zip(*spe)
# plt2 = plt.plot(th1, speci, label="specificity")
# plt.title('Absolute LLD &\nSwing phase duration')
# sensi = np.asarray(sensi)
# speci = np.asarray(speci)
# idx = np.argwhere(np.diff(np.sign(sensi-speci))).flatten()[0]
# cutoff = th1[idx]
# plt.legend(loc='best', title=f'cut off value: {cutoff:.3f}')
#
# plt.subplot(2, 3, 6)
# sen, spe = [] , []
# for _ in np.arange(y_pred_ratio.min(), y_pred_ratio.max(), 0.1): # pred 를 thresholds 처럼 사용했음
#     sen.append(get_sen(y_true_SwPD,y_pred_ratio,_ ))
#     spe.append(get_spe(y_true_SwPD,y_pred_ratio,_ ))
# th1, sensi = zip(*sen)
# plt1 = plt.plot(th1, sensi, label="sensitivity")
# th2, speci = zip(*spe)
# plt2 = plt.plot(th1, speci, label="specificity")
# plt.title('LLD ratio &\nSwing phase duration')
# sensi = np.asarray(sensi)
# speci = np.asarray(speci)
# idx = np.argwhere(np.diff(np.sign(sensi-speci))).flatten()[0]
# cutoff = th1[idx]
# plt.legend(loc='best', title=f'cut off value: {cutoff:.3f}')
#
# plt.subplots_adjust(hspace=0.3, wspace=0.2)
# plt.show()

"""
paired_t_test
"""
def paired_t_test_with_visualization(opt):
    tag = []
    label = ""
    if opt == "angle":
        label = "flexion angle"
        tag = ['HALong_max', 'HAShort_max', 'KALong_', 'KAShort_', 'AALong_', 'AAShort_']
    elif opt == "momentum":
        label = "momentum"
        tag = ['HMLong_', 'HMShort_', 'KMLong_', 'KMShort_', 'AMLong_', 'AMShort_']

    """visualization"""
    plt.subplot(1, 3, 1)
    plt.boxplot([data[tag[0]], data[tag[1]]])
    plt.title(f"Hip joint\n{label}")
    plt.xticks([1, 2], ["Longer", "Shorter"])

    plt.subplot(1, 3, 2)
    plt.boxplot([data[tag[2]], data[tag[3]]])
    plt.title(f"Knee joint\n{label}")
    plt.xticks([1, 2], ["Longer", "Shorter"])

    plt.subplot(1, 3, 3)
    plt.boxplot([data[tag[4]], data[tag[5]]])
    plt.title(f"Ankle joint\n{label}")
    plt.xticks([1, 2], ["Longer", "Shorter"])

    plt.subplots_adjust(wspace=0.5)
    plt.show()

    """paired-t-test"""
    statistic, p_value = ttest_rel(
        data[tag[0]], data[tag[1]])
    print(f"statistic for hip joint {label} : {statistic:.5f}")
    print(f"p_value for hip joint {label} = {p_value:.5f}")

    statistic, p_value = ttest_rel(
        data[tag[2]], data[tag[3]])
    print(f"statistic for knee joint {label} : {statistic:.5f}")
    print(f"p_value for knee joint {label} = {p_value:.5f}")

    statistic, p_value = ttest_rel(
        data[tag[4]], data[tag[5]])
    print(f"statistic for ankle joint {label} : {statistic:.5f}")
    print(f"p_value for ankle joint {label} = {p_value:.5f}")

paired_t_test_with_visualization("momentum")
paired_t_test_with_visualization("angle")

"""
Wilcoxon test
"""
# temp_data = data[data['LLD(ABS)']>=15.75]
# print(scipy.stats.wilcoxon(temp_data['HMLong_'], temp_data['HMShort_'], alternative='greater'))
# print(scipy.stats.wilcoxon(temp_data['HMLong_'], temp_data['HMShort_']))

# """
# demographic table 만들기 by tableone
# """
# col_tableone = ['sex', 'age', 'Rt.', 'Lt.', 'long(1:R, 2:L)', 'LLD(ABS)', 'LLD(ratio, %)']
# categorical_tableone = ['sex', 'long(1:R, 2:L)']
# mytable = TableOne(data, col_tableone, categorical_tableone)
# print(mytable.tabulate(tablefmt="grid"))