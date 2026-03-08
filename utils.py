import torch
import numpy as np
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.nn as nn
import math
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

class SplitData(BaseEstimator, TransformerMixin):
    def __init__(self, dataset):
        super(SplitData, self).__init__()
        self.dataset = dataset
        self.scaler = MinMaxScaler()
        self._is_fitted = False

    def _split_xy(self, X, labels):
        if self.dataset == 'nsl':
            y = (X[labels] != 'normal').astype('float32').to_numpy()
            X_ = X.drop(['labels5', 'labels2'], axis=1)

        elif self.dataset == 'unsw':
            y = X[labels].astype('float32').to_numpy()
            X_ = X.drop('label', axis=1)

        else:
            raise ValueError("Unsupported dataset type")

        return X_, y

    def fit(self, X, labels, y=None):
        X_, _ = self._split_xy(X, labels)
        self.scaler.fit(X_)
        self._is_fitted = True
        return self

    def transform(self, X, labels, one_hot_label=True):
        X_, y_ = self._split_xy(X, labels)

        if not self._is_fitted:
            self.fit(X, labels)

        x_ = self.scaler.transform(X_)
        return x_, y_

def description(data):
    print("Number of samples(examples) ",data.shape[0]," Number of features",data.shape[1])
    print("Dimension of data set ",data.shape)

class AE(nn.Module):
    def __init__(self, input_dim):
        super(AE, self).__init__()

        # Find the nearest power of 2 to input_dim
        nearest_power_of_2 = 2 ** round(math.log2(input_dim))

        # Calculate the dimensions of the 2nd/4th layer and the 3rd layer.
        second_fourth_layer_size = nearest_power_of_2 // 2  # A half
        third_layer_size = nearest_power_of_2 // 4         # A quarter

        # Create encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, third_layer_size),
        )

        # Create decoder
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(third_layer_size, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, input_dim),
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

class CRCLoss(nn.Module):
    def __init__(self, device, temperature=0.1, scale_by_temperature=True):
        super(CRCLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1)

        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        normal_mask = labels == 0
        abnormal_mask = labels > 0

        # CRC loss 至少需要: 2 个 normal（形成正对）+ 1 个 abnormal（形成负对）
        if normal_mask.sum() < 2 or abnormal_mask.sum() < 1:
            return features.sum() * 0.0

        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        normal_logits = logits[normal_mask]  # [n_normal, batch_size]

        # 去掉 self-pair
        self_mask = torch.eye(batch_size, device=features.device, dtype=torch.bool)[normal_mask]
        positive_mask = normal_mask.unsqueeze(0).expand(normal_logits.size(0), -1) & (~self_mask)
        negative_mask = abnormal_mask.unsqueeze(0).expand(normal_logits.size(0), -1)

        positive_logits = normal_logits[positive_mask]
        negative_logits = normal_logits[negative_mask]

        sum_neg = torch.exp(negative_logits).sum().clamp_min(1e-12)
        log_denom = torch.logaddexp(positive_logits, torch.log(sum_neg))
        loss = -(positive_logits - log_denom)

        if self.scale_by_temperature:
            loss = loss * self.temperature

        return loss.mean()
    
def score_detail(y_test,y_test_pred,if_print=False):
    # Confusion matrix
    if if_print == True:
        print("Confusion matrix")
        print(confusion_matrix(y_test, y_test_pred))
        # Accuracy 
        print('Accuracy ',accuracy_score(y_test, y_test_pred))
        # Precision 
        print('Precision ',precision_score(y_test, y_test_pred))
        # Recall
        print('Recall ',recall_score(y_test, y_test_pred))
        # F1 score
        print('F1 score ',f1_score(y_test,y_test_pred))

    return accuracy_score(y_test, y_test_pred), precision_score(y_test, y_test_pred), recall_score(y_test, y_test_pred), f1_score(y_test,y_test_pred)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def _normalized_outputs(model, x):
    features, recon = model(x)
    return F.normalize(features, p=2, dim=1), F.normalize(recon, p=2, dim=1)


def _fit_gmm_from_scores(all_scores, normal_scores, abnormal_scores):
    all_scores = np.asarray(all_scores, dtype=np.float64).reshape(-1, 1)
    normal_scores = np.asarray(normal_scores, dtype=np.float64).reshape(-1)
    abnormal_scores = np.asarray(abnormal_scores, dtype=np.float64).reshape(-1)

    means_init = np.array([
        [normal_scores.mean()],
        [abnormal_scores.mean()],
    ], dtype=np.float64)

    gmm = GaussianMixture(
        n_components=2,
        covariance_type='full',
        reg_covar=1e-6,
        means_init=means_init,
        weights_init=np.array([0.5, 0.5], dtype=np.float64),
        random_state=0,
        max_iter=200,
    )
    gmm.fit(all_scores)

    means = gmm.means_.reshape(-1)
    normal_idx = int(np.argmax(means))   # cosine 越大越像 normal
    abnormal_idx = 1 - normal_idx
    return gmm, normal_idx, abnormal_idx


def _predict_with_gmm(gmm, normal_idx, abnormal_idx, test_scores):
    test_scores = np.asarray(test_scores, dtype=np.float64).reshape(-1, 1)
    probs = gmm.predict_proba(test_scores)

    p_normal = probs[:, normal_idx]
    p_abnormal = probs[:, abnormal_idx]

    y_pred = (p_abnormal > p_normal).astype(np.int32)
    confidence = np.abs(p_abnormal - p_normal).astype(np.float32)
    return y_pred, confidence

def select_high_confidence(confidence, threshold=0.55, min_keep_ratio=0.30):
    confidence = np.asarray(confidence, dtype=np.float32)
    n = confidence.shape[0]

    if n == 0:
        return np.array([], dtype=bool)

    min_keep = max(1, int(np.ceil(min_keep_ratio * n)))

    keep_mask = confidence >= threshold

    # 如果阈值筛出来太少，就至少保留 top-k
    if keep_mask.sum() < min_keep:
        topk_idx = np.argsort(confidence)[-min_keep:]
        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[topk_idx] = True

    return keep_mask

def evaluate(normal_temp, normal_recon_temp, x_train, y_train, x_test, y_test, model, get_confidence=False, en_or_de=False):
    y_train = y_train.view(-1)
    normal_mask = y_train == 0
    abnormal_mask = y_train == 1

    if normal_mask.sum().item() == 0 or abnormal_mask.sum().item() == 0:
        raise ValueError("y_train 里必须同时包含 normal(0) 和 abnormal(1) 才能做双高斯拟合。")

    was_training = model.training
    model.eval()

    with torch.no_grad():
        train_features, train_recon = _normalized_outputs(model, x_train)
        test_features, test_recon = _normalized_outputs(model, x_test)

        normal_temp = F.normalize(normal_temp.view(1, -1), p=2, dim=1)
        normal_recon_temp = F.normalize(normal_recon_temp.view(1, -1), p=2, dim=1)

        values_features_all = F.cosine_similarity(train_features, normal_temp, dim=1).cpu().numpy()
        values_features_normal = F.cosine_similarity(train_features[normal_mask], normal_temp, dim=1).cpu().numpy()
        values_features_abnormal = F.cosine_similarity(train_features[abnormal_mask], normal_temp, dim=1).cpu().numpy()
        values_features_test = F.cosine_similarity(test_features, normal_temp, dim=1).cpu().numpy()

        values_recon_all = F.cosine_similarity(train_recon, normal_recon_temp, dim=1).cpu().numpy()
        values_recon_normal = F.cosine_similarity(train_recon[normal_mask], normal_recon_temp, dim=1).cpu().numpy()
        values_recon_abnormal = F.cosine_similarity(train_recon[abnormal_mask], normal_recon_temp, dim=1).cpu().numpy()
        values_recon_test = F.cosine_similarity(test_recon, normal_recon_temp, dim=1).cpu().numpy()

    if was_training:
        model.train()

    gmm_en, en_normal_idx, en_abnormal_idx = _fit_gmm_from_scores(
        values_features_all,
        values_features_normal,
        values_features_abnormal,
    )
    y_test_pred_2, y_test_pro_en = _predict_with_gmm(
        gmm_en,
        en_normal_idx,
        en_abnormal_idx,
        values_features_test,
    )

    gmm_de, de_normal_idx, de_abnormal_idx = _fit_gmm_from_scores(
        values_recon_all,
        values_recon_normal,
        values_recon_abnormal,
    )
    y_test_pred_4, y_test_pro_de = _predict_with_gmm(
        gmm_de,
        de_normal_idx,
        de_abnormal_idx,
        values_recon_test,
    )

    y_test_pred_no_vote = np.where(
        y_test_pro_en > y_test_pro_de,
        y_test_pred_2,
        y_test_pred_4,
    ).astype(np.int32)

    final_confidence = np.maximum(y_test_pro_en, y_test_pro_de).astype(np.float32)

    if not isinstance(y_test, int):
        if torch.is_tensor(y_test):
            y_test = y_test.detach().cpu().numpy()

        result_encoder = score_detail(y_test, y_test_pred_2)
        result_decoder = score_detail(y_test, y_test_pred_4)
        result_final = score_detail(y_test, y_test_pred_no_vote, if_print=True)

        if get_confidence:
            return result_encoder, result_decoder, result_final, final_confidence

        return result_encoder, result_decoder, result_final

    if get_confidence:
        return y_test_pred_no_vote, final_confidence

    return y_test_pred_no_vote
    num_of_layer = 0

    x_train_normal = x_train[(y_train == 0).squeeze()]
    x_train_abnormal = x_train[(y_train == 1).squeeze()]

    train_features = F.normalize(model(x_train)[num_of_layer], p=2, dim=1)
    train_features_normal = F.normalize(model(x_train_normal)[num_of_layer], p=2, dim=1)
    train_features_abnormal = F.normalize(model(x_train_abnormal)[num_of_layer], p=2, dim=1)
    test_features = F.normalize(model(x_test)[num_of_layer], p=2, dim=1)

    values_features_all, indcies = torch.sort(F.cosine_similarity(train_features, normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))
    values_features_normal, indcies = torch.sort(F.cosine_similarity(train_features_normal, normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))
    values_features_abnormal, indcies = torch.sort(F.cosine_similarity(train_features_abnormal, normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))

    values_features_all = values_features_all.cpu().detach().numpy()

    values_features_test = F.cosine_similarity(test_features, normal_temp.reshape([-1, normal_temp.shape[0]]))

    num_of_output = 1
    train_recon = F.normalize(model(x_train)[num_of_output], p=2, dim=1)
    train_recon_normal = F.normalize(model(x_train_normal)[num_of_output], p=2, dim=1)
    train_recon_abnormal = F.normalize(model(x_train_abnormal)[num_of_output], p=2, dim=1)
    test_recon = F.normalize(model(x_test)[num_of_output], p=2, dim=1)

    values_recon_all, indcies = torch.sort(F.cosine_similarity(train_recon, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))
    values_recon_normal, indcies = torch.sort(F.cosine_similarity(train_recon_normal, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))
    values_recon_abnormal, indcies = torch.sort(F.cosine_similarity(train_recon_abnormal, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))

    values_recon_all = values_recon_all.cpu().detach().numpy()

    values_recon_test = F.cosine_similarity(test_recon, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1)

    mu1_initial = np.mean(values_features_normal.cpu().detach().numpy())
    sigma1_initial = np.std(values_features_normal.cpu().detach().numpy())

    mu2_initial = np.mean(values_features_abnormal.cpu().detach().numpy())
    sigma2_initial = np.std(values_features_abnormal.cpu().detach().numpy())

    # Fitting data to two Gaussian distributions using Maximum Likelihood Estimation (MLE)
    initial_params = np.array([mu1_initial, sigma1_initial, mu2_initial, sigma2_initial]) # Initial parameters
    result = opt.minimize(log_likelihood, initial_params, args=(values_features_all,), method='Nelder-Mead')
    mu1_fit, sigma1_fit, mu2_fit, sigma2_fit = result.x # Estimated parameter values

    if mu1_fit > mu2_fit:
        gaussian1 = dist.Normal(mu1_fit, sigma1_fit)
        gaussian2 = dist.Normal(mu2_fit, sigma2_fit)
    else:
        gaussian2 = dist.Normal(mu1_fit, sigma1_fit)
        gaussian1 = dist.Normal(mu2_fit, sigma2_fit)

    pdf1 = gaussian1.log_prob(values_features_test).exp()

    pdf2 = gaussian2.log_prob(values_features_test).exp()
    y_test_pred_2 = (pdf2 > pdf1).cpu().numpy().astype("int32")
    y_test_pro_en = (torch.abs(pdf2-pdf1)).cpu().detach().numpy().astype("float32")

    if isinstance(y_test, int) == False:
        if y_test.device != torch.device("cpu"):
            y_test = y_test.cpu().numpy()

    mu3_initial = np.mean(values_recon_normal.cpu().detach().numpy())
    sigma3_initial = np.std(values_recon_normal.cpu().detach().numpy())

    mu4_initial = np.mean(values_recon_abnormal.cpu().detach().numpy())
    sigma4_initial = np.std(values_recon_abnormal.cpu().detach().numpy())

    # Fitting data to two Gaussian distributions using Maximum Likelihood Estimation (MLE)
    initial_params = np.array([mu3_initial, sigma3_initial, mu4_initial, sigma4_initial]) # Initial parameters
    result = opt.minimize(log_likelihood, initial_params, args=(values_recon_all,), method='Nelder-Mead')
    mu3_fit, sigma3_fit, mu4_fit, sigma4_fit = result.x # Estimated parameter values

    if mu3_fit > mu4_fit:
        gaussian3 = dist.Normal(mu3_fit, sigma3_fit)
        gaussian4 = dist.Normal(mu4_fit, sigma4_fit)
    else:
        gaussian4 = dist.Normal(mu3_fit, sigma3_fit)
        gaussian3 = dist.Normal(mu4_fit, sigma4_fit)

    pdf3 = gaussian3.log_prob(values_recon_test).exp()

    pdf4 = gaussian4.log_prob(values_recon_test).exp()
    y_test_pred_4 = (pdf4 > pdf3).cpu().numpy().astype("int32")
    y_test_pro_de = (torch.abs(pdf4-pdf3)).cpu().detach().numpy().astype("float32")

    if not isinstance(y_test, int):
        if y_test.device != torch.device("cpu"):
            y_test = y_test.cpu().numpy()
        result_encoder = score_detail(y_test, y_test_pred_2)
        result_decoder = score_detail(y_test, y_test_pred_4)

    y_test_pred_no_vote = torch.where(torch.from_numpy(y_test_pro_en) > torch.from_numpy(y_test_pro_de), torch.from_numpy(y_test_pred_2), torch.from_numpy(y_test_pred_4))
    
    if not isinstance(y_test, int):
        result_final = score_detail(y_test, y_test_pred_no_vote, if_print=True)
        return result_encoder, result_decoder, result_final
    else:
        return y_test_pred_no_vote
