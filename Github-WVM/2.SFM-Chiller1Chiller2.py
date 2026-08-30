import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import sys
import random
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

TRAIN_FILE_PATH = "Train Set.csv"
TEST_FILE_PATH = "Test Set.csv"
MAIN_OUTPUT_DIR = "./2.WVM-Chiller1Chiller2"
BEST_RESULT_DIR = os.path.join(MAIN_OUTPUT_DIR, "Result")
os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True)
os.makedirs(BEST_RESULT_DIR, exist_ok=True)

COND_VARS = ['TCI', 'TEO', 'Capacity']
STATE_VARS = ['TCO-TCI', 'TCA', 'TSS', 'TEI-TEO', 'TEA', "kW"]
TRAIN_LABEL_VAR = 'pseudo-label'
TEST_LABEL_VAR = 'ture-label'
polarity = 1.0

TRAIN_PERIOD_NAMES, TEST_PERIOD_NAMES = ["Cycle1"], ["Cycle2"]
ALL_PERIODS = TRAIN_PERIOD_NAMES + TEST_PERIOD_NAMES
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VIB_BETA_CANDIDATES = [0.0001, 0.0005, 0.001, 0.01, 0.1]
VIB_LR_CANDIDATES = [0.0001, 0.0005, 0.001, 0.01, 0.1]
MLP_LR_CANDIDATES = [0.0001, 0.0005, 0.001, 0.01, 0.1]
CUSUM_K_CANDIDATES = [0.2, 0.3, 0.4, 0.5]
VIB_LATENT_DIM, VIB_EPOCHS, VIB_BATCH_SIZE = 64, 100, 512
FE_LATENT_DIM, FE_EPOCHS, FE_BATCH_SIZE = 64, 100, 512

cusum_warn_candidates = [57]
cusum_maint_candidates = [95]

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.switch_backend('Agg')
SAVE_DPI = 300

print("[1/6] Executing: Data Preprocessing")
preprocessing_output_dir = os.path.join(MAIN_OUTPUT_DIR, '1_Preprocessing_Output')
os.makedirs(preprocessing_output_dir, exist_ok=True)

FILE_PATHS_DICT = {
    "Cycle1": TRAIN_FILE_PATH,
    "Cycle2": TEST_FILE_PATH
}
global_index_counter = 0
df_list = []
DATA_PERIODS = {}
for period_name, file_path in FILE_PATHS_DICT.items():
    if not os.path.exists(file_path):
        sys.exit(f"❌ Error: Data file not found at {file_path}, please verify the path!")
    df_temp = pd.read_csv(file_path, encoding='utf-8')

    if period_name == "Cycle1":
        if TRAIN_LABEL_VAR not in df_temp.columns:
            sys.exit(f"❌ Error: Column [{TRAIN_LABEL_VAR}] not found in Cycle1 data")
        df_temp['unified_label'] = df_temp[TRAIN_LABEL_VAR]
    elif period_name == "Cycle2":
        if TEST_LABEL_VAR not in df_temp.columns:
            sys.exit(f"❌ Error: Column [{TEST_LABEL_VAR}] not found in Cycle2 data")
        df_temp['unified_label'] = df_temp[TEST_LABEL_VAR]

    if 'Time' in df_temp.columns:
        df_temp['Original_Time'] = df_temp['Time']
    elif '时间' in df_temp.columns:
        df_temp['Original_Time'] = df_temp['时间']
    else:
        df_temp['Original_Time'] = df_temp.index

    time_values = np.arange(global_index_counter, global_index_counter + len(df_temp) * 2, 2)
    df_temp['Time'] = time_values
    DATA_PERIODS[period_name] = (time_values[0], time_values[-1])
    global_index_counter = time_values[-1] + 2

    df_list.append(df_temp)

df_active = pd.concat(df_list).sort_values('Time').reset_index(drop=True)

train_mask = df_active['Time'].between(*DATA_PERIODS["Cycle1"])
df_train_basis = df_active[train_mask]
vars_to_norm = COND_VARS + STATE_VARS

train_min = df_train_basis[vars_to_norm].min()
train_max = df_train_basis[vars_to_norm].max()
train_range = train_max - train_min
train_range[train_range == 0] = 1.0

df_normalized = df_active.copy()
df_normalized[vars_to_norm] = (df_normalized[vars_to_norm] - train_min) / train_range

np.save(os.path.join(preprocessing_output_dir, 'X_c_steady_points.npy'), df_normalized[COND_VARS].values)
np.save(os.path.join(preprocessing_output_dir, 'Y_c_steady_points.npy'), df_normalized[STATE_VARS].values)
np.save(os.path.join(preprocessing_output_dir, 'T_steady_points.npy'), df_normalized['Time'].values)
np.save(os.path.join(preprocessing_output_dir, 'Labels_steady_points.npy'), df_active['unified_label'].values)

all_labels = np.load(os.path.join(preprocessing_output_dir, 'Labels_steady_points.npy'))
all_timestamps = np.load(os.path.join(preprocessing_output_dir, 'T_steady_points.npy'))

if 'ture-label' not in df_active.columns:
    sys.exit("❌ Error: Column [ture-label] not found in dataset, please check the file!")
eval_labels = df_active['ture-label'].values

df_temp = pd.DataFrame({'idx': np.arange(len(all_labels)), 'ts': all_timestamps})


class VIBResidualGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu, self.fc_logvar = nn.Linear(64, latent_dim), nn.Linear(64, latent_dim)
        self.predictor = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(),
                                       nn.Linear(128, output_dim))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(mu.device)
        return mu + std * eps

    def forward(self, x):
        h = F.relu(self.fc2(F.relu(self.fc1(x))))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return self.predictor(self.reparameterize(mu, logvar)), mu, logvar


def vib_loss(pred, target, mu, logvar, beta):
    recon_loss = F.mse_loss(pred, target, reduction='mean')
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon_loss + beta * kl_loss


class FeatureExtractorMLP(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(),
                                      nn.Linear(128, latent_dim))
        self.head = nn.Linear(latent_dim, 1)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat).squeeze(-1), feat


def calc_weighted_hi(feats, pca, explained_var): 
    return np.dot(pca.transform(feats), explained_var)


def calc_cusum(hi, mu0, sigma0, k):
    curr, s = 0.0, np.zeros(len(hi))
    for i in range(len(hi)):
        curr = max(0, curr + (mu0 - hi[i]) - k)
        s[i] = curr
    return s


def calc_lead_time(pred, times):
    if len(pred) < 5: 
        return 0.0

    maint_time = times[-1]
    first_warn_idx = -1
    found = False

    for i in range(len(pred) - 4):
        if np.sum(pred[i:i + 5]) == 5:
            first_warn_idx = i
            found = True
            break

    if found:
        return times[first_warn_idx] - maint_time
    else:
        return 0.0


def calc_twarn(pred, times, dates):
    if len(pred) == 0:
        return None, None, "Fault warning not triggered for 5 consecutive days"
    
    try:
        date_series = pd.to_datetime(dates).dt.date
    except AttributeError:
        date_series = pd.Series(dates).apply(lambda x: str(x).split(' ')[0] if pd.notnull(x) else "")
        
    start_idx = None
    for i in range(len(pred)):
        if pred[i] == 1:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                unique_days = date_series.iloc[start_idx:i].nunique()
                if unique_days >= 5:
                    return start_idx, times[start_idx], dates[start_idx]
                start_idx = None
                
    if start_idx is not None:
        unique_days = date_series.iloc[start_idx:].nunique()
        if unique_days >= 5:
            return start_idx, times[start_idx], dates[start_idx]
            
    return None, None, "Fault warning not triggered for 5 consecutive days"


def evaluate_threshold(cusum_warn_p, cusum_maint_p, train_cusum_val, sigma0, mu0,
                       period_feats, polarity, pca, explained_var, eval_labels, df_temp, target_periods, cusum_k):  
    cusum_th_warn = np.percentile(train_cusum_val, cusum_warn_p)
    cusum_th_maint = np.percentile(train_cusum_val, cusum_maint_p)

    cusum_acc_list, cusum_rec_list, cusum_fpr_list, cusum_lead_list = [], [], [], []

    for pname in target_periods:
        period_mask = df_temp['ts'].between(*DATA_PERIODS[pname])
        idx = df_temp[period_mask]['idx'].values
        curr_ts = df_temp[period_mask]['ts'].values
        feats = period_feats[pname]
        
        curr_labels = eval_labels[idx]
        hi_val = calc_weighted_hi(feats, pca, explained_var) * polarity
        y_true = (curr_labels == 1).astype(int)

        cusum_val = calc_cusum(hi_val, mu0, sigma0, k=cusum_k)
        cusum_pred = (cusum_val >= cusum_th_warn).astype(int)
        c_acc = accuracy_score(y_true, cusum_pred)
        c_rec = np.mean(cusum_pred[y_true == 1] == 1) if np.sum(y_true == 1) > 0 else 1.0
        c_fpr = np.mean(cusum_pred[y_true == 0] == 1) if np.sum(y_true == 0) > 0 else 0.0
        c_lead = calc_lead_time(cusum_pred, curr_ts)
        cusum_acc_list.append(c_acc)
        cusum_rec_list.append(c_rec)
        cusum_fpr_list.append(c_fpr)
        cusum_lead_list.append(c_lead)

    return {
        'cusum_warn_p': cusum_warn_p, 'cusum_maint_p': cusum_maint_p,
        'avg_acc': np.mean(cusum_acc_list), 'avg_rec': np.mean(cusum_rec_list),
        'avg_fpr': np.mean(cusum_fpr_list), 'avg_lead': np.mean(cusum_lead_list)
    }


print("[2/6] Starting Bi-level Optimization: Hyperparameters + Thresholds (Optimizing for minor faults, please wait...)")
global_best_result = None
global_best_score = -1
X_np = np.load(os.path.join(preprocessing_output_dir, 'X_c_steady_points.npy'))
Y_np = np.load(os.path.join(preprocessing_output_dir, 'Y_c_steady_points.npy'))

train_idx = df_temp[df_temp['ts'].between(*DATA_PERIODS["Cycle1"])]["idx"].values
X_train_np = X_np[train_idx]
Y_train_np = Y_np[train_idx]
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).to(DEVICE)
Y_train_tensor = torch.tensor(Y_train_np, dtype=torch.float32).to(DEVICE)
train_dataloader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=VIB_BATCH_SIZE, shuffle=True)

for VIB_BETA in VIB_BETA_CANDIDATES:
    for VIB_LR in VIB_LR_CANDIDATES:
        for MLP_LR in MLP_LR_CANDIDATES:
            set_seed(42)
            for CUSUM_K in CUSUM_K_CANDIDATES:
                print(
                    f"\n===== Current Configuration: VIB_BETA={VIB_BETA}, VIB_LR={VIB_LR}, MLP_LR={MLP_LR}, CUSUM_K={CUSUM_K} =====")

                model = VIBResidualGenerator(X_train_np.shape[1], Y_train_np.shape[1], VIB_LATENT_DIM).to(DEVICE)
                optimizer = optim.Adam(model.parameters(), lr=VIB_LR)
                min_loss = float('inf')
                best_vib_path = os.path.join(MAIN_OUTPUT_DIR, f'vib_beta_{VIB_BETA}_lr_{VIB_LR}.pth')
                for epoch in range(VIB_EPOCHS):
                    train_loss = 0
                    batch_count = 0
                    for bx, by in train_dataloader:
                        optimizer.zero_grad()
                        pred, mu, logvar = model(bx)
                        loss = vib_loss(pred, by, mu, logvar, VIB_BETA)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        batch_count += 1
                    avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
                    if avg_train_loss < min_loss:
                        min_loss = avg_train_loss
                        torch.save(model.state_dict(), best_vib_path)

                best_vib_model = VIBResidualGenerator(X_train_np.shape[1], Y_train_np.shape[1], VIB_LATENT_DIM).to(DEVICE)
                best_vib_model.load_state_dict(torch.load(best_vib_path))
                best_vib_model.eval()

                X_tensor = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
                Y_tensor = torch.tensor(Y_np, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    recon_Y, _, _ = best_vib_model(X_tensor)
                    residuals_all = (Y_tensor - recon_Y).cpu().numpy()

                train_residuals = residuals_all[train_idx]
                res_min = np.min(train_residuals, axis=0)
                res_max = np.max(train_residuals, axis=0)
                res_range = res_max - res_min
                res_range[res_range == 0] = 1.0
                train_residuals_norm = (train_residuals - res_min) / res_range

                best_mlp_path = os.path.join(MAIN_OUTPUT_DIR, f'mlp_lr_{MLP_LR}_vib_beta_{VIB_BETA}.pth')
                fe_model = FeatureExtractorMLP(train_residuals_norm.shape[1], FE_LATENT_DIM).to(DEVICE)
                mlp_optimizer = optim.Adam(fe_model.parameters(), lr=MLP_LR)
                mlp_criterion = nn.MSELoss()

                mlp_min_loss = float('inf')
                train_res_tensor = torch.tensor(train_residuals_norm, dtype=torch.float32).to(DEVICE)
                train_labels_tensor = torch.tensor(all_labels[train_idx], dtype=torch.float32).to(DEVICE)
                mlp_train_loader = DataLoader(
                    TensorDataset(train_res_tensor, train_labels_tensor),
                    batch_size=FE_BATCH_SIZE,
                    shuffle=True
                )

                for epoch in range(FE_EPOCHS):
                    fe_model.train()
                    mlp_train_loss = 0
                    mlp_batch_count = 0
                    for bx, by in mlp_train_loader:
                        mlp_optimizer.zero_grad()
                        pred, _ = fe_model(bx)
                        loss = mlp_criterion(pred, by)
                        loss.backward()
                        mlp_optimizer.step()
                        mlp_train_loss += loss.item()
                        mlp_batch_count += 1
                    avg_mlp_loss = mlp_train_loss / mlp_batch_count if mlp_batch_count > 0 else 0
                    if avg_mlp_loss < mlp_min_loss:
                        mlp_min_loss = avg_mlp_loss
                        torch.save(fe_model.state_dict(), best_mlp_path)

                fe_model.load_state_dict(torch.load(best_mlp_path))
                fe_model.eval()
                period_feats = {}
                for p in ALL_PERIODS:
                    idx = df_temp[df_temp['ts'].between(*DATA_PERIODS[p])]['idx'].values
                    res_period = residuals_all[idx]
                    res_period_norm = (res_period - res_min) / res_range
                    with torch.no_grad():
                        feats = fe_model(torch.tensor(res_period_norm).float().to(DEVICE))[1].cpu().numpy()
                    period_feats[p] = feats

                train_feats = period_feats["Cycle1"]
                pca = PCA(n_components=0.95)
                pca.fit(train_feats)
                explained_var = pca.explained_variance_ratio_
                train_hi_raw = calc_weighted_hi(train_feats, pca, explained_var)
                test_feats = period_feats["Cycle2"]
                test_hi = calc_weighted_hi(test_feats, pca, explained_var)
                corr, _ = pearsonr(train_hi_raw, np.arange(len(train_hi_raw)))
                train_hi = train_hi_raw * polarity
                mu0, sigma0 = np.mean(train_hi), np.std(train_hi)
                print(
                    f"[HI Inversion Check]: Pearson corr={corr:.4f} | polarity={polarity} | {'Inverted' if polarity == -1.0 else 'Not Inverted'}")

                train_cusum_val = calc_cusum(train_hi, mu0, sigma0, CUSUM_K)
                search_results = []
                for c_warn in cusum_warn_candidates:
                    for c_maint in cusum_maint_candidates:
                        if c_maint <= c_warn: continue

                        res = evaluate_threshold(c_warn, c_maint, train_cusum_val, sigma0,
                                                 mu0, period_feats, polarity, pca,
                                                 explained_var, eval_labels, df_temp, TRAIN_PERIOD_NAMES,
                                                 CUSUM_K)
                        search_results.append(res)

                search_results.sort(key=lambda x: (-x['avg_rec'], -x['avg_acc'], x['avg_fpr']))
                curr_best = search_results[0]
                curr_best['VIB_BETA'] = VIB_BETA
                curr_best['VIB_LR'] = VIB_LR
                curr_best['MLP_LR'] = MLP_LR
                curr_best['CUSUM_K'] = CUSUM_K
                curr_score = curr_best['avg_rec'] * 0.7 + curr_best['avg_acc'] * 0.3
                print(
                    f"Optimal Metrics (Train Set): Recall={curr_best['avg_rec']:.4f}, Accuracy={curr_best['avg_acc']:.4f}, FPR={curr_best['avg_fpr']:.4f}")

                if curr_score > global_best_score:
                    global_best_score = curr_score
                    global_best_result = curr_best
                    global_best_result['best_vib_path'] = best_vib_path
                    global_best_result['best_mlp_path'] = best_mlp_path
                    global_best_result['pca'] = pca
                    global_best_result['explained_var'] = explained_var
                    global_best_result['mu0'] = mu0
                    global_best_result['sigma0'] = sigma0
                    global_best_result['polarity'] = polarity
                    global_best_result['period_feats'] = period_feats

print("\n" + "=" * 100)
print("🎉 Bi-level Optimization Completed (Evaluated on Training Set)")
print(
    f"📌 Optimal Model Hyperparameters -> VIB_BETA={global_best_result['VIB_BETA']}, VIB_LR={global_best_result['VIB_LR']}, MLP_LR={global_best_result['MLP_LR']}")
print(f"📌 Optimal Sensitivity Factor -> CUSUM_K={global_best_result['CUSUM_K']}")

print("[3/6] Generating Weak Label Visualizations...")
best_vib_model = VIBResidualGenerator(X_np.shape[1], Y_np.shape[1], VIB_LATENT_DIM).to(DEVICE)
best_vib_model.load_state_dict(torch.load(global_best_result['best_vib_path']))
best_vib_model.eval()
with torch.no_grad():
    residuals_all_best = (Y_tensor - best_vib_model(X_tensor)[0]).cpu().numpy()

train_idx = df_temp[df_temp['ts'].between(*DATA_PERIODS["Cycle1"])]["idx"].values
normal_res_best = residuals_all_best[train_idx]
res_min_best = np.min(normal_res_best, axis=0)
res_max_best = np.max(normal_res_best, axis=0)
res_range_best = res_max_best - res_min_best
res_range_best[res_range_best == 0] = 1.0
residuals_all_best = (residuals_all_best - res_min_best) / res_range_best

best_mlp_model = FeatureExtractorMLP(residuals_all_best.shape[1], FE_LATENT_DIM).to(DEVICE)
best_mlp_model.load_state_dict(torch.load(global_best_result['best_mlp_path']))
best_mlp_model.eval()

period_pred_labels_best = {}
period_true_labels_best = {}
for p in ALL_PERIODS:
    idx = df_temp[df_temp['ts'].between(*DATA_PERIODS[p])]['idx'].values
    with torch.no_grad():
        pred_labels_best, _ = best_mlp_model(torch.tensor(residuals_all_best[idx]).float().to(DEVICE))
        pred_labels_best = pred_labels_best.cpu().numpy()
    true_labels_best = all_labels[idx]
    period_pred_labels_best[p] = pred_labels_best
    period_true_labels_best[p] = true_labels_best

vis_output_dir_best = os.path.join(BEST_RESULT_DIR, 'Optimal_Weak_Label_Visualization')
os.makedirs(vis_output_dir_best, exist_ok=True)

for period_name in ALL_PERIODS:
    pred_vals = period_pred_labels_best[period_name]
    true_vals = period_true_labels_best[period_name]
    sample_num = np.arange(len(pred_vals))

    plt.figure(figsize=(15, 6))
    plt.plot(sample_num, pred_vals, label='Predicted pseudo-label', color='#1f77b4', linewidth=1.2)
    label_txt = 'True pseudo-label' if period_name == "Cycle1" else 'True Week Label'
    plt.scatter(sample_num, true_vals, label=label_txt, color='#ff4757', s=8, alpha=0.6)

    plt.title(f'MLP-Predicted pseudo-label ({period_name})', fontsize=14, pad=15)
    plt.xlabel('Samples', fontsize=12)
    plt.ylabel('Label Value', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    save_path = os.path.join(vis_output_dir_best, f'{period_name}_mlp_weak_label_pred_best.png')
    plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()

print(f"✅ Weak label visualization completed. Saved to: {vis_output_dir_best}")

best_train_feats = global_best_result['period_feats']['Cycle1']
best_train_hi = calc_weighted_hi(best_train_feats, global_best_result['pca'], global_best_result['explained_var']) * \
                global_best_result['polarity']
best_train_cusum_val = calc_cusum(best_train_hi, global_best_result['mu0'], global_best_result['sigma0'],
                                  global_best_result['CUSUM_K'])

best_test_feats = global_best_result['period_feats']['Cycle2']
best_test_hi = calc_weighted_hi(best_test_feats, global_best_result['pca'], global_best_result['explained_var']) * \
               global_best_result['polarity']

train_spearman, _ = spearmanr(best_train_hi, np.arange(len(best_train_hi)))
test_spearman, _ = spearmanr(best_test_hi, np.arange(len(best_test_hi)))

test_res = evaluate_threshold(
    global_best_result['cusum_warn_p'], global_best_result['cusum_maint_p'],
    best_train_cusum_val,
    global_best_result['sigma0'],
    global_best_result['mu0'],
    global_best_result['period_feats'],
    global_best_result['polarity'],
    global_best_result['pca'],
    global_best_result['explained_var'],
    eval_labels,
    df_temp,
    TEST_PERIOD_NAMES,
    global_best_result['CUSUM_K']
)

train_cusum_val = calc_cusum(calc_weighted_hi(
    global_best_result['period_feats']['Cycle1'], global_best_result['pca'], global_best_result['explained_var']) *
                             global_best_result['polarity'],
                             global_best_result['mu0'], global_best_result['sigma0'], global_best_result['CUSUM_K'])
best_cusum_warn = np.percentile(train_cusum_val, global_best_result['cusum_warn_p'])
best_cusum_maint = np.percentile(train_cusum_val, global_best_result['cusum_maint_p'])

idx_train_tw = df_temp[df_temp['ts'].between(*DATA_PERIODS["Cycle1"])]['idx'].values
time_train_tw = np.arange(0, len(best_train_cusum_val) * 2, 2)
pred_train_tw = (best_train_cusum_val >= best_cusum_warn).astype(int)
date_train_tw = df_active.iloc[idx_train_tw]['Original_Time'].values if 'Original_Time' in df_active.columns else np.full(len(time_train_tw), "")
train_tw_idx, train_tw_time, train_tw_date = calc_twarn(pred_train_tw, time_train_tw, date_train_tw)

best_test_cusum_val = calc_cusum(best_test_hi, global_best_result['mu0'], global_best_result['sigma0'], global_best_result['CUSUM_K'])
idx_test_tw = df_temp[df_temp['ts'].between(*DATA_PERIODS["Cycle2"])]['idx'].values
time_test_tw = np.arange(0, len(best_test_cusum_val) * 2, 2)
pred_test_tw = (best_test_cusum_val >= best_cusum_warn).astype(int)
date_test_tw = df_active.iloc[idx_test_tw]['Original_Time'].values if 'Original_Time' in df_active.columns else np.full(len(time_test_tw), "")
test_tw_idx, test_tw_time, test_tw_date = calc_twarn(pred_test_tw, time_test_tw, date_test_tw)

print("-" * 50)
print(f"[Monotonicity] Train HI Spearman Corr={train_spearman:.4f} | Test HI Spearman Corr={test_spearman:.4f}") 
print("📊 [Validation] Test Set (Cycle2) Generalization Metrics:")
print(
    f"[CUSUM] Recall={test_res['avg_rec']:.4f} | Accuracy={test_res['avg_acc']:.4f} | FPR={test_res['avg_fpr']:.4f}")
print(f"        Average Lead Time={test_res['avg_lead']:.2f}")
print(f"        Warning Timestamp Twarn (5 Consecutive Days):")
print(f"        -> Index={test_tw_idx}, Time={test_tw_time}, Date(Twarn)={test_tw_date}")
print("📈 [Baseline] Training Set (Cycle1) Metrics:")
print(
    f"[CUSUM] Recall={global_best_result['avg_rec']:.4f} | Accuracy={global_best_result['avg_acc']:.4f} | FPR={global_best_result['avg_fpr']:.4f}")
print(f"        Average Lead Time={global_best_result['avg_lead']:.2f}")
print(f"        Warning Timestamp Twarn (5 Consecutive Days):")
print(f"        -> Index={train_tw_idx}, Time={train_tw_time}, Date(Twarn)={train_tw_date}")
print("-" * 50)

with open(os.path.join(BEST_RESULT_DIR, 'global_best_params.txt'), 'w', encoding='utf-8') as f:
    f.write("Condenser Fouling Detection - Global Best Parameters (Training Set Optimization)\n")
    f.write(
        f"[Model Hyperparams] VIB_BETA={global_best_result['VIB_BETA']}, VIB_LR={global_best_result['VIB_LR']}, MLP_LR={global_best_result['MLP_LR']}\n")
    f.write(f"[Sensitivity] CUSUM_K={global_best_result['CUSUM_K']}\n")
    f.write(f"[Polarity] POLARITY={global_best_result['polarity']}\n")
    f.write(
        f"[CUSUM Thresholds] Warning Percentile={global_best_result['cusum_warn_p']} -> {best_cusum_warn:.6f}, Maintenance Percentile={global_best_result['cusum_maint_p']} -> {best_cusum_maint:.6f}\n")

with open(os.path.join(BEST_RESULT_DIR, 'global_best_metrics.txt'), 'w', encoding='utf-8') as f:
    f.write("Condenser Fouling Detection - Global Best Evaluation Metrics\n")
    f.write("========================================\n")
    f.write("[Training Set Performance (Optimization Basis)]\n")
    f.write(f"CUSUM Recall: {global_best_result['avg_rec']:.4f}\n")
    f.write(f"CUSUM Accuracy: {global_best_result['avg_acc']:.4f}\n")
    f.write(f"CUSUM False Positive Rate: {global_best_result['avg_fpr']:.4f}\n")
    f.write(f"Average Lead Time: {global_best_result['avg_lead']:.2f}\n")
    f.write(f"Twarn (First Sample of 5-day Consecutive Fault): Index={train_tw_idx}, Time={train_tw_time}, Date={train_tw_date}\n")
    f.write(f"HI vs Time Spearman Correlation: {train_spearman:.4f}\n")
    f.write("\n")
    f.write("[Test Set Performance (Generalization Check)]\n")
    f.write(f"CUSUM Recall: {test_res['avg_rec']:.4f}\n")
    f.write(f"CUSUM Accuracy: {test_res['avg_acc']:.4f}\n")
    f.write(f"CUSUM False Positive Rate: {test_res['avg_fpr']:.4f}\n")
    f.write(f"Average Lead Time: {test_res['avg_lead']:.2f}\n")
    f.write(f"Twarn (First Sample of 5-day Consecutive Fault): Index={test_tw_idx}, Time={test_tw_time}, Date={test_tw_date}\n")
    f.write("-" * 20 + "\n")
    f.write(f"HI vs Time Spearman Correlation: {test_spearman:.4f}\n")

shutil.copy(global_best_result['best_vib_path'], os.path.join(BEST_RESULT_DIR, 'best_vib_model.pth'))
shutil.copy(global_best_result['best_mlp_path'], os.path.join(BEST_RESULT_DIR, 'best_mlp_model.pth'))

print("[Additional] Plotting state variable residual trends...")
res_plot_dir = os.path.join(BEST_RESULT_DIR, 'Optimal_Residual_Trends')
os.makedirs(res_plot_dir, exist_ok=True)
res_train_dir = os.path.join(res_plot_dir, 'Train_Set_Cycle1')
res_test_dir = os.path.join(res_plot_dir, 'Test_Set_Cycle2')
os.makedirs(res_train_dir, exist_ok=True)
os.makedirs(res_test_dir, exist_ok=True)

best_vib_model = VIBResidualGenerator(X_np.shape[1], Y_np.shape[1], VIB_LATENT_DIM).to(DEVICE)
best_vib_model.load_state_dict(torch.load(os.path.join(BEST_RESULT_DIR, 'best_vib_model.pth')))
best_vib_model.eval()
with torch.no_grad():
    recon_y, _, _ = best_vib_model(X_tensor)
    best_raw_residuals = (Y_tensor - recon_y).cpu().numpy()

train_idx_res = df_temp[df_temp['ts'].between(*DATA_PERIODS["Cycle1"])]["idx"].values
test_idx_res = df_temp[df_temp['ts'].between(*DATA_PERIODS["Cycle2"])]["idx"].values

normal_res_subset = best_raw_residuals[train_idx_res]
res_std_final = np.std(normal_res_subset, axis=0)
res_std_final[res_std_final == 0] = 1.0
best_residuals_final = best_raw_residuals / res_std_final

train_residuals = best_residuals_final[train_idx_res]
test_residuals = best_residuals_final[test_idx_res]

for i, var_name in enumerate(STATE_VARS):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(train_residuals[:, i], color='#7a3d99', linewidth=2.0, linestyle='-', alpha=0.8, label='Residual')
    ax.set_xlabel('Samples', fontfamily='Times New Roman', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_ylabel('Residual', fontfamily='Times New Roman', fontsize=13, fontweight='bold', labelpad=12)
    ax.tick_params(axis='x', labelsize=12, labelfontfamily='Times New Roman')
    ax.tick_params(axis='y', labelsize=12, labelfontfamily='Times New Roman')
    legend = ax.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=False,
                       prop={'family': 'Times New Roman'})
    legend.get_frame().set_linewidth(0.8)
    y_min, y_max = ax.get_ylim()
    y_ticks = np.linspace(y_min, y_max, num=6)
    ax.set_yticks(y_ticks)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis='x', which='major', width=0.8)
    ax.tick_params(axis='y', which='major', width=0.8)
    ax.tick_params(axis='both', direction='in')
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(res_train_dir, f'{var_name}_Train_Residual.png'), dpi=600, bbox_inches='tight')
    plt.close()

for i, var_name in enumerate(STATE_VARS):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(test_residuals[:, i], color='#7a3d99', linewidth=2.0, linestyle='-', alpha=0.8, label='Residual')
    ax.set_xlabel('Samples', fontfamily='Times New Roman', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_ylabel('Residual', fontfamily='Times New Roman', fontsize=13, fontweight='bold', labelpad=12)
    ax.tick_params(axis='x', labelsize=12, labelfontfamily='Times New Roman')
    ax.tick_params(axis='y', labelsize=12, labelfontfamily='Times New Roman')
    legend = ax.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=False,
                       prop={'family': 'Times New Roman'})
    legend.get_frame().set_linewidth(0.8)
    y_min, y_max = ax.get_ylim()
    y_ticks = np.linspace(y_min, y_max, num=6)
    ax.set_yticks(y_ticks)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis='x', which='major', width=0.8)
    ax.tick_params(axis='y', which='major', width=0.8)
    ax.tick_params(axis='both', direction='in')
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(res_test_dir, f'{var_name}_Test_Residual.png'), dpi=600, bbox_inches='tight')
    plt.close()

print(f"✅ Residual trend plots generated:")
print(f"   - Training plots saved to: {res_train_dir}")
print(f"   - Testing plots saved to: {res_test_dir}")

plot_dir = os.path.join(BEST_RESULT_DIR, 'Optimal_Monitoring_Plots')
os.makedirs(plot_dir, exist_ok=True)
for pname in ALL_PERIODS:
    period_mask = df_temp['ts'].between(*DATA_PERIODS[pname])
    idx = df_temp[period_mask]['idx'].values
    feats = global_best_result['period_feats'][pname]
    curr_labels = all_labels[idx]
    hi_val = calc_weighted_hi(feats, global_best_result['pca'], global_best_result['explained_var']) * \
             global_best_result['polarity']
    cusum_val = calc_cusum(hi_val, global_best_result['mu0'], global_best_result['sigma0'],
                           global_best_result['CUSUM_K'])
    
    x_plot = np.arange(0, len(cusum_val) * 2, 2)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor('#FFFFFF')
    ax.plot(x_plot, hi_val, color='blue', linewidth=2.0, linestyle='-', alpha=0.8, label='HI')
    ax.set_xlabel('t (min)', fontfamily='Times New Roman', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_ylabel('HI', fontfamily='Times New Roman', fontsize=13, fontweight='bold', labelpad=12)
    ax.tick_params(axis='both', labelsize=12, direction='in')
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
    ax.yaxis.set_major_locator(plt.LinearLocator(numticks=6))
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis='both', width=0.8)
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, f'{pname}_HI.png'), dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor('#FFFFFF')
    ax.plot(x_plot, cusum_val, color='#1f77b4', linewidth=2.0, linestyle='-', alpha=0.8, zorder=2)

    ax.axhline(y=best_cusum_warn, color='#FF0000', linestyle='--', linewidth=2.0, alpha=0.8,
               label=r'$\theta_{\mathrm{WVM}}$' + f' ({best_cusum_warn:.0f})', zorder=1)
    ax.axhline(y=best_cusum_maint, color='#FF800080', linestyle='-', linewidth=2.0,
               label=r'$\theta_{\mathrm{maint}}$' + f' ({best_cusum_maint:.0f})', zorder=1)

    pred_fault_mask = cusum_val >= best_cusum_warn
    if np.any(pred_fault_mask):
        ax.scatter(x_plot[pred_fault_mask], cusum_val[pred_fault_mask],
                   color='#FF0000', s=8, alpha=0.9, marker='o', zorder=3)

    ax.set_xlabel('t (min)', fontfamily='Times New Roman', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_ylabel('S', fontfamily='Times New Roman', fontsize=13, fontweight='bold', labelpad=12)
    ax.tick_params(axis='both', labelsize=12)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
    legend = ax.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=False,
                       prop={'family': 'Times New Roman', 'size': 12})
    legend.get_frame().set_linewidth(0.8)

    y_vals = ax.get_ylim()
    current_ticks = ax.get_yticks()
    positive_ticks = current_ticks[current_ticks >= 0]
    ax.set_yticks(positive_ticks)
    ax.set_ylim(y_vals)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis='both', direction='in', width=0.8)
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, f'{pname}_CUSUM.png'), dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()

print("[5/6] Exporting evaluation summary table...")

pname_train = "Cycle1"
feats_train = global_best_result['period_feats'][pname_train]
hi_val_train = calc_weighted_hi(feats_train, global_best_result['pca'], global_best_result['explained_var']) * global_best_result['polarity']
cusum_val_train = calc_cusum(hi_val_train, global_best_result['mu0'], global_best_result['sigma0'], global_best_result['CUSUM_K'])
time_train = np.arange(0, len(cusum_val_train) * 2, 2)
pred_train = (cusum_val_train >= best_cusum_warn).astype(int)

idx_train = df_temp[df_temp['ts'].between(*DATA_PERIODS[pname_train])]['idx'].values
true_train = (eval_labels[idx_train] == 1).astype(int)
date_train = df_active.iloc[idx_train]['Original_Time'].values if 'Original_Time' in df_active.columns else np.full(len(time_train), "")

pname_test = "Cycle2"
feats_test = global_best_result['period_feats'][pname_test]
hi_val_test = calc_weighted_hi(feats_test, global_best_result['pca'], global_best_result['explained_var']) * global_best_result['polarity']
cusum_val_test = calc_cusum(hi_val_test, global_best_result['mu0'], global_best_result['sigma0'], global_best_result['CUSUM_K'])
time_test = np.arange(0, len(cusum_val_test) * 2, 2)
pred_test = (cusum_val_test >= best_cusum_warn).astype(int)

test_idx_mask = df_temp[df_temp['ts'].between(*DATA_PERIODS[pname_test])]
idx_test = df_temp[test_idx_mask]['idx'].values
true_test = (eval_labels[idx_test] == 1).astype(int)
date_test = df_active.iloc[idx_test]['Original_Time'].values if 'Original_Time' in df_active.columns else np.full(len(time_test), "")

summary_df = pd.DataFrame({
    'Train_Time': pd.Series(time_train),
    'Train_Date': pd.Series(date_train),
    'Train_Actual_Fault': pd.Series(true_train),
    'Train_Predicted_Fault': pd.Series(pred_train),
    'Train_HI': pd.Series(hi_val_train),
    'Train_CUSUM': pd.Series(cusum_val_train),
    'Train_CUSUM_Warn_Thresh': pd.Series(np.full(len(time_train), best_cusum_warn)),
    'Train_CUSUM_Maint_Thresh': pd.Series(np.full(len(time_train), best_cusum_maint)),
    
    'Test_Time': pd.Series(time_test),
    'Test_Date': pd.Series(date_test),
    'Test_Actual_Fault': pd.Series(true_test),
    'Test_Predicted_Fault': pd.Series(pred_test),
    'Test_HI': pd.Series(hi_val_test),
    'Test_CUSUM': pd.Series(cusum_val_test),
    'Test_CUSUM_Warn_Thresh': pd.Series(np.full(len(time_test), best_cusum_warn)),
    'Test_CUSUM_Maint_Thresh': pd.Series(np.full(len(time_test), best_cusum_maint))
})

csv_summary_path = os.path.join(BEST_RESULT_DIR, 'WVM_Results_Summary.csv')
summary_df.to_csv(csv_summary_path, index=False, encoding='utf_8_sig')

print(f"✅ Results summary table exported to: {csv_summary_path}")

print("\n✅ All processes finished successfully!")
print(f"📁 Global best results saved to: {BEST_RESULT_DIR}")
print(f"✅ Optimization basis: Training set (Cycle1) only")
print(f"✅ Validation: Generalization evaluated on Test set (Cycle2)")
