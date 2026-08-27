# -*- coding: utf-8 -*-
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
import time
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
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
MAIN_OUTPUT_DIR = "./1.WVM-Chiller 0(RP-1043)"

BEST_RESULT_DIR = os.path.join(MAIN_OUTPUT_DIR, "Result")
os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True)
os.makedirs(BEST_RESULT_DIR, exist_ok=True)

COND_VARS = ['TCI', 'TEO', 'Evap Tons']
STATE_VARS = ['TEI-TEO', 'TCO-TCI', 'TO_feed', 'TCA']
LABEL_VAR = 'Label'
polarity = -1.0
   
TRAIN_PERIOD_NAMES, TEST_PERIOD_NAMES = ["Cycle1"], ["Cycle2"]
ALL_PERIODS = TRAIN_PERIOD_NAMES + TEST_PERIOD_NAMES
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VIB_BETA_CANDIDATES = [0.0001, 0.001, 0.01, 0.1]
VIB_LR_CANDIDATES = [0.0001, 0.001, 0.01, 0.1]
MLP_LR_CANDIDATES = [0.0001, 0.001, 0.01, 0.1]
CUSUM_K_CANDIDATES = [0.2, 0.3, 0.4, 0.5]

VIB_LATENT_DIM, VIB_EPOCHS, VIB_BATCH_SIZE = 64, 300, 512
FE_LATENT_DIM, FE_EPOCHS, FE_BATCH_SIZE = 64, 300, 512

cusum_warn_candidates = [76]
cusum_maint_candidates = [98]

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.switch_backend('Agg')
SAVE_DPI = 600

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
        sys.exit(f"❌ Error: Data file {file_path} not found, please check the path!")
    df_temp = pd.read_csv(file_path, encoding='utf-8-sig')

    time_values = np.arange(global_index_counter, global_index_counter + len(df_temp) * 2, 2)
    df_temp['Time'] = time_values
    DATA_PERIODS[period_name] = (time_values[0], time_values[-1])
    global_index_counter = time_values[-1] + 2

    df_list.append(df_temp)

df_active = pd.concat(df_list).sort_values('Time').reset_index(drop=True)
if LABEL_VAR not in df_active.columns: sys.exit(f"❌ Error: Column [{LABEL_VAR}] not found in data")

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
np.save(os.path.join(preprocessing_output_dir, 'Labels_steady_points.npy'), df_active[LABEL_VAR].values)
all_labels = np.load(os.path.join(preprocessing_output_dir, 'Labels_steady_points.npy'))
all_timestamps = np.load(os.path.join(preprocessing_output_dir, 'T_steady_points.npy'))
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

def calc_weighted_hi(feats, pca, explained_var): return np.dot(pca.transform(feats), explained_var)


def calc_cusum(hi, mu0, sigma0, k):
    curr, s = 0.0, np.zeros(len(hi))
    for i in range(len(hi)):
        curr = max(0, curr + (mu0 - hi[i]) - k)
        s[i] = curr
    return s

def calc_lead_time(pred, times, y_true):
    if len(pred) < 5: return 0.0

    true_fault_indices = np.where(y_true == 1)[0]
    if len(true_fault_indices) == 0:
        return 0.0
        
    first_true_fault_time = times[true_fault_indices[0]]
    
    first_warn_idx = -1
    found = False

    for i in range(len(pred) - 4):
        if np.sum(pred[i:i + 5]) == 5:
            first_warn_idx = i
            found = True
            break

    if found:
        return times[first_warn_idx] - first_true_fault_time
    else:
        return 0.0

def evaluate_threshold(cusum_warn_p, cusum_maint_p, train_cusum_val, sigma0, mu0,
                       period_feats, polarity, pca, explained_var, all_labels, df_temp, target_periods,
                       cusum_k):
    cusum_th_warn = np.percentile(train_cusum_val, cusum_warn_p)
    cusum_th_maint = np.percentile(train_cusum_val, cusum_maint_p)

    cusum_acc_list, cusum_rec_list, cusum_fpr_list, cusum_lead_list = [], [], [], []

    for pname in target_periods:
        period_mask = df_temp['ts'].between(*DATA_PERIODS[pname])
        idx = df_temp[period_mask]['idx'].values
        curr_ts = df_temp[period_mask]['ts'].values
        feats = period_feats[pname]
        curr_labels = all_labels[idx]

        hi_val = calc_weighted_hi(feats, pca, explained_var) * polarity
        y_true = np.where(curr_labels == 100, 0, 1)

        cusum_val = calc_cusum(hi_val, mu0, sigma0, k=cusum_k)
        cusum_pred = (cusum_val > cusum_th_warn).astype(int)
        c_acc = accuracy_score(y_true, cusum_pred)
        c_rec = np.mean(cusum_pred[y_true == 1] == 1) if np.sum(y_true == 1) > 0 else 1.0
        c_fpr = np.mean(cusum_pred[y_true == 0] == 1) if np.sum(y_true == 0) > 0 else 0.0
        c_lead = calc_lead_time(cusum_pred, curr_ts, y_true)
        cusum_acc_list.append(c_acc)
        cusum_rec_list.append(c_rec)
        cusum_fpr_list.append(c_fpr)
        cusum_lead_list.append(c_lead)

    return {
        'cusum_warn_p': cusum_warn_p, 'cusum_maint_p': cusum_maint_p,
        'avg_acc': np.mean(cusum_acc_list), 'avg_rec': np.mean(cusum_rec_list),
        'avg_fpr': np.mean(cusum_fpr_list), 'avg_lead': np.mean(cusum_lead_list)
    }

print("[2/6] Starting Two-Level Global Optimization: Hyperparameters + Thresholds (Optimizing for incipient faults, please wait...)")
start_time = time.time()
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
                    f"\n===== Current Hyperparameters: VIB_BETA={VIB_BETA}, VIB_LR={VIB_LR}, MLP_LR={MLP_LR}, CUSUM_K={CUSUM_K} =====")

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

                best_vib_model = VIBResidualGenerator(X_train_np.shape[1], Y_train_np.shape[1], VIB_LATENT_DIM).to(
                    DEVICE)
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
                    f"[HI Inversion Check]: train_hi_raw correlation with time corr={corr:.4f} | polarity={polarity} | {'Inverted' if polarity == -1.0 else 'Not Inverted'}")
                
                train_cusum_val = calc_cusum(train_hi, mu0, sigma0, CUSUM_K)
                search_results = []
                for c_warn in cusum_warn_candidates:
                    for c_maint in cusum_maint_candidates:
                        if c_maint <= c_warn: continue
                        res = evaluate_threshold(c_warn, c_maint, train_cusum_val, sigma0,
                                                 mu0, period_feats, polarity, pca,
                                                 explained_var, all_labels, df_temp, TRAIN_PERIOD_NAMES,
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
                    f"Current best metrics (Train Set): Recall={curr_best['avg_rec']:.4f}, Overall Accuracy={curr_best['avg_acc']:.4f}")

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

# ===================== Save Global Best Results =====================
end_time = time.time() 
algo_cost_time = end_time - start_time 
print("\n" + "=" * 100)
print("🎉 Two-Level Global Optimization Completed! (Selected based on Train Set)")
print(
    f"📌 Best Hyperparameters → VIB_BETA={global_best_result['VIB_BETA']}, VIB_LR={global_best_result['VIB_LR']}, MLP_LR={global_best_result['MLP_LR']}")
print(f"📌 Best Sensitivity Coefficient → CUSUM_K={global_best_result['CUSUM_K']}")

# ===================== Plot Predicted Weak Labels =====================
print("[3/6] Plotting predicted weak label visualizations with best hyperparameters...")
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

vis_output_dir_best = os.path.join(BEST_RESULT_DIR, 'Best_Hyperparams_Weak_Label_Vis')
os.makedirs(vis_output_dir_best, exist_ok=True)

for period_name in ALL_PERIODS:
    pred_vals = period_pred_labels_best[period_name]
    true_vals = period_true_labels_best[period_name]
    sample_num = np.arange(len(pred_vals)) 

    plt.figure(figsize=(15, 6))
    plt.plot(sample_num, pred_vals, label='Predicted Weak Label', color='#1f77b4', linewidth=1.2)
    plt.scatter(sample_num, true_vals, label='True Weak Label', color='#ff4757', s=8, alpha=0.6)

    plt.title(f'MLP-Predicted Weak Label', fontsize=14, pad=15)
    plt.xlabel('Samples', fontsize=12)
    plt.ylabel('Weak Label', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    save_path = os.path.join(vis_output_dir_best, f'{period_name}_mlp_weak_label_pred_best.png')
    plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()  

print(f"✅ Weak label visualization completed. Images saved to: {vis_output_dir_best}")

best_train_feats = global_best_result['period_feats']['Cycle1']
best_train_hi = calc_weighted_hi(best_train_feats, global_best_result['pca'], global_best_result['explained_var']) * \
                global_best_result['polarity']
best_train_cusum_val = calc_cusum(best_train_hi, global_best_result['mu0'], global_best_result['sigma0'],
                                  global_best_result['CUSUM_K'])

test_res = evaluate_threshold(
    global_best_result['cusum_warn_p'], global_best_result['cusum_maint_p'],
    best_train_cusum_val,  
    global_best_result['sigma0'],
    global_best_result['mu0'],
    global_best_result['period_feats'],
    global_best_result['polarity'],
    global_best_result['pca'],
    global_best_result['explained_var'],
    all_labels,
    df_temp,
    TEST_PERIOD_NAMES,  
    global_best_result['CUSUM_K']  
)

print("-" * 50)
print("📊 [Core Verification] Test Set (Cycle2) Generalization Metrics:")
print(
    f"[CUSUM] Fault Recall={test_res['avg_rec']:.4f} | Accuracy={test_res['avg_acc']:.4f} | FPR={test_res['avg_fpr']:.4f}")
print(f"        Warning Lead Time={test_res['avg_lead']:.2f}")
print("-" * 50)
# ------------------------------------------------------------------

train_cusum_val = calc_cusum(calc_weighted_hi(
    global_best_result['period_feats']['Cycle1'], global_best_result['pca'], global_best_result['explained_var']) *
                             global_best_result[
                                 'polarity'],
                             global_best_result['mu0'], global_best_result['sigma0'], global_best_result['CUSUM_K'])
best_cusum_warn = np.percentile(train_cusum_val, global_best_result['cusum_warn_p'])
best_cusum_maint = np.percentile(train_cusum_val, global_best_result['cusum_maint_p'])

with open(os.path.join(BEST_RESULT_DIR, 'Global_Best_Parameters.txt'), 'w', encoding='utf-8') as f:
    f.write("Condenser Fouling Fault Detection - Global Best Parameters (Train Set Optimized)\n")
    f.write(
        f"[Hyperparameters] VIB_BETA={global_best_result['VIB_BETA']}, VIB_LR={global_best_result['VIB_LR']}, MLP_LR={global_best_result['MLP_LR']}\n")
    f.write(f"[Sensitivity Coefficient] CUSUM_K={global_best_result['CUSUM_K']}\n")
    f.write(
        f"[CUSUM Thresholds] Early Warning Quantile={global_best_result['cusum_warn_p']} → {best_cusum_warn:.6f}, Maintenance Quantile={global_best_result['cusum_maint_p']} → {best_cusum_maint:.6f}\n")

with open(os.path.join(BEST_RESULT_DIR, 'Global_Best_Metrics.txt'), 'w', encoding='utf-8') as f:
    f.write("Condenser Fouling Fault Detection - Global Best Evaluation Metrics\n")
    f.write("========================================\n")
    f.write(f"[Algorithm Efficiency]\n")                                      
    f.write(f"Global Optimization Elapsed Time: {algo_cost_time:.2f} s\n")       
    f.write("-" * 20 + "\n")                                          
    f.write("[Train Set Performance (Optimization Basis)]\n")
    f.write(f"CUSUM Fault Recall: {global_best_result['avg_rec']:.4f}\n")
    f.write(f"CUSUM Overall Accuracy: {global_best_result['avg_acc']:.4f}\n")
    f.write("\n")
    f.write("[Test Set Performance (Generalization Check)]\n")
    f.write(f"CUSUM Fault Recall: {test_res['avg_rec']:.4f}\n")
    f.write(f"CUSUM Overall Accuracy: {test_res['avg_acc']:.4f}\n")
    f.write(f"CUSUM False Positive Rate: {test_res['avg_fpr']:.4f}\n")
    f.write(f"Warning Lead Time: {test_res['avg_lead']:.2f}\n")

shutil.copy(global_best_result['best_vib_path'], os.path.join(BEST_RESULT_DIR, 'VIB_Global_Best_Model.pth'))
shutil.copy(global_best_result['best_mlp_path'], os.path.join(BEST_RESULT_DIR, 'MLP_Global_Best_Model.pth'))

print("[Additional] Plotting state variable residual trends (Train / Test sets separated)...")
res_plot_dir = os.path.join(BEST_RESULT_DIR, 'Best_Residual_Trends')
os.makedirs(res_plot_dir, exist_ok=True)
res_train_dir = os.path.join(res_plot_dir, 'Train_Set_Cycle1')
res_test_dir = os.path.join(res_plot_dir, 'Test_Set_Cycle2')
os.makedirs(res_train_dir, exist_ok=True)
os.makedirs(res_test_dir, exist_ok=True)

best_vib_model = VIBResidualGenerator(X_np.shape[1], Y_np.shape[1], VIB_LATENT_DIM).to(DEVICE)
best_vib_model.load_state_dict(torch.load(os.path.join(BEST_RESULT_DIR, 'VIB_Global_Best_Model.pth')))
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

print(f"✅ Residual trend plotting completed:")
print(f"   - Train set residual plots saved to: {res_train_dir}")
print(f"   - Test set residual plots saved to: {res_test_dir}")

plot_dir = os.path.join(BEST_RESULT_DIR, 'Best_Monitoring_Plots')
os.makedirs(plot_dir, exist_ok=True)
for pname in ALL_PERIODS:
    idx = df_temp[df_temp['ts'].between(*DATA_PERIODS[pname])]['idx'].values
    feats = global_best_result['period_feats'][pname]
    curr_labels = all_labels[idx]
    hi_val = calc_weighted_hi(feats, global_best_result['pca'], global_best_result['explained_var']) * \
             global_best_result['polarity']
    cusum_val = calc_cusum(hi_val, global_best_result['mu0'], global_best_result['sigma0'],
                           global_best_result['CUSUM_K'])
    y_true = np.where(curr_labels == 100, 0, 1)
    idx_plot = np.arange(len(feats)) * 2  
    fault_mask = y_true == 1

    fig, ax = plt.subplots(figsize=(9, 6))  
    ax.set_facecolor('#FFFFFF')  
    
    current_start = None
    for i in range(len(y_true)):
        if y_true[i] == 1:  
            if current_start is None:
                current_start = i
        else:  
            if current_start is not None:
                ax.axvspan(idx_plot[current_start], idx_plot[i - 1], color='#FFC0CB', alpha=1.0, zorder=1)
                current_start = None
    if current_start is not None:
        ax.axvspan(idx_plot[current_start], idx_plot[-1], color='#FFC0CB', alpha=1.0, zorder=1)

    ax.plot(idx_plot, hi_val, color='blue', linewidth=2.0, linestyle='-', alpha=0.8, label='HI')

    ax.set_xlabel('Operating time (min)', fontfamily='Times New Roman', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_ylabel('HI', fontfamily='Times New Roman', fontsize=13, fontweight='bold', labelpad=12)

    ax.tick_params(axis='both', labelsize=12, direction='in')
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')

    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=False,
                       prop={'family': 'Times New Roman', 'size': 12})
    legend.get_frame().set_linewidth(0.8)

    ax.yaxis.set_major_locator(plt.LinearLocator(numticks=6))

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis='both', width=0.8)

    ax.grid(True, alpha=0.3)

    plt.savefig(os.path.join(plot_dir, f'{pname}_HI.png'), dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.set_facecolor('#FFFFFF') 
    
    current_start = None
    for i in range(len(y_true)):
        if y_true[i] == 1:  
            if current_start is None:
                current_start = i
        else:  
            if current_start is not None:
                ax.axvspan(idx_plot[current_start], idx_plot[i - 1], color='#FFC0CB', alpha=1.0, zorder=1)
                current_start = None
    if current_start is not None:
        ax.axvspan(idx_plot[current_start], idx_plot[-1], color='#FFC0CB', alpha=1.0, zorder=1)

    ax.plot(idx_plot, cusum_val, color='#1f77b4', linewidth=2.0, linestyle='-', alpha=0.8, label='CUSUM', zorder=2)

    ax.axhline(y=best_cusum_warn, color='#FF0000', linestyle='--', linewidth=2.0, alpha=0.8,
               label=f'Fault Early Warning Threshold ({best_cusum_warn:.2f})', zorder=1)
    ax.axhline(y=best_cusum_maint, color='#FF800080', linestyle='-', linewidth=2.0,
               label=f'Maintenance Threshold ({best_cusum_maint:.2f})', zorder=1)

    pred_fault_mask = cusum_val > best_cusum_warn
    if np.any(pred_fault_mask):
        ax.scatter(idx_plot[pred_fault_mask], cusum_val[pred_fault_mask],
                   color='#FF0000', s=8, alpha=0.9, marker='o', zorder=3)

    ax.set_ylim(bottom=0)

    ax.set_xlabel('Operating time (min)', fontfamily='Times New Roman', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_ylabel('S', fontfamily='Times New Roman', fontsize=13, fontweight='bold', labelpad=12)

    ax.tick_params(axis='both', labelsize=12)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')

    legend = ax.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=False,
                       prop={'family': 'Times New Roman', 'size': 12})
    legend.get_frame().set_linewidth(0.8)

    y_limit = ax.get_ylim()
    ax.set_yticks(np.linspace(y_limit[0], y_limit[1], 6))

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis='both', direction='in', width=0.8)
    ax.grid(True, alpha=0.3)

    plt.savefig(os.path.join(plot_dir, f'{pname}_CUSUM.png'), dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()

print("[Additional] Generating Summary.csv...")

train_pname = "Cycle1"
train_idx = df_temp[df_temp['ts'].between(*DATA_PERIODS[train_pname])]['idx'].values
train_feats = global_best_result['period_feats'][train_pname]
train_curr_labels = all_labels[train_idx]

train_time = np.arange(len(train_feats)) * 2
train_actual_fault = np.where(train_curr_labels == 100, 0, 1)
train_hi_val = calc_weighted_hi(train_feats, global_best_result['pca'], global_best_result['explained_var']) * global_best_result['polarity']
train_cusum_val = calc_cusum(train_hi_val, global_best_result['mu0'], global_best_result['sigma0'], global_best_result['CUSUM_K'])
train_pred_fault = (train_cusum_val > best_cusum_warn).astype(int)

df_train_summary = pd.DataFrame({
    'Train-Time': train_time,
    'Train-Actual_Fault': train_actual_fault,
    'Train-Predicted_Fault': train_pred_fault,
    'Train-CUSUM_Value': train_cusum_val,
    'Train-CUSUM_Warning_Threshold': best_cusum_warn,
    'Train-HI': train_hi_val,
    'Train-HI_Mean': global_best_result['mu0']
})


test_pname = "Cycle2"
test_idx = df_temp[df_temp['ts'].between(*DATA_PERIODS[test_pname])]['idx'].values
test_feats = global_best_result['period_feats'][test_pname]
test_curr_labels = all_labels[test_idx]


test_time = np.arange(len(test_feats)) * 2
test_actual_fault = np.where(test_curr_labels == 100, 0, 1)
test_hi_val = calc_weighted_hi(test_feats, global_best_result['pca'], global_best_result['explained_var']) * global_best_result['polarity']
test_cusum_val = calc_cusum(test_hi_val, global_best_result['mu0'], global_best_result['sigma0'], global_best_result['CUSUM_K'])
test_pred_fault = (test_cusum_val > best_cusum_warn).astype(int)

df_test_summary = pd.DataFrame({
    'Test-Time': test_time,
    'Test-Actual_Fault': test_actual_fault,
    'Test-Predicted_Fault': test_pred_fault,
    'Test-CUSUM_Value': test_cusum_val,
    'Test-CUSUM_Warning_Threshold': best_cusum_warn,
    'Test-CUSUM_Maintenance_Threshold': best_cusum_maint,
    'Test-HI': test_hi_val
})

df_result_summary = pd.concat([df_train_summary, df_test_summary], axis=1)

summary_csv_path = os.path.join(BEST_RESULT_DIR, 'Summary.csv')
df_result_summary.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
print(f"✅ Summary table generated and saved to: {summary_csv_path}")

print("[Additional] Generating single test set record table...")

df_test_record = pd.DataFrame({
    'Operating_Time': test_time,
    'HI': test_hi_val,
    'Actual_Sample_Class': test_curr_labels
})

test_record_csv_path = os.path.join(BEST_RESULT_DIR, 'Test_Set_HI_Record.csv')
df_test_record.to_csv(test_record_csv_path, index=False, encoding='utf-8-sig')
print(f"✅ Test set record table saved to: {test_record_csv_path}")

print("\n✅ All processes completed!")
print(f"📁 Global best results saved to: {BEST_RESULT_DIR}")
print(f"✅ Optimization basis: Train Set (Cycle1) only")
print(f"✅ Verification: Generalization evaluated on Test Set (Cycle2)")