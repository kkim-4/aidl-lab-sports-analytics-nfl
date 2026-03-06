import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam
from sklearn.preprocessing import StandardScaler

# --- 1. DATA INGESTION & PRE-COMPUTED CAUSAL WINDOW ---
def load_and_prepare_nflfastr(window_size=18):
    cache_file = 'nfl_kicks_1999_2024.csv'
    
    if os.path.exists(cache_file):
        print(f"📂 Loading cached data from {cache_file}...")
        df = pd.read_csv(cache_file)
    else:
        print("📡 Downloading nflfastR data (1999-2024)...")
        base_url = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_"
        df_list = []
        for s in range(1999, 2025):
            cols = ['kicker_player_id', 'kicker_player_name', 'season', 'week', 
                    'game_date', 'field_goal_result', 'kick_distance', 
                    'quarter_seconds_remaining', 'score_differential']
            try:
                season_df = pd.read_csv(f"{base_url}{s}.csv.gz", usecols=cols, compression='gzip', low_memory=False)
                df_list.append(season_df)
            except: pass
        df = pd.concat(df_list).dropna(subset=['field_goal_result'])
        df['made'] = (df['field_goal_result'] == 'made').astype(int)
        df.to_csv(cache_file, index=False)

    df = df.sort_values(['game_date'])
    df['career_kicks'] = df.groupby('kicker_player_id').cumcount()
    feat_cols = ['kick_distance', 'quarter_seconds_remaining', 'score_differential', 'career_kicks']
    
    scaler = StandardScaler()
    df[feat_cols] = scaler.fit_transform(df[feat_cols].astype(np.float32))
    
    timeline = df[['season', 'week']].drop_duplicates().sort_values(['season', 'week'])
    timeline['t_idx'] = range(len(timeline))
    df = df.merge(timeline, on=['season', 'week'])
    
    k_ids = sorted(df['kicker_player_id'].unique())
    max_t = len(timeline)
    
    X = torch.zeros((len(k_ids), max_t, len(feat_cols)))
    y = torch.zeros((len(k_ids), max_t))
    mask = torch.zeros((len(k_ids), max_t))
    cs_mask = torch.zeros((len(k_ids), max_t), dtype=torch.bool)
    
    k_map = {kid: i for i, kid in enumerate(k_ids)}
    for _, row in df.iterrows():
        i, t = k_map[row['kicker_player_id']], int(row['t_idx'])
        X[i, t] = torch.from_numpy(row[feat_cols].values.astype(np.float32))
        y[i, t] = float(row['made'])
        mask[i, t] = 1.0
        
    for i in range(len(k_ids)):
        first_idx = torch.where(mask[i] > 0)[0]
        if len(first_idx) > 0:
            cs_mask[i, first_idx[0]:] = True

    # --- OPTIMIZATION: PRE-COMPUTE CAUSAL MEANS ---
    print(f"⚙️ Pre-computing {window_size}-week causal windows...")
    X_history = torch.zeros_like(X)
    for t in range(1, max_t):
        start = max(0, t - window_size)
        window_x = X[:, start:t, :]
        window_m = mask[:, start:t].unsqueeze(-1)
        # Weighted mean over the window to avoid bye-week bias
        X_history[:, t, :] = (window_x * window_m).sum(dim=1) / (window_m.sum(dim=1) + 1e-6)
            
    return X, X_history, y, mask, cs_mask, k_ids

# --- 2. MODEL DEFINITION ---
class AmortizedFGOE_Encoder(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features * 2 + 1, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x_hist, x_curr, volume):
        inputs = torch.cat([x_hist, x_curr, volume], dim=-1)
        params = self.net(inputs)
        return params[:, 0], torch.exp(params[:, 1].clamp(min=-3.0, max=0.5))

class FGOE_AmortizedSplineSSM(PyroModule):
    def __init__(self, in_features):
        super().__init__()
        self.baseline = nn.Sequential(nn.Linear(in_features, 8), nn.Linear(8, 1))
        self.encoder = AmortizedFGOE_Encoder(in_features)

    def model(self, X, X_history, y, mask, cs_mask):
        pyro.module("fgoe", self)
        batch_size, max_t, _ = X.shape
        z_prev = torch.zeros(batch_size).to(X.device)
        with pyro.plate("kickers", batch_size):
            for t in range(max_t):
                z_t = pyro.sample(f"z_{t+1}", dist.Normal(z_prev * 0.98, 0.05))
                z_t = torch.where(cs_mask[:, t], z_t, torch.zeros_like(z_t))
                logits = (self.baseline(X[:, t, :]).squeeze(-1) + (z_t * 15.0)) / 0.3
                pyro.sample(f"obs_{t+1}", dist.Bernoulli(logits=logits).mask(mask[:, t] > 0), obs=y[:, t])
                z_prev = z_t

    def guide(self, X, X_history, y, mask, cs_mask):
        pyro.module("encoder", self.encoder)
        batch_size, max_t, _ = X.shape
        with pyro.plate("kickers", batch_size):
            for t in range(max_t):
                mu_q, sigma_q = self.encoder(X_history[:, t, :], X[:, t, :], X[:, t, -1:])
                pyro.sample(f"z_{t+1}", dist.Normal(mu_q, sigma_q))

# --- 3. TRAINING ---
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Acceleration: {device}")

    X, X_history, y, mask, cs_mask, k_ids = load_and_prepare_nflfastr()
    X, X_history, y, mask, cs_mask = X.to(device), X_history.to(device), y.to(device), mask.to(device), cs_mask.to(device)

    model = FGOE_AmortizedSplineSSM(in_features=X.shape[2]).to(device)
    optimizer = ClippedAdam({"lr": 0.0015, "clip_norm": 5.0}) # Faster learning rate
    svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())

    pyro.clear_param_store()
    print(f"📉 Training on {len(k_ids)} kickers (1,800 steps)...")
    
    for step in range(1801):
        loss = svi.step(X, X_history, y, mask, cs_mask)
        if step % 300 == 0:
            print(f"Step {step:4} | Loss: {loss:,.2f}")

    print("💾 Saving samples...")
    predictive = Predictive(model.model, guide=model.guide, num_samples=30)
    samples = predictive(X, X_history, y, mask, cs_mask)
    torch.save(samples, 'full_era_18wk_samples.pt')
    print("✅ Success.")