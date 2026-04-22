"""
Build crystal_vae_test_carbon24.ipynb — a test notebook that:
  1. Trains VAE on carbon_24/train.csv
  2. Evaluates reconstruction loss on carbon_24/test.csv
  3. Generates new crystal structures
  4. Reports metrics: MSE, MAE, R2 on latent reconstruction
"""
import json, uuid

def cell(src, cell_type="code"):
    c = {
        "cell_type": cell_type,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": src if isinstance(src, list) else [src],
    }
    if cell_type == "code":
        c["outputs"] = []
        c["execution_count"] = None
    return c

def md(text):
    return cell(text, "markdown")

cells = [

# ── Markdown header ──────────────────────────────────────────────────────────
md("# Crystal VAE — Test Evaluation on Carbon-24\n\n"
   "**Dataset**: `carbon_24/` (10k carbon structures, 6-24 atoms per unit cell)\n\n"
   "**Pipeline**:\n"
   "1. Load `carbon_24/train.csv` → train VAE\n"
   "2. Load `carbon_24/test.csv`  → evaluate reconstruction quality\n"
   "3. Report metrics (MSE, MAE, R²) on held-out test set\n"
   "4. Generate new crystal parameters from latent space\n"
   "5. Save a generated crystal as CIF"),

# ── Install / imports ─────────────────────────────────────────────────────────
md("## 1. Imports"),

cell([
    "import pandas as pd\n",
    "import numpy as np\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
    "from pymatgen.core import Structure, Lattice\n",
    "\n",
    "torch.manual_seed(42)\n",
    "np.random.seed(42)\n",
    "print('All libraries loaded.')",
]),

# ── Feature extraction helper ─────────────────────────────────────────────────
md("## 2. Feature Extraction Helper"),

cell([
    "def extract_features(df):\n",
    "    \"\"\"Parse CIF strings and return (X, y) arrays.\"\"\"\n",
    "    features, targets = [], []\n",
    "    cif_col = [c for c in df.columns if 'cif' in c.lower()][0]\n",
    "    for i in range(len(df)):\n",
    "        try:\n",
    "            struct = Structure.from_str(df[cif_col].iloc[i], fmt='cif')\n",
    "            a, b, c = struct.lattice.abc\n",
    "            al, be, ga = struct.lattice.angles\n",
    "            features.append([len(struct), struct.volume, a, b, c, al, be, ga])\n",
    "            targets.append(df['energy_per_atom'].iloc[i])\n",
    "        except:\n",
    "            continue\n",
    "    return np.array(features), np.array(targets)\n",
    "\n",
    "print('Helper defined.')",
]),

# ── Load train ───────────────────────────────────────────────────────────────
md("## 3. Load & Prepare Training Data"),

cell([
    "print('Loading carbon_24/train.csv ...')\n",
    "train_df = pd.read_csv('carbon_24/train.csv')\n",
    "print(f'Train rows: {len(train_df)}')\n",
    "\n",
    "X_train, y_train = extract_features(train_df)\n",
    "print(f'Train features extracted: {X_train.shape}')\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X_train_sc = scaler.fit_transform(X_train)\n",
    "X_train_t  = torch.tensor(X_train_sc, dtype=torch.float32)\n",
    "print('Data normalised and converted to tensor.')",
]),

# ── Load test ────────────────────────────────────────────────────────────────
md("## 4. Load Test Data"),

cell([
    "print('Loading carbon_24/test.csv ...')\n",
    "test_df = pd.read_csv('carbon_24/test.csv')\n",
    "print(f'Test rows: {len(test_df)}')\n",
    "\n",
    "X_test, y_test = extract_features(test_df)\n",
    "print(f'Test features extracted: {X_test.shape}')\n",
    "\n",
    "X_test_sc = scaler.transform(X_test)   # use train scaler!\n",
    "X_test_t  = torch.tensor(X_test_sc, dtype=torch.float32)\n",
    "print('Test data normalised.')",
]),

# ── VAE model ────────────────────────────────────────────────────────────────
md("## 5. VAE Model Definition"),

cell([
    "class CrystalVAE(nn.Module):\n",
    "    def __init__(self, input_dim=8, latent_dim=4):\n",
    "        super().__init__()\n",
    "        self.encoder = nn.Sequential(\n",
    "            nn.Linear(input_dim, 64), nn.ReLU(),\n",
    "            nn.Linear(64, 32),        nn.ReLU()\n",
    "        )\n",
    "        self.mu     = nn.Linear(32, latent_dim)\n",
    "        self.logvar = nn.Linear(32, latent_dim)\n",
    "        self.decoder = nn.Sequential(\n",
    "            nn.Linear(latent_dim, 32), nn.ReLU(),\n",
    "            nn.Linear(32, 64),         nn.ReLU(),\n",
    "            nn.Linear(64, input_dim)\n",
    "        )\n",
    "\n",
    "    def reparameterize(self, mu, logvar):\n",
    "        std = torch.exp(0.5 * logvar)\n",
    "        return mu + torch.randn_like(std) * std\n",
    "\n",
    "    def forward(self, x):\n",
    "        h = self.encoder(x)\n",
    "        mu, logvar = self.mu(h), self.logvar(h)\n",
    "        z = self.reparameterize(mu, logvar)\n",
    "        return self.decoder(z), mu, logvar\n",
    "\n",
    "\n",
    "def vae_loss(recon, x, mu, logvar, beta=1.0):\n",
    "    recon_loss = nn.MSELoss()(recon, x)\n",
    "    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())\n",
    "    return recon_loss + beta * kl_loss, recon_loss, kl_loss\n",
    "\n",
    "\n",
    "model = CrystalVAE(input_dim=8, latent_dim=4)\n",
    "print(model)",
]),

# ── Train ────────────────────────────────────────────────────────────────────
md("## 6. Train VAE on carbon_24 Train Set"),

cell([
    "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n",
    "EPOCHS = 300\n",
    "\n",
    "train_losses = []\n",
    "for epoch in range(EPOCHS):\n",
    "    model.train()\n",
    "    optimizer.zero_grad()\n",
    "    recon, mu, logvar = model(X_train_t)\n",
    "    loss, r_loss, k_loss = vae_loss(recon, X_train_t, mu, logvar)\n",
    "    loss.backward()\n",
    "    optimizer.step()\n",
    "    train_losses.append(loss.item())\n",
    "    if epoch % 50 == 0:\n",
    "        print(f'Epoch {epoch:>3} | Total={loss.item():.4f}  Recon={r_loss.item():.4f}  KL={k_loss.item():.4f}')\n",
    "\n",
    "print('\\nTraining complete.')",
]),

# ── Evaluate on test ─────────────────────────────────────────────────────────
md("## 7. Evaluate Reconstruction on Test Set"),

cell([
    "model.eval()\n",
    "with torch.no_grad():\n",
    "    recon_test, mu_test, logvar_test = model(X_test_t)\n",
    "    test_loss, test_r, test_kl = vae_loss(recon_test, X_test_t, mu_test, logvar_test)\n",
    "\n",
    "# Back to original scale for interpretable metrics\n",
    "recon_orig = scaler.inverse_transform(recon_test.numpy())\n",
    "\n",
    "mse = mean_squared_error(X_test, recon_orig)\n",
    "mae = mean_absolute_error(X_test, recon_orig)\n",
    "r2  = r2_score(X_test, recon_orig)\n",
    "\n",
    "print('='*45)\n",
    "print(f'  Test Total Loss  : {test_loss.item():.4f}')\n",
    "print(f'  Test Recon Loss  : {test_r.item():.4f}')\n",
    "print(f'  Test KL Loss     : {test_kl.item():.4f}')\n",
    "print('-'*45)\n",
    "print(f'  MSE  (raw scale) : {mse:.4f}')\n",
    "print(f'  MAE  (raw scale) : {mae:.4f}')\n",
    "print(f'  R²   (raw scale) : {r2:.4f}')\n",
    "print('='*45)\n",
    "\n",
    "# Per-feature MAE\n",
    "feat_names = ['num_atoms','volume','a','b','c','alpha','beta','gamma']\n",
    "per_feat_mae = np.abs(X_test - recon_orig).mean(axis=0)\n",
    "print('\\nPer-feature MAE:')\n",
    "for name, val in zip(feat_names, per_feat_mae):\n",
    "    print(f'  {name:>10}: {val:.4f}')",
]),

# ── Generate new crystals ─────────────────────────────────────────────────────
md("## 8. Generate New Crystal Structures from Latent Space"),

cell([
    "N_GEN = 5\n",
    "with torch.no_grad():\n",
    "    z = torch.randn(N_GEN, 4)\n",
    "    gen_sc = model.decoder(z).numpy()\n",
    "\n",
    "generated = scaler.inverse_transform(gen_sc)\n",
    "\n",
    "print(f'Generated {N_GEN} crystal parameter sets:\\n')\n",
    "print(f'{\"#\":>3}  {\"atoms\":>6}  {\"volume\":>8}  {\"a\":>7}  {\"b\":>7}  {\"c\":>7}  {\"alpha\":>7}  {\"beta\":>7}  {\"gamma\":>7}')\n",
    "for i, g in enumerate(generated):\n",
    "    print(f'{i:>3}  {g[0]:>6.1f}  {g[1]:>8.3f}  {g[2]:>7.4f}  {g[3]:>7.4f}  {g[4]:>7.4f}  {g[5]:>7.3f}  {g[6]:>7.3f}  {g[7]:>7.3f}')",
]),

# ── Save CIF ─────────────────────────────────────────────────────────────────
md("## 9. Save Best Generated Crystal as CIF"),

cell([
    "best = generated[0]\n",
    "num_atoms, volume, a, b, c, alpha, beta, gamma = best\n",
    "\n",
    "# clamp angles to valid range\n",
    "alpha = np.clip(alpha, 60, 150)\n",
    "beta  = np.clip(beta,  60, 150)\n",
    "gamma = np.clip(gamma, 60, 150)\n",
    "a, b, c = max(a, 1.0), max(b, 1.0), max(c, 1.0)\n",
    "\n",
    "lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)\n",
    "n       = int(max(1, round(num_atoms)))\n",
    "coords  = np.random.rand(n, 3)\n",
    "species = ['C'] * n\n",
    "\n",
    "structure = Structure(lattice, species, coords)\n",
    "out_path  = 'generated_carbon24.cif'\n",
    "structure.to(filename=out_path)\n",
    "print(f'Crystal saved → {out_path}')\n",
    "print(f'  Lattice: a={a:.4f} b={b:.4f} c={c:.4f}')\n",
    "print(f'  Angles : α={alpha:.2f}° β={beta:.2f}° γ={gamma:.2f}°')\n",
    "print(f'  Atoms  : {n} Carbon atoms')",
]),

md("## ✅ Test Complete\n\n"
   "The VAE was trained on `carbon_24/train.csv` and evaluated on `carbon_24/test.csv`.\n"
   "Metrics (MSE, MAE, R²) are reported above along with per-feature reconstruction accuracy."),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14.2"},
    },
    "cells": cells,
}

out = "crystal_vae_test_carbon24.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Created {out} with {len(cells)} cells.")
