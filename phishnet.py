"""
PhishNet — Multi-Model Email Threat Detector
"""

import re, io, json, pickle, zipfile
from html import escape
from pathlib import Path
from collections import Counter
from datetime import datetime

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="PhishNet · Multi-Model",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{
  --bg:       #f5f7fa;
  --card:     #ffffff;
  --raised:   #eef1f6;
  --red:      #e02d47;
  --green:    #059669;
  --blue:     #2563eb;
  --amber:    #d97706;
  --purple:   #7c3aed;
  --txt:      #111827;
  --muted:    #6b7280;
  --border:   #d1d5db;
}
html,body,[data-testid="stAppViewContainer"]{background:#f5f7fa!important;color:var(--txt)!important;font-family:'DM Sans',sans-serif;}
[data-testid="stSidebar"]{background:#ffffff!important;border-right:1px solid var(--border);}
h1,h2,h3{font-family:'Space Mono',monospace;color:var(--txt);}
.card{background:#ffffff;border:1px solid var(--border);border-radius:14px;padding:20px 24px;margin-bottom:14px;box-shadow:0 1px 4px rgba(0,0,0,.06);}
.raised{background:#eef1f6;border:1px solid var(--border);border-radius:10px;padding:14px 18px;}
.verdict-p{background:linear-gradient(135deg,#fff1f2,#ffe4e6);border:1.5px solid #e02d47;border-radius:14px;padding:20px 28px;text-align:center;box-shadow:0 2px 12px rgba(224,45,71,.12);}
.verdict-l{background:linear-gradient(135deg,#ecfdf5,#d1fae5);border:1.5px solid #059669;border-radius:14px;padding:20px 28px;text-align:center;box-shadow:0 2px 12px rgba(5,150,105,.12);}
.vtitle{font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;margin:0;}
.vsub{font-size:0.82rem;color:var(--muted);margin-top:4px;}
.mtile{background:#ffffff;border:1px solid var(--border);border-radius:10px;padding:14px 18px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.05);}
.mval{font-family:'Space Mono',monospace;font-size:1.45rem;font-weight:700;}
.mlbl{font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-top:3px;}
.badge{display:inline-flex;align-items:center;gap:5px;background:#eef1f6;border:1px solid var(--border);border-radius:20px;padding:3px 10px;font-size:0.76rem;color:var(--muted);margin:2px;}
.slabel{font-family:'Space Mono',monospace;font-size:0.68rem;text-transform:uppercase;letter-spacing:2px;color:var(--muted);margin-bottom:8px;}
.pill{display:inline-block;margin:3px;padding:5px 9px;border-radius:7px;font-family:'Space Mono',monospace;font-size:12px;cursor:default;transition:transform .15s;}
.pill:hover{transform:scale(1.08);}
.model-card{background:#ffffff;border:1px solid var(--border);border-radius:12px;padding:16px 20px;margin-bottom:10px;cursor:pointer;transition:border-color .2s;box-shadow:0 1px 3px rgba(0,0,0,.05);}
.model-card:hover{border-color:var(--blue);}
.model-card.active{border-color:var(--blue);background:rgba(37,99,235,.04);}
.tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.7rem;font-family:'Space Mono',monospace;font-weight:700;}
.tag-bilstm{background:rgba(37,99,235,.1);color:#1d4ed8;}
.tag-rnn{background:rgba(5,150,105,.1);color:#047857;}
.tag-cnn{background:rgba(234,88,12,.1);color:#c2410c;}
.tag-ffn{background:rgba(180,83,9,.1);color:#92400e;}
.tag-sklearn{background:rgba(109,40,217,.1);color:#5b21b6;}
.tag-unknown{background:rgba(107,114,128,.1);color:#4b5563;}
.stButton>button{background:linear-gradient(135deg,#2563eb,#1d4ed8)!important;color:#fff!important;border:none!important;border-radius:8px!important;font-family:'Space Mono',monospace!important;font-weight:700!important;width:100%;box-shadow:0 2px 6px rgba(37,99,235,.3)!important;}
.stButton>button:hover{opacity:.9!important;}
.stTextArea textarea{background:#fff!important;border:1px solid var(--border)!important;border-radius:10px!important;color:var(--txt)!important;}
.stTabs [data-baseweb="tab-list"]{background:#eef1f6!important;border-radius:10px;gap:4px;padding:4px;border:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;border-radius:8px!important;font-family:'Space Mono',monospace!important;font-size:.72rem!important;}
.stTabs [aria-selected="true"]{background:#fff!important;color:var(--txt)!important;box-shadow:0 1px 3px rgba(0,0,0,.08);}
[data-testid="stExpander"]{background:#ffffff!important;border:1px solid var(--border)!important;border-radius:10px!important;}
label{color:var(--muted)!important;font-size:.8rem!important;}
.hrow{display:flex;align-items:center;gap:10px;padding:8px 12px;border-radius:8px;margin-bottom:5px;background:#eef1f6;border:1px solid var(--border);font-size:.82rem;color:var(--txt);}
.dot-r{width:9px;height:9px;border-radius:50%;background:var(--red);flex-shrink:0;}
.dot-g{width:9px;height:9px;border-radius:50%;background:var(--green);flex-shrink:0;}
.compare-col{background:#ffffff;border:1px solid var(--border);border-radius:12px;padding:18px;text-align:center;}
div[data-testid="stMarkdownContainer"] p{color:var(--txt);}
</style>
""", unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════════════

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        _, (h, _) = self.lstm(emb)
        out = self.dropout(torch.cat((h[-2], h[-1]), dim=1))
        return self.fc(out)


class TextCNN(nn.Module):
    """
    Flexible CNN — supports two architectures:
      A) Multi-kernel Conv2d:  state dict has 'convs.0.weight', 'convs.1.weight' …
      B) Single-layer Conv1d:  state dict has 'conv1.weight' or 'conv.weight'
    Architecture is inferred from the state dict so it always matches.
    """
    def __init__(self, state_dict: dict, vocab_size: int, embed_dim: int,
                 dropout: float = 0.5):
        super().__init__()

        keys = list(state_dict.keys())

        # Embedding (always present)
        emb_w = state_dict.get("embedding.weight")
        actual_vocab = emb_w.shape[0] if emb_w is not None else vocab_size
        actual_embed = emb_w.shape[1] if emb_w is not None else embed_dim
        self.embedding = nn.Embedding(actual_vocab, actual_embed, padding_idx=0)

        # ── Detect architecture variant ─────────────────────────────────────
        # Variant A: multi-kernel Conv2d (convs.N.weight)
        conv2d_keys = sorted([k for k in keys if k.startswith("convs.") and k.endswith(".weight")])
        # Variant B: single-layer Conv1d  (conv1.weight or conv.weight)
        conv1d_keys = [k for k in keys
                       if k.endswith(".weight") and (k.startswith("conv1.") or k.startswith("conv."))]

        fc_w = state_dict.get("fc.weight")
        if fc_w is None:
            raise ValueError("CNN state dict missing fc.weight")
        num_classes = fc_w.shape[0]

        if conv2d_keys:
            # Variant A
            self.variant = "conv2d_multi"
            kernel_sizes = []
            num_filters_list = []
            for k in conv2d_keys:
                w = state_dict[k]   # shape: [num_filters, 1, kernel, embed_dim]
                num_filters_list.append(w.shape[0])
                kernel_sizes.append(w.shape[2])
            self.kernel_sizes = kernel_sizes
            self.convs = nn.ModuleList([
                nn.Conv2d(1, nf, (k, actual_embed))
                for nf, k in zip(num_filters_list, kernel_sizes)
            ])
            fc_in_expected = sum(num_filters_list)
        elif conv1d_keys:
            # Variant B — single Conv1d
            self.variant = "conv1d_single"
            conv_attr = conv1d_keys[0].split(".")[0]   # 'conv1' or 'conv'
            self._conv_attr = conv_attr
            w = state_dict[conv1d_keys[0]]   # shape: [out_ch, in_ch, kernel]
            out_ch = w.shape[0]
            in_ch  = w.shape[1]
            kernel = w.shape[2]
            self.kernel = kernel
            conv_module = nn.Conv1d(in_ch, out_ch, kernel)
            setattr(self, conv_attr, conv_module)
            fc_in_expected = out_ch
        else:
            raise ValueError(f"Could not identify CNN architecture from keys: {keys[:8]}…")

        # fc layer — input dim should match what we computed
        self.fc       = nn.Linear(fc_w.shape[1], num_classes)
        self.dropout  = nn.Dropout(dropout)
        self.sigmoid_out = (num_classes == 1)

    def forward(self, x):
        emb = self.embedding(x)   # (B, T, embed_dim)
        if self.variant == "conv2d_multi":
            emb_4d = emb.unsqueeze(1)   # (B, 1, T, embed_dim)
            pooled = [F.relu(c(emb_4d)).squeeze(3) for c in self.convs]
            pooled = [F.max_pool1d(p, p.size(2)).squeeze(2) for p in pooled]
            out    = self.dropout(torch.cat(pooled, dim=1))
        else:   # conv1d_single
            # Conv1d expects (B, in_ch=embed_dim, T)
            emb_3d = emb.transpose(1, 2)
            conv   = getattr(self, self._conv_attr)
            c_out  = F.relu(conv(emb_3d))                 # (B, out_ch, T-k+1)
            pooled = F.max_pool1d(c_out, c_out.size(2))    # (B, out_ch, 1)
            out    = self.dropout(pooled.squeeze(2))       # (B, out_ch)
        return self.fc(out)


class TextRNN(nn.Module):
    """
    Flexible RNN — unidirectional OR bidirectional LSTM/GRU.
    Auto-detects every architectural detail from the state dict:
      • embedding dimensions
      • RNN attribute name ('rnn' or 'lstm' — teammates name it differently)
      • LSTM vs GRU
      • unidirectional vs bidirectional
      • number of layers
      • hidden size
      • 1-output sigmoid vs 2-output softmax
    """
    def __init__(self, state_dict: dict, vocab_size: int, embed_dim: int,
                 dropout: float = 0.5):
        super().__init__()

        keys = list(state_dict.keys())

        # ── Detect which attribute name was used: 'rnn', 'lstm', or 'gru' ──
        rnn_attr = None
        for candidate in ("rnn", "lstm", "gru"):
            if any(k.startswith(f"{candidate}.weight_ih") for k in keys):
                rnn_attr = candidate
                break
        if rnn_attr is None:
            raise ValueError(
                "Could not find RNN/LSTM/GRU layer in state dict. "
                f"Keys present: {keys[:10]}…"
            )
        self._rnn_attr = rnn_attr

        # ── Embedding ────────────────────────────────────────────────────────
        emb_w        = state_dict.get("embedding.weight")
        actual_vocab = emb_w.shape[0] if emb_w is not None else vocab_size
        actual_embed = emb_w.shape[1] if emb_w is not None else embed_dim

        # ── LSTM vs GRU based on whether attr name says lstm or by checking
        # for the 4x weight pattern (LSTM weight_ih shape is [4*H, in])
        # versus GRU's [3*H, in]
        wih = state_dict[f"{rnn_attr}.weight_ih_l0"]
        # We need hidden_size to figure this out — derive from fc.weight
        fc_w  = state_dict.get("fc.weight")
        fc_in = fc_w.shape[1] if fc_w is not None else 128
        is_bidir = any("_reverse" in k for k in keys)
        hidden_size = fc_in // (2 if is_bidir else 1)

        # weight_ih shape[0] is gate_count * hidden_size  →  4 = LSTM, 3 = GRU
        gate_factor = wih.shape[0] // hidden_size
        if gate_factor == 4:
            is_lstm = True
        elif gate_factor == 3:
            is_lstm = False
        else:
            # fall back to attribute name
            is_lstm = (rnn_attr == "lstm")

        # num_layers
        layer_nums = set()
        for k in keys:
            if k.startswith(f"{rnn_attr}.weight_ih_l"):
                part = k.replace(f"{rnn_attr}.weight_ih_l", "").replace("_reverse", "")
                try:    layer_nums.add(int(part))
                except: pass
        num_layers  = max(layer_nums) + 1 if layer_nums else 1
        num_classes = fc_w.shape[0] if fc_w is not None else 2

        # ── Build matching modules — IMPORTANT: attribute name must match ──
        rnn_cls = nn.LSTM if is_lstm else nn.GRU
        self.embedding = nn.Embedding(actual_vocab, actual_embed, padding_idx=0)

        rnn_module = rnn_cls(
            actual_embed, hidden_size, num_layers,
            batch_first=True, bidirectional=is_bidir,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Set the attribute under the ORIGINAL name from the state dict
        # so load_state_dict matches keys correctly
        setattr(self, rnn_attr, rnn_module)

        self.dropout    = nn.Dropout(dropout)
        self.fc         = nn.Linear(fc_in, num_classes)
        self.is_lstm    = is_lstm
        self.is_bidir   = is_bidir
        self.sigmoid_out = (num_classes == 1)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        rnn = getattr(self, self._rnn_attr)
        out = rnn(emb)
        # out is (output, hidden) for GRU or (output, (h, c)) for LSTM
        hidden = out[1][0] if self.is_lstm else out[1]
        if self.is_bidir:
            h = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            h = hidden[-1]
        return self.fc(self.dropout(h))


class TextFFN(nn.Module):
    """Embedding-based FFN — state dict has keys: embedding.weight, mlp.X.*"""
    def __init__(self, vocab_size, embed_dim, hidden_sizes, dropout, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout   = nn.Dropout(dropout)
        layers = []
        in_dim = embed_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        mask   = (x != 0).float().unsqueeze(-1)
        emb    = self.embedding(x) * mask
        pooled = emb.sum(1) / mask.sum(1).clamp(min=1)
        return self.mlp(self.dropout(pooled))


class TfidfFFN(nn.Module):
    """
    TF-IDF based FFN — NO embedding layer.
    Architecture is inferred entirely from the state dict so it always
    matches exactly, regardless of what hidden sizes the teammate used.
    Supports both:
      - 2-output softmax  (model.N.weight shape [..., 2])
      - 1-output sigmoid  (model.N.weight shape [..., 1])
    """
    def __init__(self, state_dict: dict):
        super().__init__()
        # Collect all Linear weight keys in order
        linear_keys = sorted(
            [k for k in state_dict if k.endswith(".weight")],
            key=lambda k: int(k.split(".")[1])
        )
        layers = []
        for i, wkey in enumerate(linear_keys):
            out_f, in_f = state_dict[wkey].shape
            layers.append(nn.Linear(in_f, out_f))
            # Add ReLU + Dropout between hidden layers (not after last)
            if i < len(linear_keys) - 1:
                layers += [nn.ReLU(), nn.Dropout(0.0)]  # dropout=0 at inference
        self.model     = nn.Sequential(*layers)
        self.sigmoid_out = (state_dict[linear_keys[-1]].shape[0] == 1)

    def forward(self, x):
        out = self.model(x)
        return out   # raw logits / sigmoid handled in prediction


def _infer_ffn_variant(state_dict: dict) -> str:
    """
    Look at state dict keys to decide which FFN class to use.
      'embedding.weight' present  →  TextFFN  (vocab-based)
      'model.0.weight'   present  →  TfidfFFN (TF-IDF based)
    """
    keys = list(state_dict.keys())
    if any(k.startswith("embedding") for k in keys):
        return "ffn_embed"
    if any(k.startswith("model.") for k in keys):
        return "ffn_tfidf"
    # fallback: if mlp key present it's embedding-based
    if any(k.startswith("mlp.") for k in keys):
        return "ffn_embed"
    return "ffn_tfidf"   # default to tfidf variant if unsure


# ══════════════════════════════════════════════════════════════════════════════
# TEXT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

def tokenize(text: str):
    return clean_text(text).split()

def text_to_ids(text, vocab, max_len):
    tokens = tokenize(text)
    unk_id = vocab.get("<UNK>", 1)
    pad_id = vocab.get("<PAD>", 0)
    ids    = [vocab.get(t, unk_id) for t in tokens]
    if len(ids) < max_len:
        ids    = ids + [pad_id] * (max_len - len(ids))
    else:
        ids    = ids[:max_len]
        tokens = tokens[:max_len]
    return tokens, ids


# ══════════════════════════════════════════════════════════════════════════════
# ZIP LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _pkl(zf: zipfile.ZipFile, name: str):
    for info in zf.infolist():
        if Path(info.filename).name == name:
            with zf.open(info) as f:
                try:
                    return pickle.load(f)
                except Exception:
                    return None
    return None

_WEIGHT_ALIASES = {
    "weights.pt", "model.pt", "bilstm_model_final.pt", "bilstm_model_best.pt",
    "model_best.pt", "model_final.pt", "checkpoint.pt", "model_weights.pt",
    "weights.pth", "model.pth", "model_best.pth", "model_final.pth",
    "checkpoint.pth", "kendrick_cnn_phishing.pth",
    "ffn_best_state.pth", "ffn_best_state.pt",
    "cnn_best_state.pth", "cnn_best_state.pt",
    "rnn_best_state.pth", "rnn_best_state.pt",
    "bilstm_best_state.pth", "bilstm_best_state.pt",
    "best_state.pth", "best_state.pt",
    "best_model.pth", "best_model.pt",
    "model_best.pt", "model_best.pth",
    "final_model.pth", "final_model.pt",
}

def _bytes(zf: zipfile.ZipFile, name: str):
    candidates = {name}
    if name in ("weights.pt", "weights.pth"):
        candidates = _WEIGHT_ALIASES
    for info in zf.infolist():
        if Path(info.filename).name in candidates:
            with zf.open(info) as f:
                return f.read()
    # Last resort: any .pt / .pth file
    if name in ("weights.pt", "weights.pth"):
        for info in zf.infolist():
            fn = Path(info.filename).name
            if fn.endswith(".pt") or fn.endswith(".pth"):
                with zf.open(info) as f:
                    return f.read()
    return None

def _json(zf: zipfile.ZipFile, name: str):
    for info in zf.infolist():
        if Path(info.filename).name == name:
            with zf.open(info) as f:
                return json.load(f)
    return None

def _folder_name(zf: zipfile.ZipFile, zip_filename: str = None) -> str:
    """
    Derive a clean display name for the model.
      1. If all files share a common top-level folder → use that
      2. Else if a ZIP filename is provided → use the cleaned filename
      3. Else fall back to 'unknown_model'
    """
    parts = [Path(i.filename).parts for i in zf.infolist()
             if i.filename and not i.filename.endswith("/")]
    top_dirs = {p[0] for p in parts if len(p) > 1}
    if len(top_dirs) == 1:
        return top_dirs.pop()

    # Try the explicit zip filename first, then fall back to zf.filename
    name = zip_filename or getattr(zf, "filename", None)
    if name:
        # Strip .zip extensions repeatedly (handles 'foo.zip.zip')
        clean = Path(name).name
        while clean.lower().endswith(".zip"):
            clean = clean[:-4]
        return clean or "unknown_model"
    return "unknown_model"

def detect_model_type(zf: zipfile.ZipFile) -> str:
    info = _json(zf, "model_info.json")
    if info and "type" in info:
        declared = info["type"].lower()
        if declared in ("bilstm", "rnn", "cnn", "ffn", "sklearn"):
            return declared

    filenames = {Path(i.filename).name for i in zf.infolist()}

    if "tfidf_vectorizer.pkl" in filenames:
        return "sklearn"

    has_weights = (
        any(f in filenames for f in _WEIGHT_ALIASES)
        or any(f.endswith(".pt") or f.endswith(".pth") for f in filenames)
    )
    if not has_weights:
        return "unknown"

    cfg      = _pkl(zf, "config.pkl") or {}
    cfg_keys = {k.lower() for k in cfg.keys()}

    if "kernel_sizes" in cfg_keys or "num_filters" in cfg_keys:
        return "cnn"

    if ("hidden_sizes" in cfg_keys or "ffn" in cfg_keys
            or cfg.get("model_type", "").lower() == "ffn"):
        return "ffn"

    rnn_clue = (
        "rnn_type" in cfg_keys
        or cfg.get("rnn_type", "").lower() in ("gru", "lstm", "rnn")
        or cfg.get("model_type", "").lower() in ("rnn", "gru")
        or cfg.get("bidirectional", True) is False
    )
    if rnn_clue:
        return "rnn"

    return "bilstm"


@st.cache_resource(show_spinner=False)
def build_pytorch_model(model_type: str, config_repr: str, weights_bytes: bytes):
    import ast as _ast
    config = _ast.literal_eval(config_repr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vs  = int(config.get("vocab_size", 30000))
    ed  = int(config.get("embedding_dim", 128))
    dr  = float(config.get("dropout", 0.5))

    # Load state dict first so we can inspect keys for FFN variant detection
    state = torch.load(io.BytesIO(weights_bytes), map_location=device,
                       weights_only=False)

    if model_type == "bilstm":
        # Use TextRNN in bidir mode — infers exact architecture from state dict
        # so it works even if teammate used different hidden sizes / layer counts
        m = TextRNN(state_dict=state, vocab_size=vs, embed_dim=ed, dropout=dr).to(device)

    elif model_type == "rnn":
        # Also inferred from state dict — handles unidirectional LSTM/GRU automatically
        m = TextRNN(state_dict=state, vocab_size=vs, embed_dim=ed, dropout=dr).to(device)

    elif model_type == "cnn":
        m = TextCNN(state_dict=state, vocab_size=vs, embed_dim=ed, dropout=dr).to(device)

    elif model_type == "ffn":
        hs = config.get("hidden_sizes", config.get("hidden_size", [256, 128]))
        if isinstance(hs, int):
            hs = [hs]

        # ── KEY FIX: peek at state dict to choose correct FFN variant ──
        variant = _infer_ffn_variant(state)

        if variant == "ffn_tfidf":
            # Build architecture entirely from state dict — no guessing needed
            m = TfidfFFN(state_dict=state).to(device)
        else:
            # Embedding-based FFN
            m = TextFFN(
                vocab_size=vs, embed_dim=ed,
                hidden_sizes=hs, dropout=dr,
            ).to(device)

    else:
        raise ValueError(
            f"Unknown pytorch model type: '{model_type}'. "
            "Expected one of: bilstm, rnn, cnn, ffn. "
            'Add model_info.json with {"type": "<type>"} to your folder.'
        )

    m.load_state_dict(state)
    m.eval()
    return m, device, model_type + ("_tfidf" if model_type == "ffn"
                                     and _infer_ffn_variant(state) == "ffn_tfidf"
                                     else "")


def load_model_from_zip(zip_bytes: bytes, zip_filename: str = None) -> dict:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        name      = _folder_name(zf, zip_filename=zip_filename)
        mtype     = detect_model_type(zf)
        config    = _pkl(zf, "config.pkl") or {}
        vocab     = (_pkl(zf, "vocab.pkl")
                     or _pkl(zf, "tokenizer.pkl")
                     or _pkl(zf, "vectorizer.pkl")
                     or {})
        label_map = _pkl(zf, "label_map.pkl")
        history   = _pkl(zf, "history.pkl")
        max_len   = int(config.get("max_len", 256))

        if label_map is None:
            label_map = {0: "Legitimate", 1: "Phishing"}
        label_map = {int(k): str(v) for k, v in label_map.items()}
        phish_idx = next((k for k, v in label_map.items() if "phish" in v.lower()), 1)
        legit_idx = 1 - phish_idx

        if mtype == "sklearn":
            tfidf = _pkl(zf, "tfidf_vectorizer.pkl")
            model = _pkl(zf, "weights.pt") or _pkl(zf, "model.pkl")
            if model is None:
                wb = _bytes(zf, "weights.pt")
                if wb:
                    model = pickle.loads(wb)
            return dict(name=name, model_type="sklearn", model_obj=model,
                        device=None, vocab={}, config=config,
                        label_map=label_map, phish_idx=phish_idx,
                        legit_idx=legit_idx, max_len=max_len,
                        tfidf=tfidf, history=history,
                        is_sklearn=True, is_tfidf_ffn=False)

        elif mtype in ("bilstm", "cnn", "rnn", "ffn"):
            wb = _bytes(zf, "weights.pt")
            if wb is None:
                raise ValueError(f"[{name}] weights.pt not found in ZIP")

            model, device, resolved_type = build_pytorch_model(mtype, repr(config), wb)
            is_tfidf_ffn = resolved_type.endswith("_tfidf")

            # For TfidfFFN we still need the sklearn vectorizer to transform text
            tfidf_vec = None
            if is_tfidf_ffn:
                tfidf_vec = (_pkl(zf, "tfidf_vectorizer.pkl")
                             or _pkl(zf, "vocab.pkl")
                             or _pkl(zf, "vectorizer.pkl"))

            return dict(name=name, model_type=mtype, model_obj=model,
                        device=device, vocab=vocab, config=config,
                        label_map=label_map, phish_idx=phish_idx,
                        legit_idx=legit_idx, max_len=max_len,
                        tfidf=tfidf_vec, history=history,
                        is_sklearn=False, is_tfidf_ffn=is_tfidf_ffn)

        else:
            found_files = [Path(i.filename).name for i in zf.infolist()
                           if not i.filename.endswith("/")]
            raise ValueError(
                f"[{name}] Could not detect model type.\n"
                f"Files found in ZIP: {found_files}\n"
                f'Add a model_info.json with {{"type": "bilstm"}} '
                f"(or rnn/cnn/ffn/sklearn) to your folder."
            )


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _pytorch_probs(bundle: dict, ids: list):
    model, device = bundle["model_obj"], bundle["device"]
    x   = torch.tensor([ids], dtype=torch.long, device=device)
    raw = model(x)
    # Handle sigmoid (1-output) vs softmax (2-output)
    sigmoid_out = getattr(model, "sigmoid_out", False)
    if sigmoid_out:
        phish = float(torch.sigmoid(raw).cpu().numpy()[0][0])
        legit = 1.0 - phish
        if bundle["phish_idx"] == 0:
            legit, phish = phish, legit
    else:
        probs = F.softmax(raw, dim=1).cpu().numpy()[0]
        legit = float(probs[bundle["legit_idx"]])
        phish = float(probs[bundle["phish_idx"]])
    return legit, phish

@torch.no_grad()
def _tfidf_ffn_probs(bundle: dict, text: str):
    """
    TfidfFFN: vectorise with sklearn vectorizer, run through PyTorch MLP.
    Handles both sigmoid (1-output) and softmax (2-output) final layers.
    """
    import numpy as np
    vec   = bundle["tfidf"].transform([text])
    x     = torch.tensor(vec.toarray(), dtype=torch.float32, device=bundle["device"])
    model = bundle["model_obj"]
    raw   = model(x)

    if model.sigmoid_out:
        # 1-output sigmoid: shape (1, 1)
        phish = float(torch.sigmoid(raw).cpu().numpy()[0][0])
        legit = 1.0 - phish
        # Respect label map
        if bundle["phish_idx"] == 0:
            legit, phish = phish, legit
    else:
        # 2-output softmax: shape (1, 2)
        probs = F.softmax(raw, dim=1).cpu().numpy()[0]
        legit = float(probs[bundle["legit_idx"]])
        phish = float(probs[bundle["phish_idx"]])
    return legit, phish

def _sklearn_probs(bundle: dict, text: str):
    tfidf = bundle["tfidf"]
    model = bundle["model_obj"]
    vec   = tfidf.transform([text])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        return float(probs[bundle["legit_idx"]]), float(probs[bundle["phish_idx"]])
    pred  = model.predict(vec)[0]
    legit = 1.0 if pred == bundle["legit_idx"] else 0.0
    return legit, 1.0 - legit

def predict_with_tokens(bundle: dict, text: str):
    if bundle["is_sklearn"]:
        l, p = _sklearn_probs(bundle, text)
        return [], [], l, p
    if bundle.get("is_tfidf_ffn"):
        l, p = _tfidf_ffn_probs(bundle, text)
        return [], [], l, p
    tokens, ids = text_to_ids(text, bundle["vocab"], bundle["max_len"])
    l, p = _pytorch_probs(bundle, ids)
    return tokens, ids, l, p


# ══════════════════════════════════════════════════════════════════════════════
# LEAVE-ONE-OUT EXPLANATION (embedding-based PyTorch only)
# ══════════════════════════════════════════════════════════════════════════════

def explain_loo(bundle: dict, tokens: list, ids: list) -> list:
    if bundle["is_sklearn"] or bundle.get("is_tfidf_ffn") or not tokens:
        return []
    model, device = bundle["model_obj"], bundle["device"]
    vocab  = bundle["vocab"]
    pad_id = vocab.get("<PAD>", 0)

    sigmoid_out = getattr(model, "sigmoid_out", False)

    @torch.no_grad()
    def _phish(id_list):
        x   = torch.tensor([id_list], dtype=torch.long, device=device)
        raw = model(x)
        if sigmoid_out:
            phish = float(torch.sigmoid(raw).cpu().numpy()[0][0])
            if bundle["phish_idx"] == 0:
                phish = 1.0 - phish
            return phish
        probs = F.softmax(raw, dim=1).cpu().numpy()[0]
        return float(probs[bundle["phish_idx"]])

    base = _phish(ids)
    out  = []
    for i, tok in enumerate(tokens):
        mod = list(ids); mod[i] = pad_id
        out.append((tok, float(base - _phish(mod))))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# HEURISTICS
# ══════════════════════════════════════════════════════════════════════════════

PHISH_KW = [
    "verify","account","suspend","click","update","password","confirm","login",
    "urgent","immediately","bank","credit","security","alert","limited","offer",
    "free","winner","prize","congratulations","claim","expire","access",
    "unauthorized","unusual","activity","validate","reset","invoice","payment",
]

def extract_features(text: str) -> dict:
    lower = text.lower()
    urls  = re.findall(r"https?://\S+", text)
    ip_u  = [u for u in urls if re.search(r"https?://\d+\.\d+\.\d+\.\d+", u)]
    kw    = [k for k in PHISH_KW if k in lower]
    words = tokenize(text)
    caps  = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    return {
        "total_tokens": len(words), "unique_tokens": len(set(words)),
        "url_count": len(urls), "ip_based_urls": len(ip_u),
        "phish_keywords_found": kw, "phish_keyword_count": len(kw),
        "exclamation_marks": text.count("!"), "question_marks": text.count("?"),
        "caps_ratio": round(caps * 100, 1),
        "contains_html": bool(re.search(r"<[a-z]+[\s>]", text, re.IGNORECASE)),
        "avg_word_length": round(sum(len(w) for w in words) / max(len(words), 1), 2),
        "top_words": Counter(words).most_common(10), "url_list": urls,
    }

def risk_scores(features: dict, phish_prob: float) -> dict:
    return {
        "Model Confidence": round(phish_prob * 100, 1),
        "Keyword Risk":     round(min(features["phish_keyword_count"] / len(PHISH_KW) * 100, 100), 1),
        "URL Risk":         round(min(features["url_count"] * 10 + features["ip_based_urls"] * 30, 100), 1),
        "Writing Style":    round(min(features["exclamation_marks"] * 8 + features["caps_ratio"] * 2, 100), 1),
        "HTML Presence":    60.0 if features["contains_html"] else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CHART / HTML BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def heatmap_html(importances):
    """
    Render the per-token heatmap with anti-saturation logic.

    Problem: relative scaling makes every token look identical when one model
    is very confident (all scores tiny but all near max → all dark). 

    Fix: use a percentile-based scale (top 90th percentile = full saturation)
    so a few outliers don't squash everything else into invisibility, AND
    apply a noise floor — tokens below the 25th percentile get rendered as
    neutral grey instead of faintly colored, so meaningful signal stands out.
    """
    if not importances:
        return "<p style='color:#6b7280'>No token importance available for this model type.</p>"

    scores      = [abs(s) for _, s in importances]
    sorted_abs  = sorted(scores)
    n           = len(sorted_abs)
    p25         = sorted_abs[int(n * 0.25)] if n > 4 else 0
    p90         = sorted_abs[int(n * 0.90)] if n > 0 else 1e-8
    scale_max   = max(p90, 1e-6)
    noise_floor = max(p25, 1e-7)

    parts = []
    for tok, score in importances:
        abs_s = abs(score)
        # Tokens below noise floor → neutral grey
        if abs_s < noise_floor:
            parts.append(
                f'<span title="≈0 (below noise floor)" class="pill" '
                f'style="background:#f3f4f6;border:1px solid #e5e7eb;color:#9ca3af">'
                f'{escape(tok)}</span>'
            )
            continue

        # Above floor: scale relative to 90th percentile, capped at 1.0
        intensity = min(abs_s / scale_max, 1.0)
        if score > 0:           # pushes → phishing
            a   = 0.18 + 0.65 * intensity
            bg  = f"rgba(224,45,71,{a:.3f})"
            brd = f"rgba(224,45,71,{min(a+.25,1):.3f})"
            tc  = "#7f1d1d" if intensity > .55 else "#111827"
            lbl = f"▲ {score:+.4f} → phishing"
        else:                   # pushes → legitimate
            a   = 0.18 + 0.65 * intensity
            bg  = f"rgba(5,150,105,{a:.3f})"
            brd = f"rgba(5,150,105,{min(a+.25,1):.3f})"
            tc  = "#064e3b" if intensity > .55 else "#111827"
            lbl = f"▼ {score:.4f} → legitimate"

        parts.append(
            f'<span title="{escape(lbl)}" class="pill" '
            f'style="background:{bg};border:1px solid {brd};color:{tc}">'
            f'{escape(tok)}</span>'
        )

    legend = (
        '<div style="display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap;align-items:center">'
        '<span class="badge"><span style="width:9px;height:9px;border-radius:50%;'
        'background:#e02d47;display:inline-block"></span> Pushes → Phishing</span>'
        '<span class="badge"><span style="width:9px;height:9px;border-radius:50%;'
        'background:#059669;display:inline-block"></span> Pushes → Legitimate</span>'
        '<span class="badge"><span style="width:9px;height:9px;border-radius:50%;'
        'background:#d1d5db;display:inline-block"></span> Negligible</span>'
        '<span class="badge">🖱 Hover for score</span></div>'
    )
    return legend + '<div style="line-height:2.4">' + "".join(parts) + "</div>"


def gauge_chart(phish_prob):
    c   = "#e02d47" if phish_prob >= .5 else "#059669"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(phish_prob * 100, 1),
        number={"suffix": "%", "font": {"color": c, "size": 34, "family": "Space Mono"}},
        gauge={"axis": {"range": [0,100], "tickfont": {"color":"#6b7280","size":10}},
               "bar": {"color": c, "thickness": .25}, "bgcolor": "#eef1f6",
               "bordercolor": "#d1d5db",
               "steps": [{"range":[0,30],"color":"#dcfce7"},
                         {"range":[30,60],"color":"#fefce8"},
                         {"range":[60,100],"color":"#ffe4e6"}],
               "threshold": {"line":{"color":c,"width":3},"thickness":.8,"value":phish_prob*100}},
    ))
    fig.update_layout(paper_bgcolor="#ffffff", height=200, margin=dict(l=20,r=20,t=15,b=15))
    return fig


def token_bar(importances, top_n=15):
    top    = sorted(importances, key=lambda x: abs(x[1]), reverse=True)[:top_n]
    top    = sorted(top, key=lambda x: x[1])
    toks   = [t for t,_ in top]
    scores = [s for _,s in top]
    fig    = go.Figure(go.Bar(
        x=scores, y=toks, orientation="h",
        marker_color=["#e02d47" if s > 0 else "#059669" for s in scores],
        marker_line_width=0,
        text=[f"{s:+.4f}" for s in scores], textposition="outside",
        textfont=dict(color="#374151", size=9, family="Space Mono"),
    ))
    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#f5f7fa",
        xaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#2563eb",
                   tickfont=dict(color="#6b7280"),
                   title=dict(text="← Legitimate  ·  Phishing →",
                              font=dict(color="#6b7280", size=10))),
        yaxis=dict(gridcolor="#e5e7eb",
                   tickfont=dict(color="#111827", family="Space Mono", size=10)),
        margin=dict(l=10,r=70,t=10,b=10),
        height=max(280, top_n*24), bargap=.3,
    )
    return fig


def radar_chart(rscores, is_phish):
    cats  = list(rscores.keys()); vals = list(rscores.values())
    color = "#e02d47" if is_phish else "#059669"
    fill  = "rgba(224,45,71,0.12)" if is_phish else "rgba(5,150,105,0.12)"
    fig   = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]],
        fill="toself", fillcolor=fill,
        line=dict(color=color, width=2), marker=dict(size=5, color=color),
    ))
    fig.update_layout(
        polar=dict(bgcolor="#ffffff",
                   radialaxis=dict(visible=True, range=[0,100], gridcolor="#e5e7eb",
                                   tickfont=dict(color="#6b7280",size=9),
                                   tickvals=[25,50,75,100]),
                   angularaxis=dict(gridcolor="#e5e7eb",
                                    tickfont=dict(color="#111827",size=10))),
        showlegend=False, paper_bgcolor="#ffffff",
        margin=dict(l=40,r=40,t=30,b=30), height=270,
    )
    return fig


def training_curves(history: dict):
    epochs = history.get("epoch", [])
    if not epochs:
        return None
    fig   = go.Figure()
    lines = [("train_loss","#f59e0b","Train Loss"),
             ("val_acc","#4f8ef7","Val Accuracy"),
             ("val_f1","#059669","Val F1"),
             ("val_prec","#7c3aed","Val Precision"),
             ("val_rec","#d97706","Val Recall")]
    for key, col, name in lines:
        if key in history and history[key]:
            fig.add_trace(go.Scatter(
                x=epochs, y=history[key], mode="lines+markers",
                name=name, line=dict(color=col, width=2),
                marker=dict(size=4, color=col)))
    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#f5f7fa",
        legend=dict(font=dict(color="#111827",size=10),
                    bgcolor="#ffffff", bordercolor="#d1d5db"),
        xaxis=dict(gridcolor="#e5e7eb", tickfont=dict(color="#6b7280"),
                   title=dict(text="Epoch", font=dict(color="#6b7280"))),
        yaxis=dict(gridcolor="#e5e7eb", tickfont=dict(color="#6b7280")),
        margin=dict(l=10,r=10,t=10,b=10), height=300,
    )
    return fig


def compare_bar(results: list):
    names  = [r["name"] for r in results]
    phish  = [r["phish_prob"] for r in results]
    legit  = [r["legit_prob"] for r in results]
    colors = ["#e02d47" if p >= .5 else "#059669" for p in phish]
    fig    = go.Figure()
    fig.add_trace(go.Bar(name="P(Phishing)", x=names, y=phish,
                         marker_color=colors, marker_line_width=0))
    fig.add_trace(go.Bar(name="P(Legitimate)", x=names, y=legit,
                         marker_color=["rgba(37,99,235,.5)"]*len(names),
                         marker_line_width=0))
    fig.add_hline(y=.5, line_dash="dot", line_color="#9ca3af",
                  annotation_text="Decision threshold",
                  annotation_font_color="#6b7280")
    fig.update_layout(
        barmode="group", paper_bgcolor="#ffffff", plot_bgcolor="#f5f7fa",
        legend=dict(font=dict(color="#111827",size=10),
                    bgcolor="#ffffff", bordercolor="#d1d5db"),
        xaxis=dict(tickfont=dict(color="#111827",family="Space Mono",size=11),
                   gridcolor="#e5e7eb"),
        yaxis=dict(range=[0,1], tickfont=dict(color="#6b7280"),
                   gridcolor="#e5e7eb",
                   title=dict(text="Probability", font=dict(color="#6b7280"))),
        margin=dict(l=10,r=10,t=10,b=10), height=300,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "models"       not in st.session_state: st.session_state.models       = {}
if "history_log"  not in st.session_state: st.session_state.history_log  = []
if "active_model" not in st.session_state: st.session_state.active_model = None

# Model type → display colour (used in sidebar and main)
_TYPE_COLORS = {
    "bilstm": "#1d4ed8", "rnn": "#047857", "cnn": "#c2410c",
    "ffn": "#92400e", "sklearn": "#5b21b6",
}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding:14px 0 20px">
      <div style="font-family:'Space Mono',monospace;font-size:1.25rem;font-weight:700;color:#111827">🎣 PhishNet</div>
      <div style="font-size:.68rem;color:#6b7280;letter-spacing:2px;text-transform:uppercase;margin-top:2px">Multi-Model Detector</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="slabel">Load Models (ZIP files)</div>', unsafe_allow_html=True)

    # Single file uploader — user uploads one ZIP, clicks Add, repeat.
    # This avoids Streamlit rerun race conditions when loading multiple models.
    uploaded_zip = st.file_uploader(
        "Choose a ZIP file", type=["zip"],
        label_visibility="collapsed",
        key="zip_uploader",
    )

    add_clicked = st.button("➕ Add Model", type="primary",
                             disabled=uploaded_zip is None)

    if add_clicked and uploaded_zip is not None:
        file_id = f"{uploaded_zip.name}__{uploaded_zip.size}"
        already_loaded = file_id in st.session_state.get("zip_registry", {})
        if already_loaded:
            st.warning(f"'{uploaded_zip.name}' is already loaded.")
        else:
            with st.spinner(f"Loading {uploaded_zip.name}…"):
                try:
                    uploaded_zip.seek(0)
                    bundle = load_model_from_zip(uploaded_zip.read(),
                                                  zip_filename=uploaded_zip.name)
                    if "zip_registry" not in st.session_state:
                        st.session_state.zip_registry = {}
                    st.session_state.models[bundle["name"]] = bundle
                    st.session_state.active_model = bundle["name"]
                    st.session_state.zip_registry[file_id] = bundle["name"]
                    variant = " · TF-IDF" if bundle.get("is_tfidf_ffn") else ""
                    st.success(f"✓ {bundle['name']} ({bundle['model_type'].upper()}{variant}) added!")
                    st.rerun()
                except Exception as e:
                    import traceback
                    st.error(f"Failed to load: {e}")
                    with st.expander("Full error details"):
                        st.code(traceback.format_exc())

    # ── Loaded model list ────────────────────────────────────────────────────
    if st.session_state.models:
        st.markdown("---")
        n = len(st.session_state.models)
        st.markdown(
            f'<div class="slabel">Loaded Models '
            f'<span style="background:#2563eb;color:#fff;border-radius:10px;'
            f'padding:1px 7px;font-size:.65rem;margin-left:4px">{n}</span></div>',
            unsafe_allow_html=True,
        )
        for mname, bundle in list(st.session_state.models.items()):
            is_active = mname == st.session_state.active_model
            mtype     = bundle["model_type"].upper()
            variant   = " · TF-IDF" if bundle.get("is_tfidf_ffn") else ""
            tcolor    = _TYPE_COLORS.get(bundle["model_type"], "#4b5563")
            bg        = "#dbeafe" if is_active else "#f9fafb"
            border    = "#2563eb" if is_active else "#e5e7eb"

            col_a, col_b = st.columns([5, 1])
            with col_a:
                st.markdown(
                    f'<div style="background:{bg};border:1.5px solid {border};'
                    f'border-radius:8px;padding:9px 12px;margin-bottom:2px">'
                    f'<div style="font-family:Space Mono,monospace;font-size:.78rem;'
                    f'color:#111827;font-weight:{"700" if is_active else "400"}">'
                    f'{"▶ " if is_active else "○ "}{mname}</div>'
                    f'<div style="font-size:.65rem;color:{tcolor};margin-top:2px">'
                    f'{mtype}{variant}</div></div>',
                    unsafe_allow_html=True,
                )
                if not is_active:
                    if st.button("Select", key=f"sel_{mname}"):
                        st.session_state.active_model = mname
                        st.rerun()
            with col_b:
                st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
                if st.button("✕", key=f"del_{mname}"):
                    del st.session_state.models[mname]
                    # Remove from registry so it can be re-uploaded if needed
                    st.session_state.zip_registry = {
                        k: v for k, v in st.session_state.zip_registry.items()
                        if v != mname
                    }
                    if st.session_state.active_model == mname:
                        rem = list(st.session_state.models.keys())
                        st.session_state.active_model = rem[0] if rem else None
                    st.rerun()

        st.markdown("---")
        st.markdown('<div class="slabel">Settings</div>', unsafe_allow_html=True)
        top_n     = st.slider("Top tokens (bar chart)", 5, 30, 15)
        show_norm = st.checkbox("Show normalized text", False)

    if st.session_state.history_log:
        st.markdown("---")
        st.markdown('<div class="slabel">Scan History</div>', unsafe_allow_html=True)
        for item in reversed(st.session_state.history_log[-6:]):
            dot = "dot-r" if item["label"] == "Phishing" else "dot-g"
            st.markdown(
                f'<div class="hrow"><div class="{dot}"></div>'
                f'<div style="flex:1;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;'
                f'font-size:.76rem">{item["preview"]}</div>'
                f'<div style="font-family:Space Mono,monospace;font-size:.7rem;color:#6b7280">'
                f'{item["model"][:10]}</div>'
                f'<div style="font-family:Space Mono,monospace;font-size:.7rem;color:#6b7280">'
                f'{item["prob"]:.0%}</div></div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="margin-bottom:24px">
  <h1 style="font-size:2.1rem;margin:0;
     background:linear-gradient(90deg,#2563eb,#059669);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">
    Email Threat Analysis
  </h1>
  <p style="color:#6b7280;margin-top:5px;font-size:.88rem">
    Multi-model phishing detection · Load team model ZIPs from the sidebar · Single or compare mode
  </p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.models:
    st.markdown("""
    <div class="card" style="text-align:center;padding:48px 32px">
      <div style="font-size:3rem;margin-bottom:12px">🎣</div>
      <div style="font-family:'Space Mono',monospace;font-size:1.05rem;color:#111827;margin-bottom:8px">
        No models loaded yet
      </div>
      <div style="color:#6b7280;font-size:.84rem;max-width:520px;margin:0 auto;line-height:1.7">
        Zip your model folder and upload it from the sidebar.
      </div>
      <div style="margin-top:20px;display:flex;gap:8px;justify-content:center;flex-wrap:wrap">
        <span class="badge">weights.pt / weights.pth</span>
        <span class="badge">config.pkl</span>
        <span class="badge">vocab.pkl / tokenizer.pkl / vectorizer.pkl</span>
        <span class="badge">label_map.pkl</span>
        <span class="badge">model_info.json</span>
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, (icon, title, desc) in zip([c1,c2,c3], [
        ("📦", "ZIP your folder", "Zip your model folder — the app auto-detects BiLSTM, RNN, CNN, FFN (embedding or TF-IDF), and sklearn."),
        ("🤖", "Auto-detection", "Peeks at state dict keys to distinguish embedding-based FFN (mlp.*) from TF-IDF FFN (model.*). No config changes needed."),
        ("⚖️", "Compare mode",   "Load 2+ models and click Compare to run all on the same email with a side-by-side probability chart and ensemble vote."),
    ]):
        with col:
            st.markdown(
                f'<div class="card" style="text-align:center;padding:22px 14px">'
                f'<div style="font-size:1.7rem;margin-bottom:8px">{icon}</div>'
                f'<div style="font-family:Space Mono,monospace;font-size:.82rem;color:#111827;margin-bottom:6px">{title}</div>'
                f'<div style="color:#6b7280;font-size:.76rem;line-height:1.5">{desc}</div>'
                f'</div>', unsafe_allow_html=True,
            )
    st.stop()


# ── Model selector strip ──────────────────────────────────────────────────────
models_loaded = st.session_state.models
active_name   = st.session_state.active_model or list(models_loaded.keys())[0]

st.markdown('<div class="slabel">Active Model</div>', unsafe_allow_html=True)
model_cols = st.columns(min(len(models_loaded), 5))
for col, (mname, bundle) in zip(model_cols, models_loaded.items()):
    with col:
        is_active = mname == active_name
        tag_col   = _TYPE_COLORS.get(bundle["model_type"], "#4b5563")
        sub_label = bundle["model_type"].upper() + (" (TF-IDF)" if bundle.get("is_tfidf_ffn") else "")
        brd_col   = "#2563eb" if is_active else "#d1d5db"
        bg_col    = "rgba(37,99,235,.06)" if is_active else "#ffffff"
        st.markdown(
            f'<div style="background:{bg_col};border:1.5px solid {brd_col};'
            f'border-radius:10px;padding:12px 14px;text-align:center">'
            f'<div style="font-family:Space Mono,monospace;font-size:.82rem;color:#111827;'
            f'font-weight:{"700" if is_active else "400"}">{mname}</div>'
            f'<div style="font-size:.65rem;color:{tag_col};margin-top:3px">{sub_label}</div>'
            f'</div>', unsafe_allow_html=True,
        )
        if not is_active:
            if st.button("Select", key=f"pick_{mname}"):
                st.session_state.active_model = mname
                st.rerun()

st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

ci, cu = st.columns([3, 1])
with ci:
    email_text = st.text_area(
        "Email body", height=220,
        placeholder="Paste the full email body here…",
        label_visibility="collapsed",
    )
with cu:
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    email_upload = st.file_uploader("Upload .txt / .eml", type=["txt","eml"],
                                     label_visibility="collapsed")
    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    analyze_btn = st.button("⚡ Analyze Email", type="primary")
    compare_btn = st.button("⚖️ Compare All Models",
                             disabled=len(models_loaded) < 2,
                             help="Run all loaded models on this email")


def get_source_text():
    txt = email_text.strip()
    if email_upload and not txt:
        email_upload.seek(0)
        raw = email_upload.read()
        try:    return raw.decode("utf-8")
        except: return raw.decode("latin-1", errors="ignore")
    return txt


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE MODEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

if analyze_btn:
    source = get_source_text()
    if not source:
        st.error("⚠️  Paste or upload an email first.")
    else:
        bundle = models_loaded[active_name]
        with st.spinner(f"Running {active_name}…"):
            tokens, ids, legit_prob, phish_prob = predict_with_tokens(bundle, source)
            label       = "Phishing" if phish_prob >= .5 else "Legitimate"
            importances = explain_loo(bundle, tokens, ids)
            feats       = extract_features(source)
            rsc         = risk_scores(feats, phish_prob)
            is_phish    = phish_prob >= .5

        st.session_state.history_log.append({
            "label": label, "prob": phish_prob, "model": active_name,
            "preview": source[:55].replace("\n"," ") + "…",
        })

        vc   = "verdict-p" if is_phish else "verdict-l"
        vcol = "#e02d47"   if is_phish else "#059669"
        icon = "🚨" if is_phish else "✅"
        conf = ("HIGH"     if abs(phish_prob-.5) > .35 else
                "MODERATE" if abs(phish_prob-.5) > .10 else "LOW")
        variant_note = " · TF-IDF FFN" if bundle.get("is_tfidf_ffn") else ""

        st.markdown(f"""
        <div class="{vc}" style="margin-bottom:18px">
          <p class="vtitle" style="color:{vcol}">{icon} {label.upper()}</p>
          <p class="vsub">{conf} CONFIDENCE · Model: <b>{active_name}</b>
             ({bundle["model_type"].upper()}{variant_note}) ·
             P(phishing)={phish_prob:.4f} · P(legit)={legit_prob:.4f}</p>
        </div>""", unsafe_allow_html=True)

        mc = st.columns(5)
        for col2, (lbl, val, clr) in zip(mc, [
            ("P(Phishing)",    f"{phish_prob:.4f}", "#e02d47" if is_phish else "#059669"),
            ("P(Legitimate)",  f"{legit_prob:.4f}", "#2563eb"),
            ("Tokens",         str(feats["total_tokens"]), "#d97706"),
            ("URLs",           str(feats["url_count"]), "#7c3aed"),
            ("Phish Keywords", str(feats["phish_keyword_count"]), "#c2410c"),
        ]):
            with col2:
                st.markdown(f'<div class="mtile">'
                            f'<div class="mval" style="color:{clr}">{val}</div>'
                            f'<div class="mlbl">{lbl}</div></div>',
                            unsafe_allow_html=True)

        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

        tab_names = ["🔍 Explanation", "📡 Risk Radar", "📊 Stats", "🔗 URLs"]
        if bundle.get("history"):
            tab_names.append("📈 Training")
        tabs = st.tabs(tab_names)

        with tabs[0]:
            cg, cb = st.columns([1, 2])
            with cg:
                st.markdown('<div class="slabel">Phishing Probability</div>', unsafe_allow_html=True)
                st.plotly_chart(gauge_chart(phish_prob), use_container_width=True,
                                config={"displayModeBar": False})
            with cb:
                st.markdown('<div class="slabel">Top Token Importances</div>', unsafe_allow_html=True)
                if importances:
                    st.plotly_chart(token_bar(importances, top_n), use_container_width=True,
                                    config={"displayModeBar": False})
                else:
                    st.info("Token-level explanation not available for TF-IDF or sklearn models.")

            if importances:
                st.markdown('<div class="slabel" style="margin-top:4px">Full Email Heatmap</div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="card">{heatmap_html(importances)}</div>',
                            unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                tp = sorted(importances, key=lambda x: x[1], reverse=True)[:10]
                tl = sorted(importances, key=lambda x: x[1])[:10]
                with c1:
                    st.markdown('<div class="slabel">Top Phishing Tokens</div>', unsafe_allow_html=True)
                    df_p = pd.DataFrame(tp, columns=["Token","Impact"]).round(5)
                    st.dataframe(df_p, use_container_width=True, hide_index=True,
                                 column_config={"Impact": st.column_config.ProgressColumn(
                                     format="%.5f", min_value=0,
                                     max_value=float(df_p["Impact"].max() or 1))})
                with c2:
                    st.markdown('<div class="slabel">Top Legit Tokens</div>', unsafe_allow_html=True)
                    df_l = pd.DataFrame([(t,abs(s)) for t,s in tl],
                                        columns=["Token","Legit Score"]).round(5)
                    st.dataframe(df_l, use_container_width=True, hide_index=True,
                                 column_config={"Legit Score": st.column_config.ProgressColumn(
                                     format="%.5f", min_value=0,
                                     max_value=float(df_l["Legit Score"].max() or 1))})

            if show_norm:
                with st.expander("Normalized text seen by model"):
                    st.code(clean_text(source))

        with tabs[1]:
            r1, r2 = st.columns([1, 1])
            with r1:
                st.markdown('<div class="slabel">Risk Radar</div>', unsafe_allow_html=True)
                st.plotly_chart(radar_chart(rsc, is_phish), use_container_width=True,
                                config={"displayModeBar": False})
            with r2:
                st.markdown('<div class="slabel">Dimension Breakdown</div>', unsafe_allow_html=True)
                for dim, score in rsc.items():
                    bc = "#e02d47" if score > 60 else "#d97706" if score > 30 else "#059669"
                    st.markdown(
                        f'<div style="margin-bottom:10px">'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px">'
                        f'<span style="font-size:.8rem;color:#111827">{dim}</span>'
                        f'<span style="font-family:Space Mono,monospace;font-size:.78rem;'
                        f'color:{bc}">{score:.0f}/100</span></div>'
                        f'<div style="background:#e5e7eb;border-radius:4px;height:5px">'
                        f'<div style="width:{score}%;background:{bc};height:5px;border-radius:4px">'
                        f'</div></div></div>', unsafe_allow_html=True,
                    )
                if feats["phish_keywords_found"]:
                    st.markdown('<div class="slabel" style="margin-top:14px">Matched Keywords</div>',
                                unsafe_allow_html=True)
                    st.markdown(" ".join(
                        f'<span class="badge" style="background:rgba(224,45,71,.1);'
                        f'border-color:rgba(224,45,71,.3);color:#9f1239">{kw}</span>'
                        for kw in feats["phish_keywords_found"]
                    ), unsafe_allow_html=True)

        with tabs[2]:
            ss1, ss2 = st.columns([1, 1])
            with ss1:
                st.markdown('<div class="slabel">Email Statistics</div>', unsafe_allow_html=True)
                for k, v in {
                    "Total tokens":    feats["total_tokens"],
                    "Unique tokens":   feats["unique_tokens"],
                    "Avg word length": feats["avg_word_length"],
                    "Exclamation !":   feats["exclamation_marks"],
                    "Questions ?":     feats["question_marks"],
                    "CAPS ratio":      f'{feats["caps_ratio"]}%',
                    "Contains HTML":   "Yes ⚠️" if feats["contains_html"] else "No ✓",
                    "IP-based URLs":   feats["ip_based_urls"],
                }.items():
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'padding:7px 0;border-bottom:1px solid #e5e7eb">'
                        f'<span style="color:#6b7280;font-size:.8rem">{k}</span>'
                        f'<span style="font-family:Space Mono,monospace;font-size:.8rem;'
                        f'color:#111827">{v}</span></div>',
                        unsafe_allow_html=True,
                    )
            with ss2:
                st.markdown('<div class="slabel">Most Frequent Words</div>', unsafe_allow_html=True)
                if feats["top_words"]:
                    wds, cts = zip(*feats["top_words"])
                    wfig = go.Figure(go.Bar(x=list(wds), y=list(cts),
                                            marker_color="#2563eb", marker_line_width=0))
                    wfig.update_layout(
                        paper_bgcolor="#ffffff", plot_bgcolor="#f5f7fa",
                        xaxis=dict(tickfont=dict(color="#111827",family="Space Mono",size=10),
                                   gridcolor="#e5e7eb"),
                        yaxis=dict(tickfont=dict(color="#6b7280"),gridcolor="#e5e7eb"),
                        margin=dict(l=10,r=10,t=10,b=10), height=210, bargap=.25)
                    st.plotly_chart(wfig, use_container_width=True,
                                    config={"displayModeBar": False})

        with tabs[3]:
            if feats["url_list"]:
                for url in feats["url_list"]:
                    has_ip  = bool(re.search(r"https?://\d+\.\d+\.\d+\.\d+", url))
                    is_long = len(url) > 80
                    is_susp = any(k in url.lower()
                                  for k in ["login","verify","account","confirm","secure"])
                    flags = []
                    if has_ip:   flags.append("🔴 IP-based URL")
                    if is_long:  flags.append("🟡 Unusually long")
                    if is_susp:  flags.append("🔴 Suspicious keyword in path")
                    st.markdown(
                        f'<div class="card" style="padding:12px 16px;margin-bottom:8px">'
                        f'<div style="font-family:Space Mono,monospace;font-size:.76rem;'
                        f'color:#2563eb;word-break:break-all">{escape(url)}</div>'
                        f'<div style="font-size:.72rem;margin-top:5px;color:#6b7280">'
                        f'{"  ·  ".join(flags) if flags else "🟢 No immediate flags"}</div>'
                        f'</div>', unsafe_allow_html=True,
                    )
            else:
                st.markdown('<div class="card" style="text-align:center;color:#6b7280;'
                            'padding:36px">No URLs detected.</div>', unsafe_allow_html=True)

        if bundle.get("history") and len(tabs) == 5:
            with tabs[4]:
                fig_t = training_curves(bundle["history"])
                if fig_t:
                    st.plotly_chart(fig_t, use_container_width=True,
                                    config={"displayModeBar": False})
                vf1 = bundle["history"].get("val_f1", [])
                if vf1:
                    bi = int(max(range(len(vf1)), key=lambda i: vf1[i]))
                    st.markdown(
                        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px">'
                        f'<span class="badge">Best epoch: {bundle["history"]["epoch"][bi]}</span>'
                        f'<span class="badge">Best val F1: {vf1[bi]:.4f}</span>'
                        f'<span class="badge">Total epochs: {bundle["history"]["epoch"][-1]}</span>'
                        f'</div>', unsafe_allow_html=True,
                    )


# ══════════════════════════════════════════════════════════════════════════════
# COMPARE MODE
# ══════════════════════════════════════════════════════════════════════════════

if compare_btn:
    source = get_source_text()
    if not source:
        st.error("⚠️  Paste or upload an email first.")
    elif len(models_loaded) < 2:
        st.warning("Load at least 2 models to compare.")
    else:
        st.markdown("---")
        st.markdown("""
        <div style="margin-bottom:16px">
          <div style="font-family:'Space Mono',monospace;font-size:1rem;color:#111827">⚖️ Model Comparison</div>
          <div style="font-size:.8rem;color:#6b7280;margin-top:2px">All loaded models on the same email</div>
        </div>""", unsafe_allow_html=True)

        results = []
        for mname, bundle in models_loaded.items():
            with st.spinner(f"Running {mname}…"):
                _, _, lp, pp = predict_with_tokens(bundle, source)
                results.append({
                    "name": mname, "model_type": bundle["model_type"],
                    "legit_prob": lp, "phish_prob": pp,
                    "label": "Phishing" if pp >= .5 else "Legitimate",
                    "is_tfidf_ffn": bundle.get("is_tfidf_ffn", False),
                })

        cmp_cols = st.columns(len(results))
        for col, r in zip(cmp_cols, results):
            with col:
                is_p     = r["label"] == "Phishing"
                brd      = "#e02d47" if is_p else "#059669"
                icon     = "🚨" if is_p else "✅"
                clr      = "#e02d47" if is_p else "#059669"
                tag_col2 = _TYPE_COLORS.get(r["model_type"], "#4b5563")
                sub      = r["model_type"].upper() + (" (TF-IDF)" if r["is_tfidf_ffn"] else "")
                st.markdown(
                    f'<div style="background:#ffffff;border:1.5px solid {brd};'
                    f'border-radius:12px;padding:16px;text-align:center;margin-bottom:10px">'
                    f'<div style="font-family:Space Mono,monospace;font-size:.78rem;'
                    f'color:#111827;margin-bottom:6px">{r["name"]}</div>'
                    f'<div style="font-size:.65rem;color:{tag_col2};margin-bottom:8px">{sub}</div>'
                    f'<div style="font-size:1.4rem">{icon}</div>'
                    f'<div style="font-family:Space Mono,monospace;font-size:1rem;'
                    f'font-weight:700;color:{clr};margin-top:4px">{r["label"]}</div>'
                    f'<div style="font-size:.75rem;color:#6b7280;margin-top:4px">'
                    f'P(phishing) = {r["phish_prob"]:.4f}</div></div>',
                    unsafe_allow_html=True,
                )

        st.plotly_chart(compare_bar(results), use_container_width=True,
                        config={"displayModeBar": False})

        phish_votes = sum(1 for r in results if r["label"] == "Phishing")
        legit_votes = len(results) - phish_votes
        consensus   = ("Phishing"  if phish_votes > legit_votes else
                       "Legitimate" if legit_votes > phish_votes else "Split")
        avg_phish   = sum(r["phish_prob"] for r in results) / len(results)
        con_col     = "#e02d47" if consensus == "Phishing" else "#059669" if consensus == "Legitimate" else "#d97706"
        st.markdown(
            f'<div class="card" style="text-align:center;padding:18px">'
            f'<div style="font-family:Space Mono,monospace;font-size:.7rem;color:#6b7280;'
            f'text-transform:uppercase;letter-spacing:2px;margin-bottom:8px">Ensemble Consensus</div>'
            f'<div style="font-family:Space Mono,monospace;font-size:1.5rem;'
            f'font-weight:700;color:{con_col}">{consensus}</div>'
            f'<div style="color:#6b7280;font-size:.8rem;margin-top:6px">'
            f'{phish_votes}/{len(results)} models voted Phishing · '
            f'avg P(phishing) = {avg_phish:.4f}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="slabel" style="margin-top:8px">Full Results Table</div>',
                    unsafe_allow_html=True)
        df_cmp = pd.DataFrame([{
            "Model":         r["name"],
            "Type":          r["model_type"].upper() + (" (TF-IDF)" if r["is_tfidf_ffn"] else ""),
            "Verdict":       r["label"],
            "P(Phishing)":   round(r["phish_prob"], 4),
            "P(Legitimate)": round(r["legit_prob"], 4),
        } for r in results])
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)