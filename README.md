# 🎣 PhishNet — Multi-Model Phishing Email Detector

A Streamlit-based phishing email detection dashboard that supports multiple deep learning architectures simultaneously. Upload trained models from a team and compare their predictions side-by-side on the same email.

![Light theme dashboard](https://img.shields.io/badge/UI-Light_Theme-2563eb) ![Multi-model](https://img.shields.io/badge/Models-BiLSTM%20%7C%20RNN%20%7C%20CNN%20%7C%20FFN%20%7C%20sklearn-059669) ![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-ff4b4b)

## Features

- **Multi-model support** — load trained models for BiLSTM, RNN/GRU/LSTM, CNN, FFN, and traditional sklearn pipelines (TF-IDF + classifier)
- **Auto-detection** — architecture is inferred directly from the state dict, so no manual config tuning is needed for most teammate models
- **Side-by-side comparison** — run all loaded models on the same email and see a probability bar chart, ensemble vote, and full results table
- **Token-level explainability** — leave-one-out heatmap with percentile-scaled colors so meaningful tokens stand out even on highly confident predictions
- **Risk radar** — multi-dimensional breakdown across model confidence, keywords, URLs, writing style, and HTML presence
- **URL inspector** — flags IP-based URLs, suspiciously-pathed links, and unusually long URLs
- **Email statistics** — token count, CAPS ratio, exclamation marks, average word length, and most frequent words
- **Training curves** — visualize loss, accuracy, F1, precision, and recall across epochs (when `history.pkl` is provided)
- **Scan history** — last 6 emails analyzed with their verdicts shown in the sidebar

## Getting Started

### Run locally

```bash
git clone https://github.com/marcusreese1/phishnet.git
cd phishnet
pip install -r requirements.txt
streamlit run phishnet.py
```

### Deploy to Streamlit Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub and select this repo
4. Set the main file to `phishnet.py`
5. Click Deploy

## Model Folder Format

Each model should be packaged as a ZIP containing:

```
your_model_name/
├── weights.pt                 # PyTorch state dict (or .pth)
├── config.pkl                 # hyperparameters dict
├── vocab.pkl                  # word → index dictionary
├── label_map.pkl              # {0: "Legitimate", 1: "Phishing"} or reversed
├── tfidf_vectorizer.pkl       # only for sklearn or TF-IDF FFN models
├── model_info.json            # optional: {"type": "bilstm"} override
└── history.pkl                # optional: training history for curves tab
```

The app auto-detects the architecture from the state dict shape, so different teammates can use slightly different model code without breaking the loader.

## Supported Architectures

| Type | Detection signal | Example use |
|---|---|---|
| **BiLSTM** | Bidirectional LSTM weights | `bilstm_model_final.pt` |
| **RNN / GRU / LSTM** | Unidirectional `rnn`, `lstm`, or `gru` attribute | Plain RNN classifier |
| **CNN** (multi-kernel) | `convs.0.weight`, `convs.1.weight` etc. | Standard text CNN |
| **CNN** (single-layer) | `conv1.weight` | Conv1d-only architectures |
| **FFN** (embedding-based) | Embedding + MLP | Sequence-aware FFN |
| **FFN** (TF-IDF) | Plain `nn.Sequential` over TF-IDF vectors | TF-IDF + MLP |
| **sklearn** | `tfidf_vectorizer.pkl` + `model.pkl` | LogisticRegression, SVM, RandomForest |

## Tech Stack

- **PyTorch** for deep learning models
- **scikit-learn** for traditional ML and TF-IDF vectorizer support
- **Streamlit** for the web interface
- **Plotly** for interactive charts (gauge, radar, comparison bars)
- **Pandas** for tabular results

## License

MIT
