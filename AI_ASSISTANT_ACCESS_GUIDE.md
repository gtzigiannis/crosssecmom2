# AI Assistant Access Guide

## Repository Information

**Repository**: `gtzigiannis/crosssecmom2`  
**URL**: https://github.com/gtzigiannis/crosssecmom2  
**Status**: Public ✅  
**Primary Branch**: `main`

---

## Quick Access Links

### Direct Repository Access
```
https://github.com/gtzigiannis/crosssecmom2
```

### Raw File URLs (Best for AI Assistants)

Use these URLs to access files directly without HTML rendering:

**Documentation:**
```
https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/README.md
https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/LEVERAGE_FIXES_SUMMARY.md
https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/CDO_PERFORMANCE_INVESTIGATION.md
https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/AI_ASSISTANT_ACCESS_GUIDE.md
```

**Main Entry Point:**
```
https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/main.py
```

**Configuration:**
```
https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/config.py
```

**Core Strategy Files:**
```
https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/alpha_models.py
https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/portfolio_construction.py
https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/walk_forward_engine.py
https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/feature_engineering.py
https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/universe_metadata.py
https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/regime.py
```

**Verification Scripts:**
```
https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/verify_accounting.py
https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/setup.py
```

### How to Access ANY File in the Repository

**URL Pattern:**
```
https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/<file_path>
```

**Examples:**
- Root file: `https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/config.py`
- Subdirectory: `https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/docs/architecture.md`
- Python package: `https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/src/models/momentum.py`

**For Branch-Specific Access:**
```
https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/<branch_name>/<file_path>
```

**Complete File Listing via API:**
```bash
# Get repository tree (all files)
curl https://api.github.com/repos/gtzigiannis/crosssecmom2/git/trees/main?recursive=1

# Parse with jq to list all files
curl -s https://api.github.com/repos/gtzigiannis/crosssecmom2/git/trees/main?recursive=1 | jq -r '.tree[].path'
```

---

## For AI Assistants

### Repository Structure

```
crosssecmom2/
├── config.py                         # Configuration parameters
├── regime.py                         # Market regime detection
├── universe_metadata.py              # ETF universe management
├── feature_engineering.py            # Feature generation
├── alpha_models.py                   # Model training & scoring
├── portfolio_construction.py         # Portfolio optimization
├── walk_forward_engine.py            # Backtesting engine
├── main.py                           # CLI entry point
├── verify_accounting.py              # Accounting verification script
├── README.md                         # Complete documentation
├── LEVERAGE_FIXES_SUMMARY.md         # Technical fix documentation
├── CDO_PERFORMANCE_INVESTIGATION.md  # Performance analysis
└── AI_ASSISTANT_ACCESS_GUIDE.md      # This file
```

### Key Information

- **Strategy Type**: Cross-sectional momentum on 116 ETFs
- **Language**: Python 3.8+
- **Main Dependencies**: pandas, numpy, scikit-learn, yfinance, cvxpy (optional)
- **Data Location**: `D:\REPOSITORY\Data\crosssecmom2\`
- **Documentation**: Comprehensive README with mathematical details

### Accessing File Contents

**Method 1: Direct Raw URLs** (Recommended)
```python
# Pattern: https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filepath}
url = "https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/config.py"

# Access in code:
import requests
content = requests.get(url).text
```

**Method 2: GitHub API**
```bash
# Get file content with metadata
curl -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/gtzigiannis/crosssecmom2/contents/config.py

# Returns base64-encoded content + metadata
```

**Method 3: List All Files Programmatically**
```python
import requests

# Get complete file tree
response = requests.get(
    "https://api.github.com/repos/gtzigiannis/crosssecmom2/git/trees/main?recursive=1"
)
tree = response.json()

# Extract all file paths
files = [item['path'] for item in tree['tree'] if item['type'] == 'blob']

# Build raw URLs for each file
for filepath in files:
    raw_url = f"https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/{filepath}"
    print(f"{filepath}: {raw_url}")
```

**Method 4: Clone Locally** (For extensive access)
```bash
git clone https://github.com/gtzigiannis/crosssecmom2.git
cd crosssecmom2
```

---

## Troubleshooting

### If AI assistants still can't access the repository:

1. **Use raw.githubusercontent.com URLs** instead of github.com URLs
2. **Copy-paste file contents** directly into the conversation
3. **Verify repository is public**: Check for 🔒 icon on GitHub
4. **Check GitHub status**: Visit https://www.githubstatus.com/
5. **Wait for indexing**: Recently public repos may take time to index

### Contact Information

- **Repository Owner**: gtzigiannis
- **Issues**: https://github.com/gtzigiannis/crosssecmom2/issues

---

## Quick Start Command Template

For AI assistants helping with this repository, use this template:

```
I need help with the crosssecmom2 repository (cross-sectional momentum strategy on 116 ETFs).

Repository: https://github.com/gtzigiannis/crosssecmom2

Key files to review:
1. README: https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/README.md
2. Config: https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/config.py
3. [Specific file]: https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/[filename]

Context:
- Strategy: Cross-sectional momentum on 116 ETFs (2011-2025)
- Status: FIX 0-5 complete (accounting & leverage verified correct)
- Performance: -69% return, 39% win rate (MODEL issue, not accounting bug)
- Investigation: See CDO_PERFORMANCE_INVESTIGATION.md for analysis framework

My question: [Your question here]
```

### Quick Access to Key Documentation

**For Understanding the Strategy:**
```
https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/README.md
```

**For Understanding Recent Fixes:**
```
https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/LEVERAGE_FIXES_SUMMARY.md
```

**For Understanding Performance Issues:**
```
https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/CDO_PERFORMANCE_INVESTIGATION.md
```

**For Accessing Code:**
```bash
# Core modules (raw URLs)
CONFIG="https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/config.py"
FEATURES="https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/feature_engineering.py"
PORTFOLIO="https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/portfolio_construction.py"
BACKTEST="https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/walk_forward_engine.py"
MODELS="https://raw.githubusercontent.com/gtzigiannis/crosssecmom2/main/alpha_models.py"
```

---

## API Access (Advanced)

If programmatic access is needed:

### GitHub API
```bash
curl -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/gtzigiannis/crosssecmom2
```

### Clone Repository
```bash
git clone https://github.com/gtzigiannis/crosssecmom2.git
```

---

**Last Updated**: November 22, 2025
