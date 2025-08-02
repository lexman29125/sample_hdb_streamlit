# HDB Resale Price & Family Housing Explorer 🏠

A comprehensive data analysis and visualization project for Singapore HDB (Housing & Development Board) resale prices, with a special focus on family-friendly housing considerations for families with young children.

## 📖 My Workflow in creating this streamlit app

This repository demonstrates a modern AI-assisted development workflow using **VS Code with GitHub Copilot (Agent Mode, Claude Sonnet 4)** to rapidly prototype and build a comprehensive Streamlit application. The quality of the generated code is significantly influenced by the LLM capabilities and prompt engineering.

### 🔄 Development Process

**1. Requirement Gathering (`notes.md`)**

- Write a brief problem statement and research questions
- Use AI to refine the requirements

**2. Code Generation (`notes.md` + `Data_Dictionary.md`)**

- This is my brief prompt to kick off the code generation:
  > "Based on notes.md and Data_Dictionary.md, generate a streamlit app that allows the user to explore the dataset."

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.10+ (recommended)
- pip or conda package manager

### Installation Steps

1. **Clone or download this repository**

   ```bash
   git clone https://github.com/zey-2/sample_hdb.git
   cd sample_hdb
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   # Using conda
   conda create -n hdb_analysis python=3.10
   conda activate hdb_analysis
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

### Required Dependencies

- `streamlit>=1.28.0` - Web application framework
- `pandas>=1.5.0` - Data manipulation and analysis
- `numpy>=1.21.0` - Numerical computing
- `plotly>=5.10.0` - Interactive visualizations
- `seaborn>=0.11.0` - Statistical data visualization
- `matplotlib>=3.5.0` - Plotting library
- `openpyxl>=3.0.0` - Excel file support

## 🚀 Quick Start

### Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
sample_hdb/
├── README.md                 # This file
├── Data_Dictionary.md        # Complete feature descriptions (Renamed from README.md)
├── instruction.md           # EDA guidelines and methodology (Not used to vibe-code this app)
├── notes.md                 # Problem statement and research questions
├── requirements.txt         # Python dependencies
├── run_app.bat             # Windows batch file to run Streamlit
├── STREAMLIT_README.md     # Streamlit app specific documentation
├── streamlit_app.py        # Main Streamlit application
├── streamlit_app_clean.py  # Clean version of Streamlit app
└── datasets/
    ├── train.csv           # Training dataset for analysis
    └── test.csv            # Test dataset
```
