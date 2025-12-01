# ui/styles.py - CSS Styling

import streamlit as st


def apply_custom_styles():
    """Apply custom CSS styles to the application"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


CUSTOM_CSS = """
<style>
/* Remove Streamlit header completely */
header[data-testid="stHeader"] {
    display: none;
}

/* Aggressively remove all top and bottom padding */
.main .block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    max-width: 100%;
}

.main {
    padding: 0 !important;
}

section[data-testid="stAppViewContainer"] > .main {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

.element-container {
    margin: 0;
}

.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

footer {visibility: hidden;}

/* Global styling */
.main {
    background-color: #ffffff;
}

/* Header styling - Minimalist & Compact */
.main-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #0f172a;
    text-align: center;
    margin-bottom: 0.15rem;
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    letter-spacing: -0.02em;
}

.sub-header {
    font-size: 0.875rem;
    color: #64748b;
    text-align: center;
    margin-bottom: 0.75rem;
    margin-top: 0;
    font-weight: 400;
}

/* Enhanced section titles - Minimalist */
.config-section-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #1e293b;
    margin-top: 0.75rem;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 0.25rem;
    padding-left: 0;
}

.config-section-title::before {
    display: none;
}

.field-description {
    display: none;
}

/* Minimalist button styling */
.stButton>button {
    width: 100%;
    background: #2563eb;
    color: white;
    border-radius: 6px;
    padding: 0.5rem 1rem;
    font-weight: 500;
    border: none;
    transition: all 0.2s ease;
    font-size: 0.875rem;
    box-shadow: none;
}

.stButton>button:hover {
    background: #1d4ed8;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2);
}

.stButton>button:active {
    transform: scale(0.98);
}

/* Secondary button */
.stButton>button:not([kind="primary"]) {
    background: #f1f5f9;
    color: #1e293b;
    border: 1px solid #e2e8f0;
}

.stButton>button:not([kind="primary"]):hover {
    background: #e2e8f0;
    border-color: #cbd5e1;
}

/* Minimalist expander styling */
div[data-testid="stExpander"] {
    background: #ffffff;
    border-radius: 6px;
    border: 1px solid #e2e8f0;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s ease;
}

div[data-testid="stExpander"]:hover {
    border-color: #cbd5e1;
}

div[data-testid="stExpander"] summary {
    padding: 0.5rem 0.75rem;
    font-size: 0.875rem;
    font-weight: 600;
    color: #1e293b;
    background-color: transparent;
}

div[data-testid="stExpander"][aria-expanded="true"] {
    border-color: #3b82f6;
}

div[data-testid="stExpander"] > div:last-child {
    padding: 0.75rem;
    background-color: #ffffff;
}

/* Minimalist input fields */
.stNumberInput label, .stSelectbox label {
    font-size: 0.8rem;
    margin-bottom: 0.25rem;
    font-weight: 500;
    color: #475569;
}

.stNumberInput, .stSelectbox {
    margin-bottom: 0.5rem;
}

.stNumberInput > div > div > input {
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 0.4rem 0.6rem;
    font-size: 0.875rem;
    background-color: #ffffff;
    transition: border-color 0.2s ease;
}

.stNumberInput > div > div > input:hover {
    border-color: #cbd5e1;
}

.stNumberInput > div > div > input:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
    background-color: #ffffff;
}

.stSelectbox > div > div > div {
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    background-color: #ffffff;
    transition: border-color 0.2s ease;
}

.stSelectbox > div > div > div:hover {
    border-color: #cbd5e1;
}

.stNumberInput button {
    border-radius: 3px;
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    color: #64748b;
    transition: all 0.2s ease;
}

.stNumberInput button:hover {
    background-color: #f1f5f9;
    border-color: #cbd5e1;
    color: #475569;
}

/* Metric cards - minimalist */
div[data-testid="stMetricValue"] {
    font-size: 1.25rem;
    font-weight: 600;
    color: #0f172a;
}

div[data-testid="stMetricLabel"] {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Info/Success/Error boxes - minimalist */
.stAlert {
    border-radius: 4px;
    border-left-width: 3px;
    padding: 0.5rem 0.75rem;
    font-size: 0.875rem;
}

/* Minimalist tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background-color: transparent;
    padding: 0;
    border-bottom: 1px solid #e2e8f0;
    margin-top: 0;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 0;
    padding: 0.4rem 0.8rem;
    font-weight: 500;
    color: #64748b;
    font-size: 0.875rem;
    border-bottom: 2px solid transparent;
}

.stTabs [aria-selected="true"] {
    background-color: transparent;
    color: #2563eb;
    border-bottom: 2px solid #2563eb;
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 0.75rem;
}

/* Dataframe - minimalist */
.stDataFrame {
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #f8fafc;
    border-right: 1px solid #e2e8f0;
}

/* Footer - minimalist */
.footer {
    text-align: center;
    color: #94a3b8;
    padding: 0.5rem 0;
    margin-top: 1rem;
    border-top: 1px solid #e2e8f0;
    font-size: 0.75rem;
}

/* Spinner styling */
.stSpinner > div {
    border-top-color: #2563eb;
}

/* Charts container - minimalist */
.js-plotly-plot {
    border-radius: 4px;
}

/* Reduce column gaps */
.row-widget.stHorizontal {
    gap: 0.5rem;
}

/* Compact column spacing */
div[data-testid="column"] {
    padding: 0 0.25rem;
}
</style>
"""
