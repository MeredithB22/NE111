import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import (
    norm, gamma, weibull_min, expon, lognorm, beta,
    uniform, chi2, t, f, poisson, exponpow, pareto,
    genextreme, gumbel_r
)
import io
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Distribution Fitting App",
    page_icon="",
    layout="wide"
)


st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .success-message {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
    }
    .error-message {
        background-color: #FEE2E2;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #EF4444;
    }
    .metric-card {
        background-color: #cf61aa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="main-header">Statistical Distribution Fitting Tool</h1>', unsafe_allow_html=True)
st.markdown("""
This app allows you to fit various statistical distributions to your data. 
Upload a CSV file, enter data manually, or use sample data to get started.
""")


if 'data' not in st.session_state:
    st.session_state.data = None
if 'current_dist' not in st.session_state:
    st.session_state.current_dist = 'normal'
if 'fitted_params' not in st.session_state:
    st.session_state.fitted_params = None
if 'manual_params' not in st.session_state:
    st.session_state.manual_params = {}


DISTRIBUTIONS = {
    'Normal': {'func': norm, 'params': ['loc', 'scale'], 'param_names': ['Mean (μ)', 'Std Dev (σ)']},
    'Gamma': {'func': gamma, 'params': ['a', 'loc', 'scale'], 'param_names': ['Shape (α)', 'Location', 'Scale']},
    'Weibull': {'func': weibull_min, 'params': ['c', 'loc', 'scale'], 'param_names': ['Shape (k)', 'Location', 'Scale']},
    'Exponential': {'func': expon, 'params': ['loc', 'scale'], 'param_names': ['Location', 'Scale']},
    'Lognormal': {'func': lognorm, 'params': ['s', 'loc', 'scale'], 'param_names': ['Shape (σ)', 'Location', 'Scale']},
    'Beta': {'func': beta, 'params': ['a', 'b', 'loc', 'scale'], 'param_names': ['α', 'β', 'Location', 'Scale']},
    'Uniform': {'func': uniform, 'params': ['loc', 'scale'], 'param_names': ['Minimum', 'Maximum']},
    'Chi-squared': {'func': chi2, 'params': ['df', 'loc', 'scale'], 'param_names': ['Degrees of Freedom', 'Location', 'Scale']},
    "Student's t": {'func': t, 'params': ['df', 'loc', 'scale'], 'param_names': ['Degrees of Freedom', 'Location', 'Scale']},
    'F-distribution': {'func': f, 'params': ['dfn', 'dfd', 'loc', 'scale'], 'param_names': ['df numerator', 'df denominator', 'Location', 'Scale']},
    'Pareto': {'func': pareto, 'params': ['b', 'loc', 'scale'], 'param_names': ['Shape (b)', 'Location', 'Scale']},
    'Gumbel': {'func': gumbel_r, 'params': ['loc', 'scale'], 'param_names': ['Location', 'Scale']}
}

def generate_sample_data(dist_name: str, size: int = 1000) -> np.ndarray:
    """Generate sample data from different distributions."""
    np.random.seed(42)
    
    if dist_name == 'Normal':
        return np.random.normal(loc=5, scale=2, size=size)
    elif dist_name == 'Gamma':
        return np.random.gamma(shape=5, scale=1, size=size)
    elif dist_name == 'Weibull':
        return np.random.weibull(a=1.5, size=size) * 2
    elif dist_name == 'Exponential':
        return np.random.exponential(scale=2, size=size)
    elif dist_name == 'Lognormal':
        return np.random.lognormal(mean=0, sigma=0.5, size=size)
    else:
        return np.random.normal(loc=0, scale=1, size=size)

def fit_distribution(data: np.ndarray, dist_name: str) -> Tuple:
    """Fit a distribution to data and return parameters."""
    dist_info = DISTRIBUTIONS[dist_name]
    dist_func = dist_info['func']
    
    try:
        params = dist_func.fit(data)
        return params, None
    except Exception as e:
        return None, str(e)

def calculate_fit_metrics(data: np.ndarray, dist_name: str, params: tuple) -> Dict:
    """Calculate goodness-of-fit metrics."""
    dist_info = DISTRIBUTIONS[dist_name]
    dist_func = dist_info['func']
    

    dist = dist_func(*params)
    

    hist, bin_edges = np.histogram(data, bins='auto', density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    

    try:
        pdf_values = dist.pdf(bin_centers)
        

        mse = np.mean((hist - pdf_values) ** 2)
        mae = np.mean(np.abs(hist - pdf_values))
        max_error = np.max(np.abs(hist - pdf_values))
        

        epsilon = 1e-10
        kl_div = np.sum(hist * np.log((hist + epsilon) / (pdf_values + epsilon)))
        
        return {
            'MSE': mse,
            'MAE': mae,
            'Max Error': max_error,
            'KL Divergence': kl_div
        }
    except:
        return {}

def create_plot(data: np.ndarray, dist_name: str, params: tuple, manual_mode: bool = False):
    """Create visualization of data and fitted distribution."""
    dist_info = DISTRIBUTIONS[dist_name]
    dist_func = dist_info['func']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    

    n, bins, patches = ax.hist(data, bins='auto', density=True, alpha=0.7, 
                               color='tab:pink', edgecolor='black', label='Data Histogram')
    

    if params is not None:
        try:
            dist = dist_func(*params)
            xmin, xmax = data.min(), data.max()
            x_range = xmax - xmin
            x = np.linspace(xmin - 0.1*x_range, xmax + 0.1*x_range, 1000)
            y = dist.pdf(x)
            
            mode_label = 'Manual Fit' if manual_mode else 'Fitted Distribution'
            ax.plot(x, y, 'r-', linewidth=2, label=mode_label)
        except:
            pass
    
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{dist_name} Distribution Fit', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


tab1, tab2, tab3 = st.tabs(["Data Input", "Fitting & Visualization", "Results & Metrics"])

with tab1:
    st.markdown('<h2 class="section-header">Data Input Options</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Option 1: Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(df)} rows")
                

                if len(df.columns) > 0:
                    selected_column = st.selectbox("Select column for analysis:", df.columns)
                    st.session_state.data = df[selected_column].dropna().values
                    
                    with st.expander("Preview Data"):
                        st.dataframe(df.head(10), use_container_width=True)
                        
                    st.metric("Data Points", len(st.session_state.data))
                    st.metric("Mean", f"{st.session_state.data.mean():.4f}")
                    st.metric("Std Dev", f"{st.session_state.data.std():.4f}")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    with col2:
        st.markdown("### Option 2: Manual Data Entry")
        data_input = st.text_area(
            "Enter data (comma, space, or newline separated):",
            height=150,
            placeholder="e.g., 1.2, 2.3, 3.4, 4.5, 5.6\nor\n1.2 2.3 3.4 4.5 5.6"
        )
        
        if data_input:
            try:

                data_input = data_input.replace('\n', ' ').replace(',', ' ')
                numbers = []
                for part in data_input.split():
                    try:
                        numbers.append(float(part))
                    except:
                        continue
                
                if numbers:
                    st.session_state.data = np.array(numbers)
                    st.success(f"Successfully parsed {len(numbers)} data points")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Count", len(numbers))
                    with col_b:
                        st.metric("Mean", f"{np.mean(numbers):.4f}")
                    with col_c:
                        st.metric("Std Dev", f"{np.std(numbers):.4f}")
                else:
                    st.error("No valid numbers found in input")
            except Exception as e:
                st.error(f"Error parsing data: {e}")
        
        st.markdown("### Option 3: Use Sample Data")
        sample_type = st.selectbox(
            "Generate sample data from:",
            ['Normal', 'Gamma', 'Weibull', 'Exponential', 'Lognormal']
        )
        
        if st.button("Generate Sample Data", use_container_width=True):
            st.session_state.data = generate_sample_data(sample_type)
            st.success(f"Generated {len(st.session_state.data)} sample points from {sample_type} distribution")
            st.rerun()

with tab2:
    st.markdown('<h2 class="section-header">Distribution Fitting</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load or enter data in the 'Data Input' tab first.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Select Distribution")
            

            selected_dist = st.selectbox(
                "Choose distribution to fit:",
                list(DISTRIBUTIONS.keys()),
                index=0
            )
            
            st.session_state.current_dist = selected_dist
            

            if st.button("Auto-Fit Distribution", use_container_width=True):
                with st.spinner("Fitting distribution..."):
                    params, error = fit_distribution(st.session_state.data, selected_dist)
                    if error:
                        st.error(f"Fitting failed: {error}")
                    else:
                        st.session_state.fitted_params = params
                        st.success("Distribution fitted successfully!")
                        st.rerun()
            
            st.markdown("---")
            st.markdown("### Manual Fitting Mode")
            manual_mode = st.checkbox("Enable manual parameter adjustment")
            
            if manual_mode:
                dist_info = DISTRIBUTIONS[selected_dist]
                param_names = dist_info['param_names']
                

                if selected_dist not in st.session_state.manual_params:

                    if st.session_state.fitted_params:
                        st.session_state.manual_params[selected_dist] = list(st.session_state.fitted_params)
                    else:

                        defaults = {
                            'Normal': [0, 1],
                            'Gamma': [1, 0, 1],
                            'Weibull': [1, 0, 1],
                            'Exponential': [0, 1],
                            'Lognormal': [1, 0, 1],
                            'Beta': [1, 1, 0, 1],
                            'Uniform': [0, 1],
                            'Chi-squared': [1, 0, 1],
                            "Student's t": [1, 0, 1],
                            'F-distribution': [1, 1, 0, 1],
                            'Pareto': [1, 0, 1],
                            'Gumbel': [0, 1]
                        }
                        st.session_state.manual_params[selected_dist] = defaults.get(selected_dist, [0, 1])
                

                current_params = st.session_state.manual_params[selected_dist]
                new_params = []
                
                data_range = st.session_state.data.max() - st.session_state.data.min()
                data_min = st.session_state.data.min() - 0.1 * data_range
                data_max = st.session_state.data.max() + 0.1 * data_range
                
                for i, (param_name, current_value) in enumerate(zip(param_names, current_params)):

                    if 'Location' in param_name or 'Mean' in param_name:
                        min_val = data_min
                        max_val = data_max
                        step = 0.1
                    elif 'Scale' in param_name or 'Std Dev' in param_name or 'Maximum' in param_name:
                        min_val = 0.1
                        max_val = data_range * 2
                        step = 0.1
                    elif 'Shape' in param_name or 'α' in param_name or 'β' in param_name or 'k' in param_name:
                        min_val = 0.1
                        max_val = 10
                        step = 0.1
                    elif 'Degrees' in param_name or 'df' in param_name:
                        min_val = 1
                        max_val = 100
                        step = 1
                    else:
                        min_val = -10
                        max_val = 10
                        step = 0.1
                    
                    new_value = st.slider(
                        param_name,
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(current_value),
                        step=float(step),
                        key=f"slider_{selected_dist}_{i}"
                    )
                    new_params.append(new_value)
                
                st.session_state.manual_params[selected_dist] = new_params
        
        with col2:
            st.markdown("### Visualization")
            
           
            if manual_mode and selected_dist in st.session_state.manual_params:
                params_to_use = tuple(st.session_state.manual_params[selected_dist])
                manual_plot = True
            elif st.session_state.fitted_params is not None:
                params_to_use = st.session_state.fitted_params
                manual_plot = False
            else:
                params_to_use = None
                manual_plot = False
            

            fig = create_plot(st.session_state.data, selected_dist, params_to_use, manual_plot)
            st.pyplot(fig)
            
         
            with st.expander("Data Statistics"):
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Count", len(st.session_state.data))
                with col_b:
                    st.metric("Mean", f"{np.mean(st.session_state.data):.4f}")
                with col_c:
                    st.metric("Std Dev", f"{np.std(st.session_state.data):.4f}")
                with col_d:
                    st.metric("Range", f"{np.ptp(st.session_state.data):.4f}")

with tab3:
    st.markdown('<h2 class="section-header">Fitting Results</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("No data available. Please load data in the 'Data Input' tab.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Fitted Parameters")
            
            if st.session_state.fitted_params is not None:
                dist_info = DISTRIBUTIONS[st.session_state.current_dist]
                param_names = dist_info['param_names']
                
                for name, value in zip(param_names, st.session_state.fitted_params):
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{name}</strong><br>
                        <span style="font-size: 1.2rem; color: #e9d6ff;">{value:.6f}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No fitted parameters yet. Click 'Auto-Fit Distribution' in the previous tab.")
        
        with col2:
            st.markdown("### Fit Metrics")
            
            if st.session_state.fitted_params is not None:
                metrics = calculate_fit_metrics(
                    st.session_state.data, 
                    st.session_state.current_dist, 
                    st.session_state.fitted_params
                )
                
                if metrics:
                    for metric_name, value in metrics.items():
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong>{metric_name}</strong><br>
                            <span style="font-size: 1.2rem; color: #e9d6ff;">{value:.6f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("Could not calculate metrics for this distribution.")
            else:
                st.info("Fit a distribution to see metrics.")
        

        st.markdown("---")
        st.markdown("### Export Results")
        
        if st.session_state.fitted_params is not None:
            col_a, col_b = st.columns(2)
            
            with col_a:

                if st.button("Export Parameters as CSV", use_container_width=True):
                    dist_info = DISTRIBUTIONS[st.session_state.current_dist]
                    results_df = pd.DataFrame({
                        'Parameter': dist_info['param_names'],
                        'Value': st.session_state.fitted_params
                    })
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{st.session_state.current_dist}_parameters.csv",
                        mime="text/csv"
                    )
            
            with col_b:
        
                if st.button("Export Plot", use_container_width=True):
                    fig = create_plot(
                        st.session_state.data, 
                        st.session_state.current_dist, 
                        st.session_state.fitted_params
                    )
                    
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=150)
                    buf.seek(0)
                    
                    st.download_button(
                        label="Download Plot",
                        data=buf,
                        file_name=f"{st.session_state.current_dist}_fit.png",
                        mime="image/png"
                    )

