import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
from statsmodels.stats.proportion import proportions_ztest

# --------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# --------------------------------------------------------
st.set_page_config(page_title="A/B Test Analytics Dashboard", layout="wide")

st.title("📊 Advanced A/B Testing Significance Dashboard")
st.markdown("""
This application evaluates the statistical significance of an e-commerce checkout flow experiment. 
It tests whether **Variant B (New Checkout)** yields a genuinely higher conversion rate than **Variant A (Control)**.
""")

# --------------------------------------------------------
# 2. SIDEBAR CONTROL PANEL
# --------------------------------------------------------
st.sidebar.header("🕹️ Experiment Controls")

# Parameter inputs
alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.10, 0.05, step=0.01, 
                          help="The probability of rejecting the null hypothesis when it is actually true (Type I error).")

test_type = st.sidebar.selectbox("Hypothesis Type", ["One-sided (B > A)", "Two-sided (B != A)"], index=0)
alternative = 'larger' if test_type == "One-sided (B > A)" else 'two-sided'

# Button to generate synthetic data if user wants to refresh
generate_data = st.sidebar.button("🔄 Regenerate Raw Experiment Data")

# --------------------------------------------------------
# 3. DATA SIMULATION (Backend Engine)
# --------------------------------------------------------
@st.cache_data
def load_or_create_data(trigger):
    # Simulating a realistic raw user-interaction log
    np.random.seed(42 if not trigger else np.random.randint(1, 1000))
    
    # Control Group Data (N=10,000, true CR = 12%)
    n_control = 10000
    control_conv = np.random.binomial(1, 0.12, n_control)
    df_control = pd.DataFrame({'user_id': range(1, n_control + 1), 'group': 'control', 'converted': control_conv})
    
    # Treatment Group Data (N=10,050, true CR = 13.1%)
    n_treatment = 10050
    treatment_conv = np.random.binomial(1, 0.131, n_treatment)
    df_treatment = pd.DataFrame({'user_id': range(n_control + 1, n_control + n_treatment + 1), 'group': 'treatment', 'converted': treatment_conv})
    
    return pd.concat([df_control, df_treatment], ignore_index=True)

df = load_or_create_data(generate_data)

# Aggregate stats for analysis
summary = df.groupby('group').agg(
    total_users=('user_id', 'count'),
    conversions=('converted', 'sum')
).reset_index()

n_A = summary.loc[summary['group'] == 'control', 'total_users'].values[0]
c_A = summary.loc[summary['group'] == 'control', 'conversions'].values[0]
n_B = summary.loc[summary['group'] == 'treatment', 'total_users'].values[0]
c_B = summary.loc[summary['group'] == 'treatment', 'conversions'].values[0]

cr_A = c_A / n_A
cr_B = c_B / n_B
relative_lift = (cr_B - cr_A) / cr_A

# --------------------------------------------------------
# 4. STATISTICAL CALCULATIONS
# --------------------------------------------------------
# Two-Proportion Z-Test
counts = np.array([c_B, c_A])
nobs = np.array([n_B, n_A])
z_stat, p_value = proportions_ztest(counts, nobs, alternative=alternative)

# Confidence Interval for difference
pooled_prob = (c_A + c_B) / (n_A + n_B)
se_diff = np.sqrt(pooled_prob * (1 - pooled_prob) * (1/n_A + 1/n_B))
z_critical = norm.ppf(1 - alpha/2) if alternative == 'two-sided' else norm.ppf(1 - alpha)
diff = cr_B - cr_A
ci_lower = diff - z_critical * se_diff
ci_upper = diff + z_critical * se_diff

# --------------------------------------------------------
# 5. DASHBOARD LAYOUT & VISUALIZATION
# --------------------------------------------------------
# High-Level Metric Cards
col1, col2, col3 = st.columns(3)
col1.metric("Control Conversion Rate (A)", f"{cr_A:.2%}", help=f"{c_A:,} / {n_A:,} users")
col2.metric("Treatment Conversion Rate (B)", f"{cr_B:.2%}", help=f"{c_B:,} / {n_B:,} users")
col3.metric("Observed Relative Lift", f"+{relative_lift:.2%}", delta=f"{cr_B-cr_A:.2%}")

st.markdown("---")

# Visualizing the Statistical Distribution
st.subheader("📈 Statistical Distribution Plot")

# Generate normal curves based on pooled standard error
x = np.linspace(-4 * se_diff, 4 * se_diff + diff, 1000)
y_null = norm.pdf(x, 0, se_diff)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y_null, mode='lines', name='Null Hypothesis (No Difference)', line=dict(color='gray', dash='dash')))

# Line for observed difference
fig.add_shape(type="line", x0=diff, y0=0, x1=diff, y1=max(y_null), line=dict(color="blue", width=3, dash="dot"))
fig.add_trace(go.Scatter(x=[diff], y=[max(y_null)/2], mode='markers+text', text=["Observed Diff"], textposition="top right", marker=dict(color='blue', size=10), showlegend=False))

fig.update_layout(
    xaxis_title="Difference in Conversion Rates (B - A)",
    yaxis_title="Probability Density",
    margin=dict(l=20, r=20, t=20, b=20),
    height=350
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------
# 6. EXPERIMENT VERDICT WINDOW
# --------------------------------------------------------
st.subheader("🎯 Statistical Verdict")

is_significant = p_value <= alpha

if is_significant:
    st.success(f"""
    ### 🟢 Statistically Significant Result!
    * **p-value:** `{p_value:.4f}` (which is $\le$ `{alpha}`)
    * **Confidence Interval (Difference):** `[{ci_lower:.4%}, {ci_upper:.4%}]`
    
    **Business Decision:** Reject the Null Hypothesis ($H_0$). There is less than a {alpha*100:.0f}% probability that this calculated uplift was caused by random variance. **Deploy the new checkout variant to 100% of traffic.**
    """)
else:
    st.error(f"""
    ### 🔴 Result Not Statistically Significant
    * **p-value:** `{p_value:.4f}` (which is > `{alpha}`)
    * **Confidence Interval (Difference):** `[{ci_lower:.4%}, {ci_upper:.4%}]`
    
    **Business Decision:** Fail to reject the Null Hypothesis ($H_0$). The detected conversion lift cannot be distinguished from background noise. **Maintain Variant A or extend test runtime to collect more data.**
    """)

# Show raw sample logs for presentation validity
if st.checkbox("🔍 View Sample Raw Conversion Log Data"):
    st.write(df.sample(10).reset_index(drop=True))
