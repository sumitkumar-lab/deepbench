import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import torch
from src.backend import ModelResearcher, ModelManager
from src.benchmarks import BenchmarkSuite

# --- Styling & Config ---
st.set_page_config(page_title="DeepBench: AI Researcher Workbench", layout="wide", page_icon="🧪")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    h1, h2, h3 { color: #00d4ff; }
    .metric-card {
        background-color: #262730; border: 1px solid #41424C;
        border-radius: 8px; padding: 15px; margin-bottom: 10px;
        text-align: center;
    }
    .metric-val { font-size: 24px; font-weight: bold; }
    .stButton>button { width: 100%; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if 'manager' not in st.session_state:
    st.session_state['manager'] = ModelManager(device="cuda" if torch.cuda.is_available() else "cpu")

# --- Sidebar ---
with st.sidebar:
    st.title("🧪 DeepBench")
    st.markdown("### Researcher Control Panel")
    task = st.selectbox("Domain", ["Language", "Vision"])
    arch = st.radio("Architecture", ["All", "Transformer", "RNN/RWKV"])
    st.markdown("---")
    st.caption("v2.2 Stable | Direct Load Mode")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🔍 Model Discovery", "⚔️ Battle Arena", "💬 Playground"])

# ================= TAB 1: DISCOVERY =================
with tab1:
    researcher = ModelResearcher()
    col_search, col_res = st.columns([1, 4])
    
    with col_search:
        if st.button("Fetch Models", use_container_width=True):
            st.session_state['models'] = researcher.search_models(task_domain=task, architecture_type=arch)

    with col_res:
        if 'models' in st.session_state:
            st.dataframe(
                st.session_state['models'],
                column_config={"downloads": st.column_config.ProgressColumn("Downloads", format="%d", min_value=0, max_value=1000000)},
                use_container_width=True
            )

# ================= TAB 2: BATTLE ARENA =================
with tab2:
    if 'models' in st.session_state:
        all_ids = st.session_state['models']['model_id'].tolist()
        select_options = ["None"] + all_ids
        
        c1, c2 = st.columns(2)
        with c1:
            model_a = st.selectbox("Champion (Model A)", select_options, index=1 if len(all_ids)>0 else 0)
        with c2:
            model_b = st.selectbox("Challenger (Model B)", select_options, index=0)

        bench_opts = ["Perplexity", "MMLU", "GSM8K", "ARC-C", "ARC-E", "HellaSwag", "PIQA"]
        selected_bench = st.multiselect("Benchmarks", bench_opts, default=["Perplexity", "MMLU"])
        
        if st.button("⚔️ Run Comparison"):
            col_a, col_mid, col_b = st.columns([1, 0.1, 1])
            results_a, results_b = {}, {}

            # --- PROCESS MODEL A ---
            with col_a:
                if model_a != "None":
                    # st.write(f"**Loading {model_a}...**")
                    st.subheader(f"🔵 {model_a}")
                    with st.spinner(f"Loading {model_a}..."):
                        succ, msg = st.session_state['manager'].load_model(model_a)
                    
                    if succ:
                        mod, tok = st.session_state['manager'].get_components(model_a)
                        suite = BenchmarkSuite(mod, tok, model_id=model_a)
                        
                        for b in selected_bench:
                            res = suite.run_benchmark(b, simulation_mode=True)
                            results_a[b] = res
                            st.markdown(f"""
                            <div class='metric-card'>
                                <div style='color:#aaa;'>{b}</div>
                                <div class='metric-val'>{res['score']:.2f}</div>
                                <div>{res['rating']}</div>
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.error(f"Failed: {msg}")

            # --- PROCESS MODEL B ---
            with col_b:
                if model_b != "None":
                    st.subheader(f"🔴 {model_b}")
                    with st.spinner(f"Loading {model_b}..."):
                        succ, msg = st.session_state['manager'].load_model(model_b)
                    
                    if succ:
                        mod, tok = st.session_state['manager'].get_components(model_b)
                        suite = BenchmarkSuite(mod, tok, model_id=model_b)
                        
                        for b in selected_bench:
                            res = suite.run_benchmark(b, simulation_mode=True)
                            results_b[b] = res
                            st.markdown(f"""
                            <div class='metric-card'>
                                <div style='color:#aaa;'>{b}</div>
                                <div class='metric-val'>{res['score']:.2f}</div>
                                <div>{res['rating']}</div>
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.error(f"Failed: {msg}")

            # --- RADAR CHART ---
            if results_a and results_b and model_a != "None" and model_b != "None":
                st.markdown("### 🕸️ Comparison Map")
                categories = list(results_a.keys())
                vals_a = [r['score'] if r['unit'] == "%" else (100-r['score']) for r in results_a.values()]
                vals_b = [r['score'] if r['unit'] == "%" else (100-r['score']) for r in results_b.values()]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=vals_a, theta=categories, fill='toself', name=model_a, line_color="#00d4ff"))
                fig.add_trace(go.Scatterpolar(r=vals_b, theta=categories, fill='toself', name=model_b, line_color="#ff0055"))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Go to Discovery tab first.")

# ================= TAB 3: PLAYGROUND =================
with tab3:
    st.subheader("💬 Generation Playground")
    
    if 'models' in st.session_state:
        all_ids = st.session_state['models']['model_id'].tolist()
        select_options_play = ["None"] + all_ids
        
        pc1, pc2 = st.columns(2)
        with pc1:
            pm_a = st.selectbox("Generator A", select_options_play, index=1 if len(all_ids)>0 else 0, key="pm_a")
        with pc2:
            pm_b = st.selectbox("Generator B", select_options_play, index=0, key="pm_b")

        user_prompt = st.text_area("Prompt", value="Explain quantum computing like I'm 5.")

        if st.button("Generate Text"):
            c1, c2 = st.columns(2)
            
            # --- GEN A ---
            with c1:
                if pm_a != "None":
                    with st.spinner(f"Loading & Running {pm_a}..."):
                        succ, msg = st.session_state['manager'].load_model(pm_a)
                        if succ:
                            out = st.session_state['manager'].generate_text(pm_a, user_prompt)
                            st.info(out)
                        else:
                            st.error(msg)
            
            # --- GEN B ---
            with c2:
                if pm_b != "None":
                    with st.spinner(f"Loading & Running {pm_b}..."):
                        succ, msg = st.session_state['manager'].load_model(pm_b)
                        if succ:
                            out = st.session_state['manager'].generate_text(pm_b, user_prompt)
                            st.success(out)
                        else:
                            st.error(msg)
    else:
        st.warning("Please fetch models in Tab 1 first.")