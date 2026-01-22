import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from src.backend import ModelResearcher, ModelEvaluator

# --- Page Configuration & Custom Styling ---
st.set_page_config(page_title="Neural Architect Research Lab", layout="wide", page_icon="🧠")

# Artistic Custom CSS
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e0e0e0;
    }
    /* Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        background: -webkit-linear-gradient(45deg, #00dbde, #fc00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    /* Cards for Models */
    div.stDataFrame {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
    }
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #000;
        border: none;
        border-radius: 20px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Controls ---
with st.sidebar:
    st.header("🔍 Research Filters")
    
    arch_type = st.radio(
        "Architecture Paradigm",
        ["Attention (Transformer)", "Recurrent (RNN/RWKV/Mamba)"]
    )
    
    sort_option = st.selectbox(
        "Sort Models By",
        ["downloads", "likes", "created_at"]
    )
    
    device_opt = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"⚡ Hardware Detected: **{device_opt.upper()}**")
    
    st.markdown("---")
    st.caption("Neural Architect Lab v1.0")

# --- Main Interface ---
st.title("🧠 Neural Architect: Model Insight Lab")
st.markdown("### Compare Architectures & Performance Metrics in Real-Time")

# 1. Automatic Search Section
st.subheader("1. Model Discovery Engine")

researcher = ModelResearcher()

if st.button("🚀 Search HuggingFace Hub"):
    with st.spinner("Scanning the neural web..."):
        df_models = researcher.search_models(architecture_type=arch_type, sort_by=sort_option)
        st.session_state['models'] = df_models
        st.success(f"Found {len(df_models)} models matching criteria.")

if 'models' in st.session_state:
    # Display artistic dataframe
    st.dataframe(
        st.session_state['models'], 
        column_config={
            "model_id": "Model Identity",
            "likes": st.column_config.NumberColumn("Likes", format="%d ❤️"),
            "downloads": st.column_config.ProgressColumn("Popularity", min_value=0, max_value=1000000),
        },
        use_container_width=True
    )
    
    # 2. Model Selection
    st.markdown("---")
    st.subheader("2. Select Subject for Evaluation")
    
    model_list = st.session_state['models']['model_id'].tolist()
    selected_model = st.selectbox("Choose a model to load:", model_list)

    # 3. Benchmark Configuration
    st.subheader("3. Benchmark Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        metric_choice = st.multiselect("Select Metrics", ["Perplexity (WikiText)", "Accuracy (Simulated)"], default=["Perplexity (WikiText)"])
    with col2:
        dataset_choice = st.text_input("HuggingFace Dataset (Default: wikitext)", value="wikitext")

    # 4. Execution & Plotting
    if st.button("🧪 Run Experiments"):
        st.markdown("---")
        st.subheader("4. Research Results")
        
        evaluator = ModelEvaluator(selected_model, device=device_opt)
        
        # Loading Phase
        with st.status("Initializing Neural Weights...", expanded=True) as status:
            st.write("Downloading model config...")
            success, msg = evaluator.load_model()
            if not success:
                st.error(f"Failed to load: {msg}")
                status.update(label="Load Failed", state="error")
            else:
                st.write("Model loaded to memory.")
                
                results = {}
                
                # Perplexity Check
                if "Perplexity (WikiText)" in metric_choice:
                    st.write("Calculating Perplexity (this may take a moment)...")
                    ppl = evaluator.calculate_perplexity()
                    if ppl:
                        results["Perplexity"] = ppl
                        st.metric("WikiText Perplexity", f"{ppl:.2f}", delta_color="inverse")
                    else:
                        st.error("Perplexity calculation failed.")
                
                # Simulated Accuracy (Placeholder for complex eval)
                if "Accuracy (Simulated)" in metric_choice:
                    import random
                    # In a real app, you would run MMLU here. 
                    # Simulating for demonstration speed as MMLU takes hours on CPU.
                    acc = random.uniform(60.0, 85.0) 
                    results["MMLU Accuracy"] = acc
                    st.metric("MMLU Accuracy (Simulated)", f"{acc:.2f}%")

                status.update(label="Experiment Complete!", state="complete")
        
        # 5. Artistic Plotting
        if results:
            st.markdown("### 📊 Visual Insights")
            
            # Create data for plotting
            plot_df = pd.DataFrame(list(results.items()), columns=["Metric", "Value"])
            
            # Artistic Bar Chart
            fig = px.bar(
                plot_df, 
                x="Metric", 
                y="Value", 
                color="Metric",
                title=f"Performance Profile: {selected_model}",
                template="plotly_dark",
                color_discrete_sequence=["#00dbde", "#fc00ff"]
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Click 'Search HuggingFace Hub' to begin your research.")