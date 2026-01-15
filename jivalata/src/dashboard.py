"""
JIVALATA MVP - Module 5: Interactive Dashboard (Full UI/UX)
==========================================================
Main entry point with Landing Page, Region Selection, and Simulation Dashboard.
Integrates Modules 1-4 with a polished user experience.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path

# Import UI Components
try:
    from src.ui_components import (
        inject_landing_css, inject_dashboard_css,
        render_footer, render_dashboard_footer,
        render_dia_message, render_priority_legend
    )
    from src.data_loader import load_features, load_data
    from src.flood_risk_model import FloodRiskModel
    from src.simulation import RestorationSimulator
    from src.priority_scoring import PriorityScorer
except ModuleNotFoundError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
    from src.ui_components import (
        inject_landing_css, inject_dashboard_css,
        render_footer, render_dashboard_footer,
        render_dia_message, render_priority_legend
    )
    from src.data_loader import load_features, load_data
    from src.flood_risk_model import FloodRiskModel
    from src.simulation import RestorationSimulator
    from src.priority_scoring import PriorityScorer

# --- Configuration ---
DATA_DIR = Path("data")
DEM_PATH = DATA_DIR / "haridwar_merged_dem.tif"
NDVI_PATH = DATA_DIR / "ndvi_aligned_to_dem.tif"
FEAT_PATH = DATA_DIR / "ml_features_full.npy"

st.set_page_config(
    page_title="JIVALATA Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- State Initialization ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'landing'
if 'sim_results' not in st.session_state:
    st.session_state['sim_results'] = None
if 'ndvi_used' not in st.session_state:
    st.session_state['ndvi_used'] = 0.0
if 'elev_used' not in st.session_state:
    st.session_state['elev_used'] = 0.0


# --- Cached Resources ---
@st.cache_resource
def load_system():
    """Load data and train model (cached)."""
    if not DEM_PATH.exists():
        return None, None, None, None
    
    raw_data = load_data(str(DEM_PATH), str(NDVI_PATH))
    grid_shape = raw_data.metadata.shape
    feature_df = raw_data.feature_table

    if FEAT_PATH.exists():
        X = load_features(preprocessed_path=str(FEAT_PATH))
    else:
        X = load_features(dem_path=str(DEM_PATH), ndvi_path=str(NDVI_PATH))

    model = FloodRiskModel()
    model.train(X)

    return model, X, grid_shape, feature_df


def reshape_to_grid(flat_array, target_shape, df_indices):
    """Reconstruct 2D map from 1D array."""
    grid = np.full(target_shape, np.nan)
    rows = df_indices['row'].values
    cols = df_indices['col'].values
    grid[rows, cols] = flat_array
    return grid


# ============================================================
# PAGE 1: LANDING
# ============================================================
def render_landing():
    """Render the animated landing page."""
    inject_landing_css()
    
    # Spacer to center content
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
    
    # Floating Title
    st.markdown('''
    <div class="floating" style="text-align: center;">
        <h1 class="landing-title">JIVALATA</h1>
    </div>
    ''', unsafe_allow_html=True)
    
    # Floating Tagline
    st.markdown('''
    <div class="floating-delayed" style="text-align: center;">
        <p class="landing-tagline">Restoring floodplains. Reducing risk. Sustaining life.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # CTA Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🌊 Enter Simulation", type="primary", use_container_width=True):
            # Transition feedback
            with st.spinner("Preparing region controls..."):
                import time
                time.sleep(0.5)
            st.session_state['page'] = 'region_select'
            st.rerun()
    
    render_footer()


# ============================================================
# PAGE 2: REGION SELECTION
# ============================================================
def render_region_selection():
    """Render region selection with gated logic."""
    inject_region_css()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Centered Card
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h2 class="region-title">📍 Select Region</h2>', unsafe_allow_html=True)
        st.markdown('<p class="region-subtitle">Choose your area of interest for flood risk analysis.</p>', unsafe_allow_html=True)
        st.markdown("---")
        
        # State Dropdown
        states = ["Uttarakhand", "Uttar Pradesh (Coming Soon)", "Bihar (Coming Soon)"]
        selected_state = st.selectbox("State", states, index=0)
        
        # District Dropdown
        if selected_state == "Uttarakhand":
            districts = ["Haridwar", "Tehri Garhwal (Coming Soon)", "Dehradun (Coming Soon)"]
        else:
            districts = ["Select state first"]
        selected_district = st.selectbox("District", districts, index=0)
        
        # Sub-region Dropdown
        if selected_district == "Haridwar":
            subregions = ["NE", "NW (Coming Soon)", "SE (Coming Soon)", "SW (Coming Soon)"]
        else:
            subregions = ["Select district first"]
        selected_subregion = st.selectbox("Sub-region", subregions, index=0)
        
        st.markdown("---")
        
        # Validation
        is_valid = (
            selected_state == "Uttarakhand" and
            selected_district == "Haridwar" and
            selected_subregion == "NE"
        )
        
        if is_valid:
            if st.button("🚀 Proceed to Dashboard", type="primary", use_container_width=True):
                # Transition feedback
                with st.spinner("Initializing simulation workspace..."):
                    import time
                    time.sleep(0.5)
                st.session_state['page'] = 'dashboard'
                st.rerun()
        else:
            st.warning("⚠️ Only **Uttarakhand → Haridwar → NE** is currently available. Other regions coming soon!")
            st.button("🚀 Proceed to Dashboard", type="primary", use_container_width=True, disabled=True)
    
    render_footer()


# ============================================================
# PAGE 3: MAIN DASHBOARD
# ============================================================
def render_dashboard():
    """Render the main simulation dashboard."""
    inject_dashboard_css()
    
    # Custom CSS to reduce top whitespace
    st.markdown('<style>div.block-container{padding-top:1.5rem;}</style>', unsafe_allow_html=True)
    
    # Title
    st.title("JIVALATA: NE HARIDWAR FLOOD RISK ANALYSIS")
    st.markdown("*Decision Intelligence for Floodplain Restoration*")
    
    # Load System with feedback
    with st.spinner("🧠 DIA is loading the flood risk model for Haridwar (NE)..."):
        model, X, grid_shape, feature_df = load_system()
    
    if model is None:
        st.error("❌ Data files not found. Please ensure GeoTIFF files are in the data/ folder.")
        render_dashboard_footer()
        return
    
    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("🎛️ Simulation Controls")
        
        ndvi_increase = st.slider(
            "Vegetation Restoration (NDVI Increase)",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
            help="Simulate planting more vegetation."
        )
        
        ndvi_decrease = st.slider(
            "Vegetation Loss (NDVI Decrease)",
            min_value=-0.5,
            max_value=0.0,
            value=0.0,
            step=0.05,
            help="Simulate vegetation degradation/removal."
        )
        
        elevation_change = st.slider(
            "Elevation Modification (m)",
            min_value=-1.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Simulate local topography changes (e.g., embankments or dredging)."
        )
        
        run_sim = st.button("▶️ Run Simulation", type="primary", use_container_width=True)
        
        st.markdown("---")
        
        # Back button
        if st.button("← Back to Region Selection"):
            st.session_state['page'] = 'region_select'
            st.rerun()
    
    # --- DIA Agent: Introduction ---
    if st.session_state['sim_results'] is None:
        render_dia_message(
            "Welcome to the NE Haridwar flood risk simulator. "
            "This region is vulnerable due to its low elevation near river channels and sparse vegetation cover. "
            "**NDVI (Normalized Difference Vegetation Index)** measures greenness—higher values mean more vegetation. "
            "Increasing NDVI simulates restoration efforts like planting trees or creating buffer zones. "
            "The baseline map below shows current flood risk. Use the slider to simulate restoration impact. "
            "*Note: This is a simulation only—no real-world changes are applied.*",
            emoji="🧠"
        )
    
    # --- Run Simulation ---
    if run_sim:
        # Centered progress message with buffer
        placeholder = st.empty()
        with placeholder.container():
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center;'><h2>Running AI Simulation...</h2><p>Dia is calculating geospatial impacts</p></div>", unsafe_allow_html=True)
            import time
            time.sleep(1.0) # Artificial buffer for UX
            
            simulator = RestorationSimulator(model)
            # Combine NDVI changes
            total_ndvi_change = ndvi_increase + ndvi_decrease
            new_classes, risk_red_map = simulator.run_simulation(
                X, 
                ndvi_change=total_ndvi_change, 
                elevation_change=elevation_change
            )
            
            scorer = PriorityScorer()
            ranked_df = scorer.get_ranked_zones(feature_df, risk_red_map, top_n=50)
            
            st.session_state['sim_results'] = {
                'new_classes': new_classes,
                'risk_red_map': risk_red_map,
                'ranked_df': ranked_df
            }
            st.session_state['ndvi_used'] = total_ndvi_change
            st.session_state['elev_used'] = elevation_change
            
            time.sleep(0.5) # Finalizing buffer
        placeholder.empty()
        
        # Success confirmation
        st.success("✅ Simulation complete. Impacts calculated.")
    
    # --- Visualization: Risk Maps ---
    st.subheader("FLOOD RISK ASSESSMENT")
    
    col1, col2 = st.columns(2)
    
    base_classes = model.predict(X)
    
    with col1:
        st.markdown("**Current Baseline Risk**")
        base_map = reshape_to_grid(base_classes, grid_shape, feature_df[['row', 'col']])
        fig_base = px.imshow(
            base_map,
            color_continuous_scale=['#38a169', '#dd6b20', '#e53e3e'],
            title="Baseline Flood Risk",
            labels={'color': 'Risk Level'}
        )
        fig_base.update_layout(height=550, margin=dict(l=10, r=10, t=40, b=10))
        fig_base.update_coloraxes(colorbar_title="Risk<br>Level")
        st.plotly_chart(fig_base, use_container_width=True)
    
    with col2:
        st.markdown(f"**Simulated Scenario (+{st.session_state.get('ndvi_used', 0):.2f} NDVI)**")
        if st.session_state['sim_results']:
            sim_classes = st.session_state['sim_results']['new_classes']
            sim_map = reshape_to_grid(sim_classes, grid_shape, feature_df[['row', 'col']])
            
            fig_sim = px.imshow(
                sim_map,
                color_continuous_scale=['#38a169', '#dd6b20', '#e53e3e'],
                title="Simulated Flood Risk (After Restoration)",
                labels={'color': 'Risk Level'}
            )
            fig_sim.update_layout(height=550, margin=dict(l=10, r=10, t=40, b=10))
            fig_sim.update_coloraxes(colorbar_title="Risk<br>Level")
            st.plotly_chart(fig_sim, use_container_width=True)
            
            # Metrics
            n_high_base = int(np.sum(base_classes == base_classes.max()))
            n_high_sim = int(np.sum(sim_classes == sim_classes.max()))
            st.metric("High Risk Pixels Reduced", f"{n_high_base - n_high_sim:,}")
        else:
            st.info("👆 Click **Run Simulation** to see results.")
            
    # --- DIA Agent: Post-Simulation Analysis (Moved) ---
    if st.session_state['sim_results'] is not None:
        base_classes = model.predict(X)
        sim_classes = st.session_state['sim_results']['new_classes']
        
        n_level_0_base = int(np.sum(base_classes == 0))
        n_level_0_sim = int(np.sum(sim_classes == 0))
        
        # High risk is generally max class
        high_class = base_classes.max()
        n_high_base = int(np.sum(base_classes == high_class))
        n_high_sim = int(np.sum(sim_classes == high_class))
        diff = n_high_base - n_high_sim
        
        ndvi_val = st.session_state['ndvi_used']
        elev_val = st.session_state['elev_used']
        
        if diff > 0:
            msg = (
                f"With NDVI change of **{ndvi_val:.2f}** and elevation adjustment of **{elev_val:.1f}m**, "
                f"high-risk zones decreased by **{diff:,} pixels**. "
                f"On the priority map below, **red zones = highest restoration priority** (maximum flood reduction), "
                f"**orange = medium priority**, and **yellow = lower priority**."
            )
            render_dia_message(msg, emoji="👧")
        elif diff == 0:
            render_dia_message(
                f"With NDVI change of {ndvi_val:.2f} and elevation adjustment of {elev_val:.1f}m, "
                "no significant change in risk classification was detected. Higher intervention might be needed.",
                emoji="👧"
            )
        else:
            render_dia_message(
                f"Warning: With these parameters (NDVI: {ndvi_val:.2f}, Elev: {elev_val:.1f}m), "
                f"High-risk zones appear to have **increased** by {abs(diff):,} pixels.",
                emoji="👧"
            )
    
    # --- Visualization: Priority Map ---
    st.markdown("---")
    st.subheader("RESTORATION PRIORITY ZONES")
    st.caption("Red = Highest priority for restoration (maximum flood risk reduction)")
    render_priority_legend()
    
    if st.session_state['sim_results']:
        col3, col4 = st.columns([2, 1])
        
        with col3:
            scorer = PriorityScorer()
            slope_vals = feature_df['slope'].values
            risk_vals = st.session_state['sim_results']['risk_red_map']
            
            prio_vals, _ = scorer.compute_scores(risk_vals, slope_vals)
            prio_map = reshape_to_grid(prio_vals, grid_shape, feature_df[['row', 'col']])
            
            fig_prio = px.imshow(
                prio_map,
                color_continuous_scale=['#ecc94b', '#dd6b20', '#e53e3e'],
                title="Priority Score Heatmap (Red = Highest Priority)"
            )
            fig_prio.update_layout(height=550, margin=dict(l=10, r=10, t=40, b=10))
            fig_prio.update_coloraxes(colorbar_title="Priority<br>Score")
            st.plotly_chart(fig_prio, use_container_width=True)
        
        with col4:
            st.markdown("**📋 Top Recommended Zones**")
            st.caption("Grid Location (Row, Column) — These correspond to positions on the displayed priority map above.")
            ranked = st.session_state['sim_results']['ranked_df']
            
            disp_df = ranked[['row', 'col', 'elevation', 'priority_score']].head(10).copy()
            disp_df.columns = ['Grid Row', 'Grid Col', 'Elevation (m)', 'Priority Score']
            disp_df['Priority Score'] = disp_df['Priority Score'].apply(lambda x: f"{x:.2e}")
            st.dataframe(disp_df, hide_index=True, use_container_width=True)
            
            csv = ranked.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Full Plan (CSV)",
                csv,
                "restoration_plan.csv",
                "text/csv",
                use_container_width=True
            )
    else:
        st.info("Run simulation to generate priority ranking.")
    
    render_dashboard_footer()


# ============================================================
# MAIN CONTROLLER
# ============================================================
def main():
    """Main application controller."""
    page = st.session_state.get('page', 'landing')
    
    if page == 'landing':
        render_landing()
    elif page == 'region_select':
        render_region_selection()
    elif page == 'dashboard':
        render_dashboard()
    else:
        st.session_state['page'] = 'landing'
        st.rerun()


if __name__ == "__main__":
    main()
