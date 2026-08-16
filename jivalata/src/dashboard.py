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
        inject_landing_css, inject_region_css, inject_dashboard_css,
        render_footer, render_dashboard_footer,
        render_dia_message, render_priority_legend,
        render_kpi_cards, render_transition_prompt, render_intro_overlay
    )
    from src.data_loader import load_features, load_data, load_population, load_wetland_sensitivity
    from src.flood_risk_model import FloodRiskModel
    from src.simulation import RestorationSimulator
    from src.priority_scoring import PriorityScorer
except ModuleNotFoundError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
    from src.ui_components import (
        inject_landing_css, inject_region_css, inject_dashboard_css,
        render_footer, render_dashboard_footer,
        render_dia_message, render_priority_legend,
        render_kpi_cards, render_transition_prompt, render_intro_overlay
    )
    from src.data_loader import load_features, load_data, load_population, load_wetland_sensitivity
    from src.flood_risk_model import FloodRiskModel
    from src.simulation import RestorationSimulator
    from src.priority_scoring import PriorityScorer

# --- Configuration ---
# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DEM_PATH = DATA_DIR / "haridwar_merged_dem.tif"
NDVI_PATH = DATA_DIR / "ndvi_aligned_to_dem.tif"
FEAT_PATH = DATA_DIR / "ml_features_full.npy"
NODATA_VAL = -9999
POP_ALIGNED = DATA_DIR / "worldpop_2026_aligned_to_dem.tif"
POP_RAW = DATA_DIR / "worldpop_2026.tif"
WETLAND_PATH = DATA_DIR / "wetland_sensitivity_aligned_to_dem.tif"
WETLAND_PENALTY_STRENGTH = 0.40  # 40% max penalty for high sensitivity zones

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
if 'intro_shown' not in st.session_state:
    st.session_state['intro_shown'] = False
if 'sim_run_id' not in st.session_state:
    st.session_state['sim_run_id'] = 0


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
    
    # Load Population (Optional)
    pop_data = None
    pop_source = None
    try:
        if POP_ALIGNED.exists():
            pop_data = load_population(str(POP_ALIGNED))
            pop_source = POP_ALIGNED.name
        elif POP_RAW.exists():
            pop_data = load_population(str(POP_RAW))
            pop_source = POP_RAW.name
    except Exception as e:
        print(f"Warning: Could not load population data: {e}")
    
    # Load Wetland Sensitivity (Optional)
    wetland_data = None
    try:
        if WETLAND_PATH.exists():
            wetland_data = load_wetland_sensitivity(str(WETLAND_PATH))
    except Exception as e:
        print(f"Warning: Could not load wetland sensitivity data: {e}")

    return model, X, grid_shape, feature_df, pop_data, pop_source, wetland_data



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
    """Render the animated landing page with gradient background."""
    inject_landing_css()
    
    
    # Spacer to center content
    st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)
    
    # Title
    st.markdown('<div class="jivalata-landing-floating" style="text-align: center;"><h1 class="jivalata-landing-title">JĪVALATĀ</h1></div>', unsafe_allow_html=True)
    
    # Tagline
    st.markdown('<div class="jivalata-landing-floating-delayed" style="text-align: center;"><p class="jivalata-landing-tagline">Restoring floodplains. Reducing risk. Sustaining life.</p></div>', unsafe_allow_html=True)
    
    # Descriptor
    st.markdown('<div style="text-align: center;"><p class="jivalata-landing-descriptor">Decision Intelligence for Floodplain Restoration</p></div>', unsafe_allow_html=True)
    
    # CTA Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Select Region", type="primary", use_container_width=True):
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
        st.markdown('<h2 class="region-title">SELECT REGION</h2>', unsafe_allow_html=True)
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
def set_preset(ndvi_inc, ndvi_dec, elev, clear_sim=False):
    """Callback to update session state values from preset buttons."""
    st.session_state['ndvi_increase'] = ndvi_inc
    st.session_state['ndvi_decrease'] = ndvi_dec
    st.session_state['elevation_change'] = elev
    if clear_sim:
        st.session_state['sim_results'] = None
        st.session_state['ndvi_used'] = 0.0
        st.session_state['elev_used'] = 0.0
    st.session_state['sim_run_id'] += 1


def render_dashboard():
    """Render the main simulation dashboard."""
    inject_dashboard_css()
    
    # Custom CSS to reduce top whitespace
    st.markdown('<style>div.block-container{padding-top:1.5rem;}</style>', unsafe_allow_html=True)
    
    # Title
    st.title("JIVALATA: NE HARIDWAR FLOOD RISK ANALYSIS")
    st.markdown("*Decision Intelligence for Floodplain Restoration*")
    
    # Load System with feedback
    # Load System with feedback
    with st.spinner("🧠 DIA is loading the flood risk model for Haridwar (NE)..."):
        system_data = load_system()
        # Handle unpacking flexibly to support cached version updates
        pop_source = None
        wetland_data = None
        if len(system_data) == 7:
             model, X, grid_shape, feature_df, pop_data, pop_source, wetland_data = system_data
        elif len(system_data) == 6:
             model, X, grid_shape, feature_df, pop_data, pop_source = system_data
        elif len(system_data) == 5:
            model, X, grid_shape, feature_df, pop_data = system_data
        else:
             model, X, grid_shape, feature_df = system_data
             pop_data = None
        
        if pop_source:
             st.caption(f"✓ Using population raster: `{pop_source}`")
        else:
             st.caption("⚠️ Population raster not available (Metrics hidden)")
        
        if wetland_data is None:
             st.caption("ℹ️ Wetland sensitivity data not loaded (priority scoring uses base logic)")

    
    if model is None:
        st.error("❌ Data files not found. Please ensure GeoTIFF files are in the data/ folder.")
        render_dashboard_footer()
        return
        
    if pop_data is None:
        st.warning("⚠️ Population dataset not found. Exposure metrics will be unavailable.")
    
    # --- Intro Overlay (First Time Only) ---
    if not st.session_state['intro_shown']:
        st.markdown(render_intro_overlay(), unsafe_allow_html=True)
        # Transparent full-viewport dismiss button
        # We use a unique label but we will target it by position and section
        if st.button("DISMISS_INTRO", key="intro_dismiss_btn"):
            st.session_state['intro_shown'] = True
            st.rerun()
        
        # Inject CSS to make THIS button cover the screen
        # Target any button in the main section while intro is active
        st.markdown("""
            <style>
            .stMain div[data-testid="stButton"] button {
                position: fixed !important;
                top: 0 !important;
                left: 0 !important;
                width: 100vw !important;
                height: 100vh !important;
                background: transparent !important;
                border: none !important;
                color: transparent !important;
                z-index: 100000 !important;
                cursor: pointer !important;
                pointer-events: auto !important;
            }
            /* Ensure the text is hidden */
            .stMain div[data-testid="stButton"] button p {
                display: none !important;
            }
            </style>
        """, unsafe_allow_html=True)
    
    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("🎛️ Simulation Controls")
        
        # Initialize slider values if not set
        if 'ndvi_increase' not in st.session_state:
            st.session_state['ndvi_increase'] = 0.1
        if 'ndvi_decrease' not in st.session_state:
            st.session_state['ndvi_decrease'] = 0.0
        if 'elevation_change' not in st.session_state:
            st.session_state['elevation_change'] = 0.0
        
        # Sliders now use the session_state keys directly
        # No 'value' argument needed as it pulls from session_state[key]
        ndvi_increase = st.slider(
            "Vegetation Restoration (NDVI Increase)",
            min_value=0.0,
            max_value=0.5,
            step=0.05,
            help="Simulate planting more vegetation.",
            key='ndvi_increase'
        )
        
        ndvi_decrease = st.slider(
            "Vegetation Loss (NDVI Decrease)",
            min_value=-0.5,
            max_value=0.0,
            step=0.05,
            help="Simulate vegetation degradation/removal.",
            key='ndvi_decrease'
        )
        
        elevation_change = st.slider(
            "Elevation Modification (m)",
            min_value=-1.0,
            max_value=1.0,
            step=0.1,
            help="Simulate local topography changes (e.g., embankments or dredging).",
            key='elevation_change'
        )
        
        col1, col2 = st.columns(2)
        with col1:
            run_sim = st.button("▶️ Run Simulation", type="primary", use_container_width=True)
        with col2:
            if st.button("🔄 Reset", use_container_width=True, on_click=set_preset, args=(0.1, 0.0, 0.0, True)):
                st.rerun()
        
        st.markdown("---")
        st.markdown("**Preset Scenarios**")
        st.caption("Click to set slider values (does not auto-run)")
        
        # Preset buttons update the SAME session state keys as the sliders
        if st.button("🌿 Nature-based Restoration", use_container_width=True, on_click=set_preset, args=(0.25, -0.05, 0.10)):
            st.rerun()
        st.caption("Sets NDVI +0.25 | NDVI −0.05 | Elevation +0.10m")
        
        if st.button("⚖️ Mixed Intervention", use_container_width=True, on_click=set_preset, args=(0.30, -0.10, 0.35)):
            st.rerun()
        st.caption("Sets NDVI +0.30 | NDVI −0.10 | Elevation +0.35m")
        
        if st.button("🚧 Aggressive Intervention", use_container_width=True, on_click=set_preset, args=(0.40, -0.15, 0.70)):
            st.rerun()
        st.caption("Sets NDVI +0.40 | NDVI −0.15 | Elevation +0.70m")
        
        st.markdown("---")
        
        # Back button
        if st.button("← Back to Region Selection"):
            st.session_state['page'] = 'region_select'
            st.rerun()
    
    # --- DIA Agent: Pre-Simulation Introduction ---
    if st.session_state['sim_results'] is None:
        render_dia_message(
            "<ul>"
            "<li><strong>Why NE Haridwar is vulnerable:</strong> Low elevation near river channels combined with current vegetation patterns increases flood exposure.</li>"
            "<li><strong>How the system works:</strong> The model integrates elevation (DEM), vegetation index (NDVI), slope, and gridded population exposure data to quantify human risk under different restoration scenarios.</li>"
            "<li><strong>Your task:</strong> Use the sliders in the sidebar to test what-if restoration scenarios and observe their impact on flood risk.</li>"
            "</ul>"
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
            # Get base ranked zones
            ranked_df = scorer.get_ranked_zones(feature_df, risk_red_map, top_n=50)
            
            # Apply wetland penalty to priority scores if wetland data available
            if wetland_data is not None:
                # Extract wetland values for each zone
                wetland_vals = wetland_data[ranked_df['row'].values, ranked_df['col'].values]
                
                # Apply penalty with floor
                base_scores = ranked_df['priority_score'].values
                adjusted_scores = base_scores * (1 - WETLAND_PENALTY_STRENGTH * wetland_vals)
                floor_scores = base_scores * 0.2
                ranked_df['priority_score'] = np.maximum(adjusted_scores, floor_scores)
                
                # Re-sort by adjusted scores
                ranked_df = ranked_df.sort_values('priority_score', ascending=False).reset_index(drop=True)
            
            st.session_state['sim_results'] = {
                'new_classes': new_classes,
                'risk_red_map': risk_red_map,
                'ranked_df': ranked_df
            }
            st.session_state['ndvi_used'] = total_ndvi_change
            st.session_state['elev_used'] = elevation_change
            st.session_state['sim_run_id'] += 1
            
            time.sleep(0.5) # Finalizing buffer
        placeholder.empty()
        
        # Success confirmation
        st.success("✅ Simulation complete. Impacts calculated.")
    
    # --- Visualization: Risk Maps ---
    st.subheader("FLOOD RISK ASSESSMENT")
    
    # Standard Plotly config for both maps
    chart_config = {
        'displaylogo': False,
        'responsive': True,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'fullscreen'],
        'displayModeBar': True
    }
    
    col1, col2 = st.columns(2)
    
    base_classes = model.predict(X)
    has_sim = st.session_state.get('sim_results') is not None
    
    with col1:
        st.markdown("**Current Baseline Risk**")
        base_map = reshape_to_grid(base_classes, grid_shape, feature_df[['row', 'col']])
        fig_base = px.imshow(
            base_map,
            color_continuous_scale=['#38a169', '#dd6b20', '#e53e3e'],
            title="Baseline Flood Risk",
            labels={'color': 'Risk Level'}
        )
        # Force identical layout and axis scaling to simulated map
        fig_base.update_layout(
            height=600,
            autosize=False,
            margin=dict(l=20, r=20, t=80, b=20),
            coloraxis_showscale=True,
            title_x=0.5,
            title_y=0.95
        )
        fig_base.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_base.update_coloraxes(colorbar_title="Risk<br>Level", colorbar_len=0.8)
        st.plotly_chart(
            fig_base, 
            use_container_width=True, 
            config=chart_config,
            key=f"base_map_run_{st.session_state['sim_run_id']}"
        )
    
    with col2:
        if has_sim:
            st.markdown(f"**Simulated Scenario (+{st.session_state.get('ndvi_used', 0):.2f} NDVI)**")
            sim_classes = st.session_state['sim_results']['new_classes']
            sim_map = reshape_to_grid(sim_classes, grid_shape, feature_df[['row', 'col']])
            
            fig_sim = px.imshow(
                sim_map,
                color_continuous_scale=['#38a169', '#dd6b20', '#e53e3e'],
                title="Simulated Flood Risk (After Restoration)",
                labels={'color': 'Risk Level'}
            )
            # Reference map layout - must be EXACTLY identical to baseline
            fig_sim.update_layout(
                height=600,
                autosize=False,
                margin=dict(l=20, r=20, t=80, b=20),
                coloraxis_showscale=True,
                title_x=0.5,
                title_y=0.95
            )
            fig_sim.update_yaxes(scaleanchor="x", scaleratio=1)
            fig_sim.update_coloraxes(colorbar_title="Risk<br>Level", colorbar_len=0.8)
            st.plotly_chart(
                fig_sim, 
                use_container_width=True, 
                config=chart_config,
                key=f"sim_map_run_{st.session_state['sim_run_id']}"
            )
        else:
            st.markdown("**Simulated Scenario**")
            st.info("👆 Click **Run Simulation** to see results.")

    # --- Metrics & DIA Insights ---
    if has_sim:
        st.markdown("---")
        st.subheader("IMPACT ANALYSIS")
        
        # 1. Define Standard Risk Threshold
        RISK_THRESHOLD = 0.7
        
        # 2. Get 1D Probability Arrays (High Risk Class is last column)
        # Baseline
        probs_base_all = model.predict_proba(X)
        base_risk_probs = probs_base_all[:, -1]
        
        # Simulated
        # We recover simulated probabilities from the reduction map:
        # risk_reduction = base_prob - sim_prob  =>  sim_prob = base_prob - risk_reduction
        risk_reduction = st.session_state['sim_results']['risk_red_map']
        sim_risk_probs = base_risk_probs - risk_reduction
        
        # 3. Create 1D Masks based on Threshold
        base_risk_mask_1d = base_risk_probs >= RISK_THRESHOLD
        sim_risk_mask_1d = sim_risk_probs >= RISK_THRESHOLD
        
        # Pixel Metrics (derived from threshold-based masks for consistency)
        n_high_base = int(np.sum(base_risk_mask_1d))
        n_high_sim = int(np.sum(sim_risk_mask_1d))
        diff_pixels = n_high_sim - n_high_base
        
        # Construct Base Message Parts
        direction = "reduced" if diff_pixels < 0 else ("increased" if diff_pixels > 0 else "unchanged")
        ndvi_val = st.session_state['ndvi_used']
        elev_val = st.session_state['elev_used']
        ndvi_sign = "+" if ndvi_val >= 0 else "−"
        elev_sign = "+" if elev_val >= 0 else "−"
        
        # Population Metrics Logic
        pop_kpis_ready = False
        pop_base = 0
        pop_sim = 0
        net_change_pop = 0
        pct_change_pop = None
        
        # Sanity Check & Calc
        if pop_data is not None:
             total_pop = np.nansum(pop_data)
             nonzero_pixels = np.count_nonzero(pop_data > 0)
             
             if total_pop <= 0 or nonzero_pixels == 0:
                 st.warning("⚠️ Population raster loaded but contains only 0 values (check alignment/nodata). Exposure metrics will be unreliable.")
             
             # Reshape 1D masks to 2D grid
             base_mask_grid = reshape_to_grid(base_risk_mask_1d, grid_shape, feature_df[['row', 'col']])
             sim_mask_grid = reshape_to_grid(sim_risk_mask_1d, grid_shape, feature_df[['row', 'col']])
             
             # Fill NaNs with False (invalid pixels are not high risk)
             base_mask_grid = np.nan_to_num(base_mask_grid, nan=0).astype(bool)
             sim_mask_grid = np.nan_to_num(sim_mask_grid, nan=0).astype(bool)
             
             # Calculate Exposure
             pop_base = np.nansum(pop_data[base_mask_grid])
             pop_sim = np.nansum(pop_data[sim_mask_grid])
             
             net_change_pop = pop_sim - pop_base
             
             if pop_base > 0:
                 pct_change_pop = (net_change_pop / pop_base) * 100
             else:
                 pct_change_pop = 0.0 if pop_base == 0 else None
             
             pop_kpis_ready = True

        # Render KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        if pop_kpis_ready:
            # Govt-style labels requested
            kpi1.metric("Population Exposed (Before Simulation)", f"{int(pop_base):,}")
            kpi2.metric("Population Exposed (After Simulation)", f"{int(pop_sim):,}")
            kpi3.metric("Change in Exposure (People)", f"{int(net_change_pop):+,}", delta_color="inverse")
            kpi4.metric("Change in Exposure (%)", f"{pct_change_pop:+.1f}%" if pct_change_pop is not None else "N/A", delta_color="inverse")
            
            # DIA Message Text Construction
            if net_change_pop < 0:
                line1 = f"Exposure reduced by {abs(int(net_change_pop)):,} people ({abs(pct_change_pop):.1f}%)."
            elif net_change_pop > 0:
                line1 = f"Exposure increased by {int(net_change_pop):,} people ({abs(pct_change_pop):.1f}%)."
            else:
                line1 = "No significant change in exposure."
        else:
            # Fallback to Pixel KPIs
            kpi1.metric("High Risk Pixels (Before)", f"{n_high_base:,}")
            kpi2.metric("High Risk Pixels (After)", f"{n_high_sim:,}")
            kpi3.metric("Net Change (Pixels)", f"{diff_pixels:+,}", delta_color="inverse")
            kpi4.metric("Change (%)", "N/A")
            
            line1 = f"High-risk area {direction} by {abs(diff_pixels):,} pixels."
            if pop_data is None:
                line1 += " (Population metrics unavailable)."

        # Shared DIA Lines
        line2 = f"Drivers: NDVI increase = {ndvi_sign}{abs(ndvi_val):.2f}, Elevation = {elev_sign}{abs(elev_val):.1f}m."
        line3 = "Next: Proceed to Restoration Priority Zones below to identify intervention locations."
        
        # Add wetland penalty note if applicable
        wetland_note = ""
        if wetland_data is not None:
            wetland_note = "\n\nEcologically sensitive wetland zones receive moderated restoration priority to avoid habitat disturbance."
        
        render_dia_message(f"{line1}\n\n{line2}\n\n{line3}{wetland_note}")
        
        # Transition Line
        st.markdown("<div style='text-align: center; color: gray; font-style: italic; margin-top: 10px;'>Proceed to Restoration Priority Zones below to view recommended intervention areas.</div>", unsafe_allow_html=True)

        # Priority Zones Table            

    if st.session_state.get('sim_results'):
        # --- Visualization: Priority Map ---
        st.markdown("---")
        st.subheader("RESTORATION PRIORITY ZONES")
        render_priority_legend()
        
        col3, col4 = st.columns([2, 1])
        
        with col3:
            # Wetland Sensitivity Legend (if applicable)
            if wetland_data is not None:
                st.markdown("**🌿 Ecological Sensitivity (Wetland Buffer)**")
                st.caption("Blue overlay indicates wetland sensitivity zones with moderated restoration priority.")
                st.markdown("")
            
            scorer = PriorityScorer()
            slope_vals = feature_df['slope'].values
            risk_vals = st.session_state['sim_results']['risk_red_map']
            
            # Compute base priority scores
            prio_vals, _ = scorer.compute_scores(risk_vals, slope_vals)
            
            # Apply wetland penalty to priority scores if wetland data available
            # Use adjusted_score for EVERYTHING (heatmap, ranking, table)
            if wetland_data is not None:
                # Flatten wetland to 1D matching feature_df for scoring
                wetland_1d = wetland_data[feature_df['row'].values, feature_df['col'].values]
                
                # Apply penalty formula with floor
                adjusted_prio = prio_vals * (1 - WETLAND_PENALTY_STRENGTH * wetland_1d)
                floor_val = prio_vals * 0.2
                prio_vals = np.maximum(adjusted_prio, floor_val)
            
            prio_map = reshape_to_grid(prio_vals, grid_shape, feature_df[['row', 'col']])
            
            # Create priority heatmap (Base Layer)
            fig_prio = px.imshow(
                prio_map,
                color_continuous_scale=['#F2C94C', '#E6A23C', '#C84B4B'],
                title="Priority Score Heatmap"
            )
            
            # Add wetland overlay trace if available
            if wetland_data is not None:
                # Use wetland sensitivity directly for opacity/color
                # We use a white-to-blue colorscale, where 0 is transparent (white) and 1 is blue
                # To make it work as an overlay, we'll use a custom colorscale or just map the values
                
                # Plotly Heatmap overlay approach
                fig_prio.add_trace(
                    dict(
                        type='heatmap',
                        z=wetland_data,
                        colorscale='Blues',
                        showscale=False,
                        opacity=0.3,  # Fixed opacity as requested
                        hoverinfo='skip'
                    )
                )
            
            # Consistent layout with other maps
            fig_prio.update_layout(
                height=600,
                autosize=False,
                margin=dict(l=20, r=20, t=60, b=20),
                coloraxis_showscale=True,
                title_x=0.5,
                title_y=0.95,
                showlegend=False
            )
            fig_prio.update_yaxes(scaleanchor="x", scaleratio=1)
            fig_prio.update_coloraxes(colorbar_title="Priority<br>Score", colorbar_len=0.8)
            st.plotly_chart(fig_prio, use_container_width=True)
        
        with col4:
            st.markdown("**📋 Top Recommended Zones**")
            st.caption("Grid locations ranked by restoration impact potential")
            
            # We need to re-rank based on the adjust_prio if wetland was applied
            # The 'ranked_df' in session_state might be stale or base-only if not updated
            # Let's reconstruct the ranking dataframe here for display to be safe and consistent
            
            # Create a local DF for display
            local_df = feature_df.copy()
            local_df['priority_score'] = prio_vals # This has the penalty applied
            local_df = local_df.sort_values('priority_score', ascending=False).reset_index(drop=True)
            
            disp_df = local_df[['row', 'col', 'elevation', 'ndvi', 'priority_score']].head(10).copy()
            disp_df.columns = ['Row', 'Col', 'Elev (m)', 'NDVI', 'Score']
            disp_df['Elev (m)'] = disp_df['Elev (m)'].apply(lambda x: f"{x:.1f}")
            disp_df['NDVI'] = disp_df['NDVI'].apply(lambda x: f"{x:.3f}")
            disp_df['Score'] = disp_df['Score'].apply(lambda x: f"{x:.1e}")
            disp_df.index = range(1, len(disp_df) + 1)
            st.dataframe(disp_df, use_container_width=True)
            
            # Calculate and display ecological adjustment metric
            if wetland_data is not None:
                # Metric: Percentage of top 20 zones where wetland_sensitivity > 0.1
                top_20 = local_df.head(20)
                
                # Get wetland values for top 20 (need to lookup from 2D array or 1D if we had it)
                # We can map row/col back to wetland_data
                wetland_vals_top20 = wetland_data[top_20['row'].values, top_20['col'].values]
                
                # Count zones with sensitivity > 0.1
                moderated_count = np.sum(wetland_vals_top20 > 0.1)
                
                pct_moderated = (moderated_count / len(top_20)) * 100 if len(top_20) > 0 else 0
                
                st.markdown("")
                st.metric(
                    "Ecologically Sensitive Zones Moderated",
                    f"{pct_moderated:.0f}%",
                    help="Percentage of top 20 zones intersecting with significant wetland sensitivity (>0.1)"
                )
            
            st.markdown("")
            csv = local_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Full Plan (CSV)",
                csv,
                "restoration_plan.csv",
                "text/csv",
                use_container_width=True,
                help="Download complete restoration priority data for field implementation"
            )
    else:
        st.markdown("---")
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
