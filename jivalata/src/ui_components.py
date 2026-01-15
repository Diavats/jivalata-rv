"""
JIVALATA MVP - UI Components & Styles
=====================================
Contains CSS, animations, footer, and reusable UI helpers.
"""

import streamlit as st

# --- CSS Styles ---
LANDING_CSS = """
<style>
    /* Hide Streamlit defaults on landing */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Animated River Flow Background */
    .stApp {
        background: linear-gradient(90deg, #1a365d 0%, #2d4a6f 25%, #3b6e8f 50%, #2d4a6f 75%, #1a365d 100%);
        background-size: 200% 100%;
        animation: riverFlow 20s linear infinite;
    }
    
    @keyframes riverFlow {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }
    
    /* Floating Animation */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .floating {
        animation: float 3s ease-in-out infinite;
    }
    
    .floating-delayed {
        animation: float 3s ease-in-out infinite;
        animation-delay: 0.5s;
    }
    
    /* Landing Title - Enhanced */
    .landing-title {
        font-size: 7rem;
        font-weight: 700;
        color: #f0f8ff;
        text-align: center;
        text-shadow: 3px 3px 25px rgba(0,0,0,0.4);
        margin-bottom: 0.5rem;
        font-family: 'Georgia', serif;
    }
    
    /* Landing Tagline */
    .landing-tagline {
        font-size: 1.5rem;
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-style: italic;
        margin-bottom: 3rem;
    }
    
    /* CTA Button */
    .cta-container {
        display: flex;
        justify-content: center;
        margin-top: 2rem;
    }
    
    /* Footer */
    .custom-footer {
        position: fixed;
        bottom: 10px;
        left: 0;
        width: 100%;
        text-align: center;
        color: rgba(255,255,255,0.7);
        font-size: 0.9rem;
    }
    
    /* Region Selection Card */
    .region-card {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    
    /* Region Selection Custom Colors */
    .region-title {
        color: #2c5282;
        font-weight: 600;
    }
    
    .region-subtitle {
        color: #4a5568;
    }
</style>
"""

REGION_CSS = """
<style>
    /* Region Selection Gradient Background */
    .stApp {
        background: linear-gradient(180deg, #add8e6 0%, #90ee90 100%);
    }
</style>
"""

DASHBOARD_CSS = """
<style>
    /* Dashboard specific styles */
    .dia-agent {
        background: linear-gradient(135deg, #e6fffa 0%, #ebf8ff 100%);
        border-left: 4px solid #319795;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 1rem;
    }
    
    .dia-header {
        font-weight: 600;
        color: #234e52;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .dia-emoji {
        display: inline-block;
        animation: float 2s ease-in-out infinite;
        font-size: 1.3rem;
    }
    
    .dia-text {
        color: #2d3748;
        line-height: 1.6;
        font-size: 1.1rem;
    }
    
    /* Priority Legend */
    .legend-container {
        display: flex;
        gap: 1rem;
        justify-content: center;
        margin: 1rem 0;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .legend-dot {
        width: 15px;
        height: 15px;
        border-radius: 50%;
    }
    
    /* Footer for dashboard */
    .dashboard-footer {
        position: fixed;
        bottom: 5px;
        left: 0;
        width: 100%;
        text-align: center;
        color: #718096;
        font-size: 0.8rem;
        background: rgba(255,255,255,0.9);
        padding: 5px 0;
    }
</style>
"""


def inject_landing_css():
    """Inject CSS for landing page."""
    st.markdown(LANDING_CSS, unsafe_allow_html=True)


def inject_region_css():
    """Inject CSS for region selection page."""
    st.markdown(REGION_CSS, unsafe_allow_html=True)


def inject_dashboard_css():
    """Inject CSS for dashboard page."""
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


def render_footer():
    """Render the global footer."""
    st.markdown(
        '<div class="custom-footer">Made by Dia Vats</div>',
        unsafe_allow_html=True
    )


def render_dashboard_footer():
    """Render footer for dashboard (different styling)."""
    st.markdown(
        '<div class="dashboard-footer">Made by Dia Vats</div>',
        unsafe_allow_html=True
    )


def render_dia_message(message: str, emoji: str = "👧"):
    """Render a message from the DIA agent."""
    st.markdown(f'''
    <div class="dia-agent">
        <div class="dia-header">
            <span class="dia-emoji">{emoji}</span>
            <span>Dia says:</span>
        </div>
        <div class="dia-text">{message}</div>
    </div>
    ''', unsafe_allow_html=True)


def render_priority_legend():
    """Render the priority color legend."""
    st.markdown('''
    <div class="legend-container">
        <div class="legend-item">
            <div class="legend-dot" style="background: #e53e3e;"></div>
            <span>High Priority</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: #dd6b20;"></div>
            <span>Medium Priority</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: #ecc94b;"></div>
            <span>Lower Priority</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)
