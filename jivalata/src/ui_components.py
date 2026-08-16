"""
JIVALATA MVP - UI Components & Styles
=====================================
Contains CSS, animations, footer, and reusable UI helpers.
"""

import streamlit as st
import base64
from pathlib import Path

# --- CSS Styles ---
LANDING_CSS = """
<style>
    /* Landing Page Specific Rules */
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
    @keyframes jivalata-float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .jivalata-landing-floating {
        animation: jivalata-float 3s ease-in-out infinite;
    }
    
    .jivalata-landing-floating-delayed {
        animation: jivalata-float 3s ease-in-out infinite;
        animation-delay: 0.5s;
    }
    
    /* Landing Title - Enhanced */
    .jivalata-landing-title {
        font-size: 5rem;
        font-weight: 700;
        color: #FFFFFF !important;
        text-align: center;
        text-shadow: 2px 4px 12px rgba(0,0,0,0.8);
        margin-bottom: 0.1rem;
        font-family: 'Georgia', serif;
        letter-spacing: 2px;
    }
    
    /* Landing Tagline */
    .jivalata-landing-tagline {
        font-size: 1.5rem;
        color: #FFFFFF !important;
        text-align: center;
        font-style: italic;
        text-shadow: 1px 2px 8px rgba(0,0,0,0.8);
        margin-bottom: 0.5rem;
        font-family: 'Georgia', serif;
    }

    /* Landing Descriptor */
    .jivalata-landing-descriptor {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.85) !important;
        text-align: center;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.6);
        margin-bottom: 2.5rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
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
    
    /* Center the card and shift upwards */
    .block-container {
        padding-top: 2rem !important;
    }
    
    .region-card {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        margin-top: -2rem; /* Shift upwards */
    }
    
    .region-title {
        color: #2c5282;
        font-weight: 700;
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .region-subtitle {
        color: #2d3748;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Style labels and dropdowns */
    label p {
        color: #2c5282 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        background-color: white;
        border-radius: 8px;
    }
</style>
"""

DASHBOARD_CSS = """
<style>
    /* Dashboard Background */
    .stApp {
        background: #F4F8FB;
    }
    
    /* Intro Overlay */
    .intro-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(31, 45, 61, 0.95);
        backdrop-filter: blur(10px);
        z-index: 9999;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        pointer-events: none; /* Let clicks pass through to the dismiss button */
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .intro-overlay.fade-out {
        animation: fadeOut 0.6s ease-out forwards;
    }
    
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; pointer-events: none; }
    }
    
    .intro-avatar-container {
        animation: diaFloat 3s ease-in-out infinite, diaWave 2s ease-in-out 1;
        margin-bottom: 2rem;
    }
    
    .intro-avatar-img {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        border: 4px solid #2C6E91;
        object-fit: cover;
        box-shadow: 0 10px 40px rgba(44, 110, 145, 0.3);
    }
    
    @keyframes diaWave {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        25% { transform: translateY(-8px) rotate(-5deg); }
        75% { transform: translateY(-8px) rotate(5deg); }
    }
    
    .intro-text {
        color: #F4F8FB;
        text-align: center;
        max-width: 600px;
        padding: 0 2rem;
    }
    
    .intro-text h2 {
        font-family: Georgia, serif;
        font-size: 2rem;
        margin-bottom: 1rem;
        color: #FFFFFF;
    }
    
    .intro-text p {
        font-size: 1.2rem;
        line-height: 1.6;
        margin-bottom: 0.8rem;
        color: #E9F1F7;
    }
    
    .intro-continue {
        margin-top: 2rem;
        font-size: 1rem;
        color: #4CAF8F;
        font-weight: 600;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* DIA Agent Box */
    /* DIA Agent Box */
    .dia-agent {
        background: #E9F1F7;
        border-left: 5px solid #2C6E91;
        padding: 1.5rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(44, 110, 145, 0.1);
        display: flex;
        align-items: flex-start;
        gap: 1.5rem;
    }
    
    .dia-avatar-container {
        flex-shrink: 0;
        animation: diaFloat 3s ease-in-out infinite;
    }
    
    .dia-avatar-img {
        width: 70px;
        height: 70px;
        border-radius: 50%;
        border: 3px solid #2C6E91;
        object-fit: cover;
    }
    
    @keyframes diaFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }
    
    .dia-content {
        flex-grow: 1;
    }
    
    .dia-header {
        font-weight: 700;
        color: #2C6E91;
        margin-bottom: 0.5rem;
        font-size: 1.3rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-family: Georgia, serif;
    }
    
    .dia-text {
        color: #1F2D3D;
        line-height: 1.7;
        font-size: 1.05rem;
    }
    
    .dia-text ul {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    
    .dia-text li {
        margin-bottom: 0.5rem;
        color: #1F2D3D;
    }
    
    .dia-text strong {
        color: #2C6E91;
    }
    
    /* KPI Cards */
    .kpi-container {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
        justify-content: center;
    }
    
    .kpi-card {
        background: #E9F1F7;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 3px 10px rgba(44, 110, 145, 0.08);
        text-align: center;
        min-width: 180px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(44, 110, 145, 0.15);
    }
    
    .kpi-card.improvement {
        border-color: #4CAF8F;
    }
    
    .kpi-card.worsening {
        border-color: #C84B4B;
    }
    
    .kpi-label {
        font-size: 0.9rem;
        color: #6B7C93;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1F2D3D;
    }
    
    .kpi-value.positive {
        color: #4CAF8F;
    }
    
    .kpi-value.negative {
        color: #C84B4B;
    }
    
    /* Transition Prompt */
    .transition-prompt {
        text-align: center;
        padding: 1rem;
        margin: 1.5rem 0;
        color: #6B7C93;
        font-size: 1.05rem;
        font-style: italic;
    }
    
    /* Priority Legend */
    .legend-container {
        display: flex;
        gap: 1.5rem;
        justify-content: center;
        margin: 1rem 0;
        padding: 1rem;
        background: #E9F1F7;
        border-radius: 8px;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    
    .legend-dot {
        width: 16px;
        height: 16px;
        border-radius: 50%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .legend-text {
        color: #1F2D3D;
        font-weight: 500;
        font-size: 0.95rem;
    }
    
    /* Control Panel Styling */
    .control-panel {
        background: #E9F1F7;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    
    /* Button Styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(44, 110, 145, 0.2);
    }
    
    /* Preset Buttons */
    .preset-buttons {
        display: flex;
        gap: 0.8rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    
    .preset-btn {
        background: #F4F8FB;
        border: 2px solid #2C6E91;
        color: #2C6E91;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        cursor: pointer;
        font-size: 0.9rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .preset-btn:hover {
        background: #2C6E91;
        color: #F4F8FB;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: Georgia, serif;
        color: #1F2D3D;
    }
    
    p, span, div {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Footer for dashboard */
    .dashboard-footer {
        position: fixed;
        bottom: 5px;
        left: 0;
        width: 100%;
        text-align: center;
        color: #6B7C93;
        font-size: 0.8rem;
        background: rgba(244, 248, 251, 0.95);
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
        '<div class="custom-footer" style="color: rgba(255,255,255,0.5); font-size: 0.8rem;">Crafted with ❤️ by Team TRIVENI • Dia Vats</div>',
        unsafe_allow_html=True
    )


def render_dashboard_footer():
    """Render footer for dashboard (different styling)."""
    st.markdown(
        '<div class="dashboard-footer">Made by Dia Vats</div>',
        unsafe_allow_html=True
    )


def get_base64_avatar():
    """Load and encode the DIA avatar image."""
    try:
        avatar_path = Path(__file__).parent / "assets" / "dia_avatar.png"
        if avatar_path.exists():
            with open(avatar_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
    except Exception:
        pass
    return ""





def render_dia_message(message: str, emoji: str = None):
    """Render a message from the DIA agent with custom avatar."""
    b64_avatar = get_base64_avatar()
    avatar_html = ""
    if b64_avatar:
        avatar_html = f'<img src="data:image/png;base64,{b64_avatar}" class="dia-avatar-img">'
    else:
        # Fallback to emoji if image missing
        avatar_html = f'<div class="dia-avatar-img" style="display:flex;align-items:center;justify-content:center;font-size:40px;background:white;">👧</div>'

    st.markdown(f'''
    <div class="dia-agent">
        <div class="dia-avatar-container">
            {avatar_html}
        </div>
        <div class="dia-content">
            <div class="dia-header">DIA</div>
            <div class="dia-text">{message}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)


def render_priority_legend():
    """Render the priority color legend with DIA explanation."""
    b64_avatar = get_base64_avatar()
    avatar_html = ""
    if b64_avatar:
        avatar_html = f'<img src="data:image/png;base64,{b64_avatar}" class="dia-avatar-img">'
    else:
        avatar_html = f'<div class="dia-avatar-img" style="display:flex;align-items:center;justify-content:center;font-size:40px;background:white;">👧</div>'
    
    st.markdown(f'''
    <div class="dia-agent">
        <div class="dia-avatar-container">
            {avatar_html}
        </div>
        <div class="dia-content">
            <div class="dia-header">Priority Zone Guide</div>
            <div class="dia-text">
                <div class="legend-container" style="justify-content: flex-start; background: transparent; padding: 0;">
                    <div class="legend-item">
                        <div class="legend-dot" style="background: #C84B4B;"></div>
                        <span class="legend-text">Highest priority</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-dot" style="background: #E6A23C;"></div>
                        <span class="legend-text">Medium priority</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-dot" style="background: #F2C94C;"></div>
                        <span class="legend-text">Low priority</span>
                    </div>
                </div>
                <strong>Start with red zones to maximize flood risk reduction.</strong>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)


def render_kpi_cards(n_high_base, n_high_sim, diff):
    """Render KPI summary cards after simulation."""
    improvement_class = "improvement" if diff > 0 else ("worsening" if diff < 0 else "")
    value_class = "positive" if diff > 0 else ("negative" if diff < 0 else "")
    
    st.markdown(f'''
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">High-Risk Pixels (Before)</div>
            <div class="kpi-value">{n_high_base:,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">High-Risk Pixels (After)</div>
            <div class="kpi-value">{n_high_sim:,}</div>
        </div>
        <div class="kpi-card {improvement_class}">
            <div class="kpi-label">Net Change</div>
            <div class="kpi-value {value_class}">{diff:+,}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)


def render_transition_prompt():
    """Render transition prompt to guide user to priority zones."""
    st.markdown('''
    <div class="transition-prompt">
        📍 Proceed below to identify where restoration efforts should be prioritized for maximum impact.
    </div>
    ''', unsafe_allow_html=True)


def render_intro_overlay():
    """Render the intro overlay with DIA introduction."""
    b64_avatar = get_base64_avatar()
    avatar_html = ""
    if b64_avatar:
        avatar_html = f'<img src="data:image/png;base64,{b64_avatar}" class="intro-avatar-img">'
    else:
        avatar_html = f'<div class="intro-avatar-img" style="display:flex;align-items:center;justify-content:center;font-size:60px;background:white;">👧</div>'
    
    overlay_html = f'''
    <div class="intro-overlay" id="intro-overlay">
        <div class="intro-avatar-container">
            {avatar_html}
        </div>
        <div class="intro-text">
            <h2>Hi, I'm DIA — your Decision Intelligence Assistant.</h2>
            <p>I'll help you explore flood risk and restoration impact in this region.</p>
        </div>
        <div class="intro-continue">Click anywhere to continue</div>
    </div>
    '''
    return overlay_html

