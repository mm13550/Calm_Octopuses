"""
ui_components/theme.py
======================
Centralized visual design system and CSS injection for the application.

This module defines:
1. The global CSS variables (colors, spacing, shadows).
2. Injected styles for standard Streamlit components (tabs, inputs, metrics).
3. Custom HTML components (hero section, section headers).
4. Page layout and typography defaults.
"""
from __future__ import annotations

from html import escape

import streamlit as st


def apply_global_theme() -> None:
    """Inject a light, minimal visual theme for the current page."""
    st.markdown(
        """
        <style>
        :root {
            --co-bg: #fbfaf8;
            --co-bg-accent: #fff2ea;
            --co-surface: #ffffff;
            --co-surface-soft: #fff8f3;
            --co-ink: #1f2328;
            --co-ink-muted: #69737d;
            --co-text: #1f2328;
            --co-text-muted: #69737d;
            --co-accent: #ef6a47;
            --co-accent-strong: #d94f33;
            --co-accent-soft: rgba(239, 106, 71, 0.10);
            --co-success: #46d483;
            --co-border: #e8dfd7;
            --co-border-strong: #d9cec4;
            --co-card-border: #e8dfd7;
            --co-shadow-lg: 0 18px 42px rgba(31, 35, 40, 0.06);
            --co-shadow-md: 0 10px 24px rgba(31, 35, 40, 0.05);
            --co-radius-xl: 30px;
            --co-radius-lg: 22px;
            --co-radius-md: 16px;
            --co-radius-sm: 12px;
        }

        html, body, [class*="css"] {
            font-family: "Inter", "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
            color-scheme: light;
        }

        p, li, label, .stMarkdown, .stText, .stCaption, .st-emotion-cache-10trblm, .st-emotion-cache-16idsys {
            color: var(--co-text);
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--co-text);
            font-family: "Inter", "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
            font-weight: 800;
            letter-spacing: -0.03em;
        }

        a {
            color: var(--co-accent-strong);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top right, rgba(239, 106, 71, 0.05), transparent 20rem),
                linear-gradient(180deg, #fffdfa 0%, var(--co-bg) 100%);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.35rem;
            padding-bottom: 3.25rem;
        }

        div[data-baseweb="tab-list"] {
            gap: 0.55rem;
            padding: 0.45rem;
            margin: 0.2rem 0 1.4rem 0;
            border: 1px solid var(--co-border);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.96);
        }

        button[data-baseweb="tab"] {
            min-height: 2.95rem;
            border-radius: 999px;
            padding: 0 1.05rem;
            color: var(--co-text-muted);
            background: transparent;
            font-weight: 700;
            transition: all 0.18s ease;
        }

        button[data-baseweb="tab"]:hover {
            color: var(--co-ink);
            background: var(--co-surface-soft);
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #ffffff;
            background: linear-gradient(135deg, var(--co-accent) 0%, var(--co-accent-strong) 100%);
            box-shadow: 0 8px 18px rgba(239, 106, 71, 0.18);
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--co-card-border);
            border-radius: var(--co-radius-xl);
            background: var(--co-surface);
            box-shadow: none;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] p,
        div[data-testid="stVerticalBlockBorderWrapper"] li,
        div[data-testid="stVerticalBlockBorderWrapper"] label,
        div[data-testid="stVerticalBlockBorderWrapper"] .stMarkdown,
        div[data-testid="stVerticalBlockBorderWrapper"] .stText,
        div[data-testid="stVerticalBlockBorderWrapper"] strong {
            color: var(--co-ink);
        }

        [data-testid="stMetric"] {
            border: 1px solid var(--co-border);
            border-radius: var(--co-radius-md);
            background: #ffffff;
            box-shadow: none;
            padding: 0.9rem 1rem;
        }

        [data-testid="stMetricLabel"] {
            color: var(--co-ink-muted);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }

        [data-testid="stMetricValue"] {
            color: var(--co-ink);
            font-weight: 800;
        }

        [data-testid="stTextInputRootElement"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stSlider"] label,
        [data-testid="stRadio"] label,
        [data-testid="stFileUploader"] label {
            color: var(--co-ink) !important;
            font-weight: 600;
        }

        [data-testid="stTextInputRootElement"] input,
        div[data-baseweb="select"] > div,
        div[data-testid="stFileUploaderDropzone"],
        textarea {
            border-radius: 18px !important;
            border: 1px solid var(--co-border-strong) !important;
            background: #ffffff !important;
            color: var(--co-ink) !important;
            box-shadow: none !important;
        }

        [data-testid="stTextInputRootElement"] input {
            min-height: 3.3rem;
            font-size: 1.1rem;
            font-weight: 600;
            caret-color: var(--co-ink) !important;
        }

        [data-testid="stTextInputRootElement"] input::placeholder,
        textarea::placeholder {
            color: #9aa2aa;
        }

        textarea {
            caret-color: var(--co-ink) !important;
        }

        [data-testid="stTextInputRootElement"] input:focus,
        textarea:focus {
            border-color: rgba(239, 106, 71, 0.80) !important;
            box-shadow: 0 0 0 1px rgba(239, 106, 71, 0.80) !important;
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input {
            color: var(--co-ink) !important;
        }

        div[data-baseweb="select"] svg {
            fill: var(--co-ink-muted) !important;
        }

        div[role="radiogroup"] label {
            color: var(--co-ink) !important;
        }

        div[data-baseweb="popover"] > div {
            background: #ffffff !important;
            border: 1px solid var(--co-border) !important;
            border-radius: 18px !important;
            box-shadow: var(--co-shadow-md) !important;
        }

        div[data-baseweb="popover"] ul,
        div[data-baseweb="popover"] [role="listbox"] {
            background: #ffffff !important;
            border-radius: 18px !important;
        }

        div[data-baseweb="popover"] li,
        div[data-baseweb="popover"] [role="option"],
        div[data-baseweb="popover"] [role="option"] * {
            color: var(--co-ink) !important;
            background: #ffffff !important;
        }

        div[data-baseweb="popover"] li:hover,
        div[data-baseweb="popover"] [role="option"]:hover,
        div[data-baseweb="popover"] [aria-selected="true"] {
            background: var(--co-surface-soft) !important;
        }

        .stSlider [data-baseweb="slider"] [role="slider"] {
            background: var(--co-accent);
            border: none;
        }

        .stSlider [data-baseweb="slider"] > div > div {
            background: var(--co-accent);
        }

        .stButton button,
        [data-testid="baseButton-secondary"] {
            border-radius: 999px;
            border: none;
            background: linear-gradient(135deg, var(--co-accent) 0%, var(--co-accent-strong) 100%);
            color: #fffdf9;
            font-weight: 800;
            letter-spacing: 0.01em;
            box-shadow: 0 8px 18px rgba(239, 106, 71, 0.16);
            transition: transform 0.16s ease, box-shadow 0.16s ease;
        }

        .stButton button:hover,
        [data-testid="baseButton-secondary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 20px rgba(239, 106, 71, 0.20);
        }

        [data-testid="stSuccess"],
        [data-testid="stInfo"],
        [data-testid="stWarning"] {
            border-radius: var(--co-radius-md);
            border: 1px solid var(--co-border);
            background: #ffffff;
            color: var(--co-ink);
        }

        [data-testid="stSuccess"] {
            border-color: rgba(70, 212, 131, 0.28);
            background: rgba(70, 212, 131, 0.10);
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--co-border);
            border-radius: var(--co-radius-md);
            background: #ffffff;
            overflow: hidden;
        }

        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary {
            background: #ffffff !important;
            color: var(--co-ink) !important;
        }

        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] details summary {
            color: var(--co-ink) !important;
            font-weight: 600;
        }

        [data-testid="stDataFrame"] {
            border-radius: var(--co-radius-lg);
            overflow: hidden;
            border: 1px solid var(--co-border);
            background: #ffffff;
            box-shadow: none;
        }

        [data-testid="stDataFrame"] * {
            color: var(--co-ink);
        }

        [data-testid="stTable"] table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            overflow: hidden;
            border: 1px solid var(--co-border);
            border-radius: 18px;
            background: #ffffff;
        }

        [data-testid="stTable"] th,
        [data-testid="stTable"] td {
            padding: 0.8rem 0.9rem;
            border-bottom: 1px solid #f0e7df;
            color: var(--co-ink) !important;
            text-align: left;
            vertical-align: top;
        }

        [data-testid="stTable"] th {
            background: #faf7f4;
            color: var(--co-ink-muted) !important;
            font-weight: 700;
        }

        [data-testid="stTable"] tr:last-child td {
            border-bottom: none;
        }

        [data-testid="stImage"] img {
            border-radius: 20px;
        }

        .co-hero {
            margin: 0 0 0.9rem 0;
            padding: 1.5rem 1.6rem 1.35rem 1.6rem;
            border-radius: 24px;
            border: 1px solid var(--co-border);
            background: #ffffff;
            box-shadow: none;
            text-align: left;
        }

        .co-eyebrow {
            margin: 0 0 0.6rem 0;
            color: var(--co-accent-strong);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.18em;
            text-transform: uppercase;
        }

        .co-hero-title {
            margin: 0;
            max-width: 13.5ch;
            color: var(--co-ink);
            font-size: clamp(2.1rem, 4vw, 3.25rem);
            line-height: 1.02;
        }

        .co-hero-copy {
            max-width: 46rem;
            margin: 0.8rem 0 0;
            color: var(--co-text-muted);
            font-size: 0.98rem;
            line-height: 1.6;
        }

        .co-hero-stats {
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            margin-top: 1rem;
        }

        .co-stat-pill {
            min-width: 10.5rem;
            padding: 0.72rem 0.85rem;
            border-radius: 16px;
            border: 1px solid var(--co-border);
            background: #fcfaf8;
        }

        .co-stat-label {
            display: block;
            color: var(--co-text-muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            font-weight: 700;
        }

        .co-stat-value {
            display: block;
            margin-top: 0.3rem;
            color: var(--co-ink);
            font-size: 1.02rem;
            font-weight: 700;
        }

        .co-section-intro {
            margin: 0.15rem 0 0.75rem 0;
        }

        .co-section-title {
            margin: 0.12rem 0 0.25rem 0;
            color: var(--co-ink);
            font-size: 2rem;
            line-height: 1.05;
        }

        .co-section-copy {
            margin: 0;
            max-width: 46rem;
            color: var(--co-text-muted);
            line-height: 1.6;
            font-size: 1rem;
        }

        .co-card-kicker {
            margin: 0 0 0.35rem 0;
            color: var(--co-ink-muted);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        .co-card-title {
            margin: 0;
            color: var(--co-ink);
            font-size: clamp(1.9rem, 3vw, 2.45rem);
            line-height: 1.02;
            font-weight: 800;
        }

        .co-card-heading {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
        }

        .co-michelin-medal {
            flex: 0 0 auto;
            min-width: 6.8rem;
            padding: 0.68rem 0.8rem;
            border-radius: 16px;
            border: 1px solid rgba(148, 57, 31, 0.20);
            background: linear-gradient(180deg, #fff8e8 0%, #fff0d2 100%);
            box-shadow: 0 8px 18px rgba(148, 57, 31, 0.10);
            text-align: center;
        }

        .co-michelin-medal--listed {
            background: #faf7f4;
            border-color: var(--co-border);
            box-shadow: none;
        }

        .co-michelin-medal-value {
            display: block;
            color: #94391f;
            font-size: 1.05rem;
            line-height: 1;
            font-weight: 900;
            letter-spacing: 0.04em;
        }

        .co-michelin-medal--listed .co-michelin-medal-value {
            color: var(--co-ink-muted);
            font-size: 0.72rem;
            letter-spacing: 0.12em;
        }

        .co-michelin-medal-label {
            display: block;
            margin-top: 0.28rem;
            color: var(--co-ink);
            font-size: 0.76rem;
            line-height: 1.1;
            font-weight: 800;
            white-space: nowrap;
        }

        .co-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.75rem 0 0.5rem 0;
        }

        .co-inline-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            border: 1px solid var(--co-border);
            color: var(--co-ink);
            background: #ffffff;
        }

        .co-inline-badge--soft {
            background: #faf7f4;
        }

        .co-inline-badge--accent {
            color: #94391f;
            background: rgba(239, 106, 71, 0.10);
            border-color: rgba(239, 106, 71, 0.18);
        }

        .co-card-id {
            margin: 0;
            color: var(--co-ink-muted);
            font-size: 0.84rem;
        }

        .co-link-pill {
            display: inline-flex;
            align-items: center;
            margin-top: 0.15rem;
            padding: 0.64rem 0.92rem;
            border-radius: 999px;
            background: rgba(239, 106, 71, 0.08);
            border: 1px solid rgba(239, 106, 71, 0.18);
            color: #9a3b24;
            text-decoration: none;
            font-weight: 800;
        }

        .co-link-pill:hover {
            background: rgba(239, 106, 71, 0.14);
            color: #7f2f1a;
        }

        .co-result-meta {
            margin-bottom: 0.85rem;
        }

        .co-menu-match {
            margin-bottom: 0.55rem;
            padding: 0.78rem 0.92rem;
            border-radius: 16px;
            border: 1px solid rgba(239, 106, 71, 0.14);
            background: rgba(239, 106, 71, 0.06);
            color: var(--co-ink);
            line-height: 1.55;
        }

        .co-menu-match strong {
            color: #8c341c;
        }

        .co-menu-item {
            margin: 0 0 0.45rem 0;
            padding: 0.7rem 0.85rem;
            border-radius: 14px;
            border: 1px solid var(--co-border);
            background: #ffffff;
            color: var(--co-ink);
            line-height: 1.55;
            font-size: 0.95rem;
        }

        .co-note {
            margin-top: -0.15rem;
            margin-bottom: 0.7rem;
            color: var(--co-text-muted);
            font-size: 0.94rem;
        }

        .co-content-card {
            margin-bottom: 0.75rem;
            padding: 1rem 1.05rem;
            border: 1px solid var(--co-border);
            border-radius: 18px;
            background: #ffffff;
        }

        .co-content-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.75rem;
        }

        .co-content-title {
            margin: 0;
            color: var(--co-ink);
            font-size: 1rem;
            font-weight: 700;
            line-height: 1.35;
        }

        .co-content-copy {
            margin: 0.45rem 0 0 0;
            color: var(--co-ink-muted);
            line-height: 1.6;
            font-size: 0.95rem;
        }

        .co-review-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-bottom: 0.55rem;
        }

        .co-review-body {
            margin: 0;
            color: var(--co-ink);
            line-height: 1.7;
            font-size: 0.98rem;
        }

        .co-raw-shell {
            margin-top: 1rem;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-top: 0.95rem;
            }

            .co-hero {
                padding: 1.25rem 1.05rem 1.1rem 1.05rem;
                border-radius: 20px;
            }

            .co-hero-title {
                max-width: 14ch;
            }

            .co-stat-pill {
                width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_hero(total_restaurants: int) -> None:
    """Render the top-of-page hero panel."""
    st.markdown(
        f"""
        <section class="co-hero">
            <p class="co-eyebrow">Calm Octopuses</p>
            <h1 class="co-hero-title">Search Michelin NYC by dish, mood, and taste.</h1>
            <p class="co-hero-copy">
                A multimodal restaurant discovery engine focused on what people actually choose:
                the dishes, the atmosphere, and the places most likely to fit the moment.
            </p>
            <div class="co-hero-stats">
                <div class="co-stat-pill">
                    <span class="co-stat-label">Catalog</span>
                    <span class="co-stat-value">{total_restaurants} restaurants</span>
                </div>
                <div class="co-stat-pill">
                    <span class="co-stat-label">Search</span>
                    <span class="co-stat-value">Text, image, and exact dish</span>
                </div>
                <div class="co-stat-pill">
                    <span class="co-stat-label">Personalization</span>
                    <span class="co-stat-value">Ratings into recommendations</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(eyebrow: str, title: str, description: str) -> None:
    """Render a lightweight section heading block."""
    st.markdown(
        f"""
        <section class="co-section-intro">
            <p class="co-eyebrow">{escape(eyebrow)}</p>
            <h2 class="co-section-title">{escape(title)}</h2>
            <p class="co-section-copy">{escape(description)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
