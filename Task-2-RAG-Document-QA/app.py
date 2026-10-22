
# -------------------- PROFESSIONAL UI --------------------

st.markdown("""
<style>
    .stApp {
        background: #F7FAFC;
        color: #172B4D;
    }

    .block-container {
        max-width: 1600px;
        padding: 1.5rem 2rem 2rem;
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    .brand {
        font-size: 1.65rem;
        font-weight: 800;
        color: #15304B;
        letter-spacing: -0.5px;
    }

    .hero {
        background: linear-gradient(120deg, #15304B, #167D83);
        padding: 24px 28px;
        border-radius: 18px;
        color: #FFFFFF;
        margin-bottom: 20px;
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: 1.8rem;
        margin: 0 0 8px 0;
    }

    .hero p {
        color: #E1F5F4 !important;
        margin: 0;
        font-size: 0.95rem;
    }

    .panel-title {
        font-size: 1.08rem;
        font-weight: 750;
        color: #15304B;
        margin-bottom: 8px;
    }

    /* Chat message text */
    [data-testid="stChatMessage"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        color: #172B4D !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] div {
        color: #172B4D !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        color: #172B4D !important;
        background-color: #FFFFFF !important;
        -webkit-text-fill-color: #172B4D !important;
        caret-color: #172B4D !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748B !important;
        -webkit-text-fill-color: #64748B !important;
    }

    /* General text and source passages */
    .stMarkdown,
    .stMarkdown p,
    .stMarkdown li {
        color: #172B4D;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    #MainMenu,
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)
