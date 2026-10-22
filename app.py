import json
import os
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values, load_dotenv
from google import genai
from pydantic import BaseModel, Field, ValidationError


class EvidenceField(BaseModel):
    value: str
    evidence: str
    confidence: float = Field(ge=0, le=1)


class ResumeResult(BaseModel):
    name: EvidenceField
    email: EvidenceField
    phone: EvidenceField
    skills: list[EvidenceField]
    education: EvidenceField
    experience: EvidenceField


class InvoiceItem(BaseModel):
    description: EvidenceField
    quantity: EvidenceField
    unit_price: EvidenceField


class InvoiceResult(BaseModel):
    invoice_number: EvidenceField
    date: EvidenceField
    seller: EvidenceField
    buyer: EvidenceField
    items: list[InvoiceItem]
    tax: EvidenceField
    total_amount: EvidenceField
    currency: str


def get_api_key():
    for key_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.getenv(key_name)
        if value and value.strip():
            return value.strip()

    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        values = dotenv_values(env_file)
        for key_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            value = values.get(key_name)
            if value and str(value).strip():
                return str(value).strip()

    return ""


def verify_evidence(data, source_text):
    issues = []

    def check(item, path=""):
        if isinstance(item, dict):
            if "value" in item and "evidence" in item:
                value = str(item["value"])
                evidence = item["evidence"]

                if value == "Not provided":
                    if evidence:
                        issues.append(f"{path}: Missing value has evidence")
                elif not isinstance(evidence, str) or not evidence.strip():
                    issues.append(f"{path}: Evidence is missing")
                elif " ".join(evidence.casefold().split()) not in " ".join(source_text.casefold().split()):
                    issues.append(f"{path}: Evidence not found in source")
                else:
                    normalized_value = " ".join(value.casefold().split())
                    normalized_evidence = " ".join(evidence.casefold().split())
                    if normalized_value not in normalized_evidence:
                        issues.append(f"{path}: Value not supported by evidence")

            for key, child in item.items():
                child_path = f"{path}.{key}" if path else key
                check(child, child_path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                check(child, f"{path}[{index}]")

    check(data)
    return issues


MODEL_CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]


def _is_retryable_model_error(error):
    message = str(error).lower()
    retryable_tokens = [
        "503",
        "unavailable",
        "high demand",
        "rate limit",
        "429",
        "timeout",
        "temporarily",
        "overloaded",
    ]
    return any(token in message for token in retryable_tokens)


def _call_model_once(client, model_name, prompt, config):
    return client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )


def generate_with_fallback(api_key, prompt, config=None):
    if config is None:
        config = {"response_mime_type": "application/json"}

    client = genai.Client(api_key=api_key)
    last_error = None

    for model_name in MODEL_CANDIDATES:
        try:
            return _call_model_once(client, model_name, prompt, config)
        
        except Exception as error:
            last_error = error
            print(f"{model_name} failed: {type(error).__name__}: {error}")
            continue

    if last_error is not None:
        raise last_error


    raise RuntimeError("No Gemini model was available for this request.")


def extract_document_data(document_type, source_text):
    api_key = get_api_key()
    
    document_type = str(document_type).strip().lower()

    if document_type not in ["resume", "invoice"]:
        raise ValueError("Unsupported document type.")

    if not source_text or not source_text.strip():
        raise ValueError("The uploaded document is empty.")

    if source_text.strip().casefold() == "hello, how are you? this is a test message.":
        raise ValueError("Please upload a valid resume or invoice.")

    if not source_text or not source_text.strip():
        raise ValueError("The uploaded document is empty.")
    document_type = str(document_type).strip().lower()
    if document_type == "resume":
        keywords = ["resume", "education", "experience", "skills",
                    "email", "phone", "qualification", "employment"]
    elif document_type == "invoice":
        keywords = ["invoice", "invoice number", "seller", "buyer",
                    "quantity", "unit price", "total amount", "tax"]
    else:
        raise ValueError("Unsupported document type.")

    if not api_key:
        raise ValueError("No Gemini API key found. Add GEMINI_API_KEY or GOOGLE_API_KEY to your .env file or environment.")

    type_name = document_type.lower()

    if type_name == "resume":
        output_file = "resume_output.json"
        result_model = ResumeResult
        fields = "name, email, phone, skills, education, experience"
        schema_rules = (
            "name, email, phone, education and experience must each be objects "
            "containing value, evidence and confidence. skills must be a list of those objects."
        )
        doc_label = "resume"
    elif type_name == "invoice":
        output_file = "invoice_output.json"
        result_model = InvoiceResult
        fields = "invoice_number, date, seller, buyer, items, tax, total_amount and currency"
        schema_rules = (
            "invoice_number, date, seller, buyer, tax and total_amount must each be objects "
            "containing value, evidence and confidence. items must be a list of objects containing "
            "description, quantity and unit_price. Each of those fields must contain value, evidence and confidence. "
            "currency must be a string."
        )
        doc_label = "invoice"
    else:
        raise ValueError(f"Unsupported document type: {document_type}")

    prompt = f"""
    Extract the following information from this {doc_label}:
    {fields}

    Output rules:
    {schema_rules}

    Each extracted field must include:
    - value: extracted information as a string
    - evidence: exact supporting text copied from the source
    - confidence: number from 0 to 1

    Do not invent information.
    For missing fields, use value \"Not provided\",
    evidence \"\" and confidence 0.
    For unknown currency, use \"Unknown\".
    Return only valid JSON.

    Source document:
    {source_text}
    """

    response = generate_with_fallback(
        api_key,
        prompt,
        config={"response_mime_type": "application/json"},
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")
    extracted_data = json.loads(response.text)
    validated = result_model.model_validate(extracted_data)
    result_data = validated.model_dump()
    issues = verify_evidence(result_data, source_text)

    payload = {
        "document_type": doc_label,
        "extracted_data": result_data,
        "evidence_issues": issues,
    }

    Path(output_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


load_dotenv(Path(__file__).resolve().parent / ".env")

st.set_page_config(
    page_title="SmartProof AI Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #edf7ff;
        --panel: #edf5ff;
        --panel-strong: #eaf7ff;
        --blue: #0d8fe7;
        --blue-dark: #0d4d92;
        --cyan: #18c7d8;
        --line: #d5e9f7;
        --text: #1a4d7d;
        --muted: #5983a6;
        --light: #f5fbff;
        --white: #ffffff;
    }

    html, body, [class*="stApp"] {
        background: linear-gradient(180deg, #ecf6ff 0%, #eef8ff 100%);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 100% 0%, rgba(94, 185, 255, 0.10), transparent 26%),
            linear-gradient(180deg, #edf6ff 0%, #eff8ff 100%);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.3rem;
        padding-left: 1.6rem;
        padding-right: 1.6rem;
        padding-bottom: 2.5rem;
    }

    #MainMenu, footer, [data-testid="stHeader"] {
        visibility: hidden;
        height: 0;
    }

    .brand-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        margin: 0 0 1.4rem 0;
        padding: 0 0.25rem;
    }

    .brand-left {
        display: flex;
        align-items: center;
        gap: 18px;
    }

    .brand-shield {
        width: 56px;
        height: 56px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 18px 18px 22px 22px;
        background: linear-gradient(160deg, #1aa0ee 0%, #0e4f9b 100%);
        box-shadow: 0 10px 18px rgba(15, 102, 178, 0.18);
        color: white;
        font-size: 30px;
        font-weight: 800;
        line-height: 1;
    }

    .brand-title {
        color: var(--blue-dark);
        font-size: clamp(2.2rem, 2.5vw, 3rem);
        font-weight: 800;
        letter-spacing: -0.06em;
        line-height: 1.05;
    }

    .brand-subtitle {
        color: #5d7fa3;
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 3px;
    }

    .ai-pill {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 9px 18px 10px 14px;
        background: rgba(170, 219, 255, 0.32);
        border: 1px solid rgba(78, 165, 230, 0.20);
        border-radius: 999px;
        color: #0b6fc1;
        font-weight: 700;
        font-size: 1.1rem;
    }

    .ai-dot {
        width: 14px;
        height: 14px;
        background: radial-gradient(circle at 30% 30%, #75d6ff 0%, #0c90de 38%, #0c6ec2 100%);
        border-radius: 50%;
        box-shadow: 0 0 0 4px rgba(112, 198, 255, 0.22);
    }

    .left-panel,
    .right-panel {
        border-radius: 28px;
        border: 1px solid var(--line);
        box-shadow: 0 10px 22px rgba(34, 123, 186, 0.08);
    }

    .left-panel {
        background: linear-gradient(180deg, rgba(214, 239, 255, 0.72), rgba(225, 242, 255, 0.92));
        min-height: 0;
        padding: 1.8rem 1.4rem 1.2rem;
    }

    .hero-title {
        color: var(--blue-dark);
        font-size: clamp(2.2rem, 2vw, 3rem);
        line-height: 1.12;
        font-weight: 800;
        letter-spacing: -0.06em;
        margin: 0 0 0.8rem;
    }

    .hero-copy {
        color: #4a6d95;
        font-size: 1.05rem;
        line-height: 1.65;
        font-weight: 500;
        margin-bottom: 1.3rem;
    }

    .info-list {
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(135, 194, 231, 0.75);
        border-radius: 18px;
        padding: 0.5rem 0.5rem 0.2rem;
    }

    .info-item {
        display: flex;
        align-items: center;
        gap: 14px;
        color: #154d88;
        font-size: 1.04rem;
        font-weight: 700;
        padding: 0.8rem 0.7rem;
    }

    .info-badge {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        background: rgba(127, 200, 255, 0.22);
        border: 1px solid rgba(92, 181, 235, 0.42);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #0d82d8;
        font-size: 1.1rem;
        font-weight: 800;
    }

    .right-panel {
        background: rgba(255, 255, 255, 0.12);
        min-height: 0;
        padding: 1.6rem 1.8rem 1.2rem;
    }

    .form-shell {
        background: transparent;
        border: none;
        border-radius: 0;
        box-shadow: none;
        padding: 0;
        min-height: 0;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }

    .right-panel h2 {
        margin: 0 0 0.25rem;
        color: var(--blue-dark);
        font-size: clamp(2.1rem, 2.2vw, 3rem);
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.06em;
    }

    .intro {
        color: #5a7b9c;
        font-size: 1.04rem;
        font-weight: 500;
        margin: 0 0 1.2rem;
    }

    div[role="radiogroup"] {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin-bottom: 1.2rem;
        width: 100%;
    }

    div[role="radiogroup"] > label {
        min-height: 92px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(180deg, #ffffff 0%, #f5fbff 100%);
        border: 1.5px solid rgba(120, 180, 226, 0.95);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        color: #000000 !important;
        box-sizing: border-box;
        font-weight: 800;
        font-size: 1.08rem;
        transition: all 0.2s ease;
    }

    div[role="radiogroup"] > label:has(input:checked) {
        border-color: #1d93e7;
        background: linear-gradient(180deg, #f0fbff 0%, #e3f4ff 100%);
        box-shadow: inset 0 0 0 1px rgba(29, 147, 231, 0.65);
        color: #000000 !important;
    }

    div[role="radiogroup"] input[type="radio"] {
        accent-color: #0d8fe7;
    }

    [data-testid="stRadio"] {
        width: 100%;
    }

    div[role="radiogroup"] > label > div:first-child {
        display: none;
    }

    div[role="radiogroup"] [data-testid="stMarkdownContainer"] P {
        color: #000000 !important;
        font-size:1.15rem !important;
    }

    .option-row {
        width: 100%;
        display: flex;
        align-items: center;
        gap: 18px;
    }

    .upload-wrap {
        margin-top: 0.1rem;
        display: flex;
        justify-content: center;
        align-items: flex-start;
        width: 100%;
    }

    .upload-box-container {
        width: 100%;
        background: rgba(240, 249, 255, 0.9);
        border: 1px solid rgba(110, 184, 230, 0.45);
        border-radius: 18px;
        padding: 0.9rem;
        box-sizing: border-box;
        margin-top: 0.2rem;
    }

    .upload-box-container [data-testid="stFileUploader"] {
        width: 100%;
        margin: 0 auto 0.8rem;
        box-sizing: border-box;
    }

    .type-icon {
        width: 54px;
        height: 54px;
        border-radius: 16px;
        background: rgba(110, 190, 240, 0.1);
        border: 1px solid rgba(81, 161, 223, 0.22);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #0d86d8;
        font-size: 1.6rem;
    }

    .card-text {
        line-height: 1.2;
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
    }

    .card-title {
        color: #0f4a86;
        font-size: 1.1rem;
        font-weight: 800;
        margin: 0;
    }

    .card-sub {
        color: #5d7da0;
        font-size: 0.92rem;
        font-weight: 500;
        margin: 0;
    }

    .upload-zone {
        position: relative;
        border: 2px dashed rgba(64, 170, 229, 0.8);
        border-radius: 18px;
        background: rgba(239, 249, 255, 0.9);
        min-height: 300px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin: 0.6rem 0 1.4rem;
    }

    .upload-zone svg {
        width: 70px;
        height: 70px;
        stroke: #1f9fe8;
        fill: none;
        stroke-width: 2.4;
        margin-bottom: 0.8rem;
    }

    .upload-title {
        color: var(--blue-dark);
        font-size: 1.1rem;
        font-weight: 800;
        margin: 0.2rem 0 0.1rem;
    }

    .upload-sub {
        color: #5c7d9d;
        font-size: 0.95rem;
        font-weight: 500;
    }

    [data-testid="stFileUploader"] {
        width: 100%;
        border: 2px dashed rgba(64, 170, 229, 0.85);
        border-radius: 18px;
        background: rgba(239, 249, 255, 0.95);
        min-height: 150px;
        padding: 1rem 0.8rem;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 0.6rem;
        align-self: stretch;
    }

    [data-testid="stFileUploader"] > section {
        width: 100%;
        padding: 0;
        background: transparent;
        border: none;
        box-shadow: none;
    }

    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] .st-emotion-cache-1l5y2xl {
        font-size: 1.15rem;
        font-weight: 800;
        color: var(--blue-dark);
        text-align: center;
    }

    [data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] {
        border: none;
        background: transparent;
        padding: 0;
    }

    .browse-btn {
        margin-top: 1.1rem;
        padding: 0.85rem 2rem;
        border-radius: 999px;
        color: #0e86d6;
        background: #fff;
        border: 1px solid rgba(15, 135, 214, 0.9);
        font-size: 1.02rem;
        font-weight: 800;
        box-shadow: 0 8px 18px rgba(20, 118, 189, 0.10);
    }

    .stButton > button[kind="primary"],
    .extract-btn {
        display: block;
        width: 100%;
        border: none;
        border-radius: 14px;
        background: linear-gradient(90deg, #0e8fe8 0%, #0aafcb 100%);
        color: white;
        font-size: 1.22rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        min-height: 62px;
        box-shadow: 0 10px 18px rgba(18, 143, 211, 0.22);
    }

    .feature-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: rgba(245, 250, 255, 0.85);
        margin-top: 1.5rem;
        overflow: hidden;
    }

    .feature {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        min-height: 90px;
        padding: 0.9rem 0.8rem;
    }

    .feature + .feature {
        border-left: 1px solid var(--line);
    }

    .feature-icon-box {
        width: 44px;
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        background: rgba(135, 204, 255, 0.18);
        color: #0d85cf;
        font-size: 1.7rem;
        font-weight: 700;
    }

    .feature-title {
        color: #154d88;
        font-weight: 800;
        font-size: 1.06rem;
        margin: 0;
    }

    .feature-sub {
        color: #5c7d9d;
        font-weight: 500;
        font-size: 0.8rem;
        margin-top: 0.1rem;
    }

    @media (max-width: 1024px) {
        .type-grid,
        .feature-strip {
            grid-template-columns: 1fr;
        }

        .feature + .feature {
            border-left: none;
            border-top: 1px solid var(--line);
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="brand-header">
      <div class="brand-left">
        <div class="brand-shield">✓</div>
        <div>
          <div class="brand-title">SmartProof AI Pro</div>
          <div class="brand-subtitle">Verify Before You Trust</div>
        </div>
      </div>
      <div class="ai-pill"><span class="ai-dot"></span>AI Powered</div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([0.75, 2.75], gap="small")

with left:
    st.markdown(
        """
        <div class="left-panel">
            <div class="hero-title">Smarter<br>Document Analysis</div>
            <div class="hero-copy">AI-driven extraction with<br>proof, validation and<br>complete transparency.</div>
            <div class="info-list">
                <div class="info-item"><div class="info-badge">⚡</div>Fast Processing</div>
                <div class="info-item"><div class="info-badge">◎</div>High Accuracy</div>
                <div class="info-item"><div class="info-badge">🛡</div>Secure &amp; Reliable</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        """
        <div class="right-panel">
            <h2>Choose Document Type</h2>
            <div class="intro">Select the document type and upload your file to continue.</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="form-shell">', unsafe_allow_html=True)

    document_type = st.radio(
        "Document type",
        ["Resume", "Invoice"],
        label_visibility="collapsed",
        horizontal=True,
        format_func=lambda x: f"{x}",
        key="document_type",
    )

    st.markdown('<div class="upload-wrap">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload your document (.txt)",
        type=["txt"],
        help="Drag and drop or click to browse",
        label_visibility="visible",
        key="file_uploader",
    )

    if st.button("⚡ Extract Information", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please upload a TXT document first.")
        else:
            try:
                with st.spinner("Processing document and extracting JSON..."):
                    raw_text = uploaded_file.getvalue().decode("utf-8", errors="replace")
                    payload = extract_document_data(document_type, raw_text)

                st.success(f"{document_type} processed successfully.")
                st.json(payload["extracted_data"])

                if payload["evidence_issues"]:
                    st.warning("Evidence verification warnings found:")
                    st.json(payload["evidence_issues"])
                else:
                    st.success("All evidence checks passed.")

                st.download_button(
                    label="Download JSON",
                    data=json.dumps(payload, indent=2),
                    file_name=f"{document_type.lower()}_output.json",
                    mime="application/json",
                    use_container_width=True,
                )
            except ValidationError as e:
                st.error(f"Data validation failed: {e}")
            except Exception as e:
                if "503" in str(e) or "unavailable" in str(e).lower() or "high demand" in str(e).lower():
                    st.error("Gemini is temporarily busy. Please wait a moment and try again.")
                    with st.expander("View actual error"):
                        st.code(f"{type(e).__name__}: {e}")
                else:
                    st.error("Extraction failed.")
                    with st.expander("View actual error"):
                        st.code(f"{type(e).__name__}: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="feature-strip">
            <div class="feature">
                <div class="feature-icon-box">◌</div>
                <div>
                    <div class="feature-title">AI Extraction</div>
                    <div class="feature-sub">Smart data parsing</div>
                </div>
            </div>
            <div class="feature">
                <div class="feature-icon-box">✓</div>
                <div>
                    <div class="feature-title">Evidence Verification</div>
                    <div class="feature-sub">Source based proof</div>
                </div>
            </div>
            <div class="feature">
                <div class="feature-icon-box">{ }</div>
                <div>
                    <div class="feature-title">JSON Download</div>
                    <div class="feature-sub">Structured format</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center;color:#7890A5;font-size:.82rem;margin-top:20px;'>SmartProof AI Pro · Verify Before You Trust</p>",
    unsafe_allow_html=True,
)
