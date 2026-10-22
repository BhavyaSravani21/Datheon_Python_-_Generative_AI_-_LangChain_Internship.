
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field, ValidationError


# ---------- DATA SCHEMAS ----------

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


# ---------- EVIDENCE VERIFICATION ----------

def verify_evidence(data, source_text):
    issues = []

    def check(item, path=""):
        if isinstance(item, dict):
            if "value" in item and "evidence" in item:
                value = str(item["value"])
                evidence = item["evidence"]

                if value == "Not provided":
                    if evidence:
                        issues.append(
                            f"{path}: Missing value has evidence"
                        )

                elif not isinstance(evidence, str) or not evidence.strip():
                    issues.append(f"{path}: Evidence is missing")

                elif evidence.casefold() not in source_text.casefold():
                    issues.append(
                        f"{path}: Evidence not found in source"
                    )

                else:
                    normalized_value = " ".join(
                        value.casefold().split()
                    )
                    normalized_evidence = " ".join(
                        evidence.casefold().split()
                    )

                    if normalized_value not in normalized_evidence:
                        issues.append(
                            f"{path}: Value not supported by evidence"
                        )

            for key, child in item.items():
                child_path = f"{path}.{key}" if path else key
                check(child, child_path)

        elif isinstance(item, list):
            for index, child in enumerate(item):
                check(child, f"{path}[{index}]")

    check(data)
    return issues


# ---------- DOCUMENT EXTRACTION ----------

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("API key not found. Check your .env file.")
        return

    print("\nSMARTPROOF AI PRO")
    print("1. Resume extraction")
    print("2. Invoice extraction")

    choice = input("Choose document type (1 or 2): ").strip()

    if choice == "1":
        input_file = "resume.txt"
        output_file = "resume_output.json"
        result_model = ResumeResult
        document_type = "resume"

        fields = """
        name, email, phone, skills, education, experience
        """

        schema_rules = """
        name, email, phone, education and experience must each be
        objects containing value, evidence and confidence.
        skills must be a list of those objects.
        """

    elif choice == "2":
        input_file = "invoice.txt"
        output_file = "invoice_output.json"
        result_model = InvoiceResult
        document_type = "invoice"

        fields = """
        invoice_number, date, seller, buyer, items, tax,
        total_amount and currency
        """

        schema_rules = """
        invoice_number, date, seller, buyer, tax and total_amount
        must each be objects containing value, evidence and confidence.
        items must be a list of objects containing description,
        quantity and unit_price. Each of those fields must contain
        value, evidence and confidence.
        currency must be a string.
        """

    else:
        print("Invalid choice. Please choose 1 or 2.")
        return

    path = Path(input_file)

    if not path.exists():
        print(f"{input_file} file not found!")
        return

    source_text = path.read_text(encoding="utf-8")

    prompt = f"""
    Extract the following information from this {document_type}:
    {fields}

    Output rules:
    {schema_rules}

    Each extracted field must include:
    - value: extracted information as a string
    - evidence: exact supporting text copied from the source
    - confidence: number from 0 to 1

    Do not invent information.
    For missing fields, use value "Not provided",
    evidence "" and confidence 0.
    For unknown currency, use "Unknown".
    Return only valid JSON.

    Source document:
    {source_text}
    """

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )

        if not response.text:
            print("Gemini returned an empty response.")
            return

        extracted_data = json.loads(response.text)

        # Validate structure and types
        validated = result_model.model_validate(extracted_data)
        result_data = validated.model_dump()

        # Check evidence against original document
        issues = verify_evidence(result_data, source_text)

        print(f"\n{document_type.upper()} EXTRACTION RESULT")
        print("-" * 35)
        print(json.dumps(result_data, indent=2))

        print("\nEVIDENCE VERIFICATION")
        print("-" * 35)

        if issues:
            for issue in issues:
                print("WARNING:", issue)
        else:
            print("All evidence checks passed.")

        print("\nPYDANTIC VALIDATION PASSED")

        output = {
            "document_type": document_type,
            "extracted_data": result_data,
            "evidence_issues": issues
        }

        Path(output_file).write_text(
            json.dumps(output, indent=2),
            encoding="utf-8"
        )

        print(f"Result saved to {output_file}")

    except json.JSONDecodeError:
        print("Error: Gemini returned invalid JSON.")

    except ValidationError as e:
        print("Data validation failed:")
        print(e)

    except Exception as e:
        print("Extraction failed:")
        print(e)


if __name__ == "__main__":
    main()
