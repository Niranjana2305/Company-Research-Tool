from google import genai
from google.genai.types import GenerateContentConfig, GoogleSearch, Tool
import os
from dotenv import load_dotenv
print("***************Company Research Tool****************")
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
tools = [Tool(google_search=GoogleSearch())]
config = GenerateContentConfig(tools=tools)

def ask_gemini(company_name: str) -> dict:
    prompt = (
        f"You are a factual assistant. For the U.S. company named '{company_name}', "
        "provide:\n"
        "- Industry\n"
        "- Employee size (just a number, or 'Not found')\n"
        "- Domain (e.g., example.com, or 'Not found')\n"
        "Respond in this exact format:\n"
        "Industry: ...\n"
        "Employee size: ...\n"
        "Domain: ...\n"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=config
    )
    text = response.text.strip()
    result = {}
    for line in text.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            result[key.strip().lower()] = val.strip()
    return result

def compare_and_suggest(user_val: str, ai_val: str, field_name: str) -> str:
    if not user_val:
        return ai_val if ai_val else "Not found"
    user_val_norm = user_val.lower().replace(" ", "")
    ai_val_norm = ai_val.lower().replace(" ", "") if ai_val else ""
    if user_val_norm != ai_val_norm and ai_val:
        return f"{user_val} (Best Results: {ai_val})"
    return user_val

def main():
    print("Type 'end' to quit.\n")
    while True:
        company = input("Enter company name: ").strip()
        if not company:
            continue
        if company.lower() == "end":
            print("Goodbye!")
            break
        
        industry_in = input("Enter Company Industry (optional, press Enter to skip): ").strip()
        emp_in = input("Enter Employee Size (optional, press Enter to skip): ").strip()
        domain_in = input("Enter Company Website/Domain (optional, press Enter to skip): ").strip()
        
        ai_data = ask_gemini(company)

        industry_out = compare_and_suggest(industry_in, ai_data.get("industry"), "Industry")
        emp_out = compare_and_suggest(emp_in, ai_data.get("employee size"), "Employee Size")
        domain_out = compare_and_suggest(domain_in, ai_data.get("domain"), "Domain")

        print("\n**********Company Profile**********")
        print(f"Company Name: {company}")
        print(f"Industry: {industry_out}")
        print(f"Employee Size: {emp_out}")
        print(f"Domain: {domain_out}")
        print("*************************************")
        print("Enter another company or type 'end' to quit.\n")

if __name__ == "__main__":
    main()
