# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:56:24 2024

@author: user
"""

import pdfplumber
import re
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
#from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field

llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4o-mini"  
)

class WarningLetterInfo(BaseModel):
    warning_letter_number: Optional[str] = Field(description="Warning Letter identifier (e.g., 320-24-63)")
    marcs_cms_number: Optional[str] = Field(description="MARCS-CMS tracking number")
    date_issued: Optional[str] = Field(description="Date of warning letter issuance")
    recipient_name: Optional[str] = Field(description="Name of recipient")
    recipient_title: Optional[str] = Field(description="Title of recipient")
    company_name: Optional[str] = Field(description="Company name")
    company_address: Optional[str] = Field(description="Company address")
    fei_number: Optional[str] = Field(description="(FEI) FDA Establishment Identifier number e.g FEI 3008449456")
    inspection_date: Optional[str] = Field(description="Date of inspection or records request")
    response_date: Optional[str] = Field(description="Date of company's response if mentioned")
    issuing_office: Optional[str] = Field(description="FDA office issuing the warning letter")

class violations_model(BaseModel):
    violations: List[dict] = Field(
        description="List of violations, each containing a heading and subsections",
        example=[
            {
                "violation_heading": "Title summarizing the violation section",
                "subsections": [
                    {
                        "subsection_header": "Subsection A Header",
                        "subsection_summary": "Detail-rich summary of the specific issues in this subsection",
                        "regulatory_citations": [
                            "CITATION_1",
                            "CITATION_2",
                            "...",
                            "AUGMENTED_CITATION_IF_APPLICABLE"
                        ],
                        "remediation_requirements": [
                            "Detailed remediation step 1",
                            "Detailed remediation step 2",
                            "..."
                        ]
                    },
                    {
                        "subsection_header": "Subsection B Header",
                        "subsection_summary": "Detail-rich summary of the specific issues in this subsection",
                        "regulatory_citations": [
                            "CITATION_1",
                            "CITATION_2",
                            "...",
                            "AUGMENTED_CITATION_IF_APPLICABLE"
                        ],
                        "remediation_requirements": [
                            "Detailed remediation step 1",
                            "Detailed remediation step 2",
                            "..."
                        ]
                    }
                ]
            }
        ]
    )

class summarizing_model(BaseModel):
    summary: Optional[str] = Field(description="High level summarization of inspection summary content")

letter_info_parser = JsonOutputParser(pydantic_object=WarningLetterInfo)
violations_parser = JsonOutputParser(pydantic_object=violations_model)
summary_parser = JsonOutputParser(pydantic_object=summarizing_model)

def extract_warning_letter_content_new(pdf_path):
    warning_letter_info = ""
    inspection_summary = ""
    violations = ""
    conclusion = ""
    response_instructions = ""

    with pdfplumber.open(pdf_path) as pdf:
        pages_text = [page.extract_text() for page in pdf.pages]

        # Warning Letter Info
        full_text_page_1 = pages_text[0]
        warning_letter_info_match = re.search(
            r"(.*?This warning letter summarizes)", full_text_page_1, re.DOTALL | re.IGNORECASE
        )
        if warning_letter_info_match:
            warning_letter_info = warning_letter_info_match.group(1).strip()

        # Inspection Summary
        inspection_summary_match = re.search(
            r"This warning letter summarizes.*", full_text_page_1, re.DOTALL
        )
        if inspection_summary_match:
            inspection_summary = inspection_summary_match.group(0).strip()

        # Violations (from the beginning of page 2 until "Conclusion" or "CGMP Consultant Recommended")
        violations_start = pages_text[1].strip() if len(pages_text) > 1 else ""
        violations_content = ""
        for page_text in pages_text[1:]:
            violations_content += page_text

        violations_match = re.search(
            r"(.*?)((Conclusion)|(CGMP Consultant Recommended))", violations_content, re.DOTALL | re.IGNORECASE
        )
        if violations_match:
            violations = violations_match.group(1).strip()
        else:
            violations = violations_content.strip()  # Default to all content if no end marker is found

        # Conclusion (if "Conclusion" exists as heading)
        for page_text in pages_text[1:]:  # Look through all pages after page 1
            conclusion_match = re.search(
                r"Conclusion(.*?)(This letter notifies you of our findings)", page_text, re.DOTALL | re.IGNORECASE
            )
            if conclusion_match:
                conclusion = conclusion_match.group(1).strip()
                break

        # Response Instructions (everything after "This letter notifies you of our findings")
        for page_text in pages_text[1:]:
            response_instructions_match = re.search(
                r"This letter notifies you of our findings(.*)", page_text, re.DOTALL | re.IGNORECASE
            )
            if response_instructions_match:
                response_instructions = response_instructions_match.group(1).strip()
                break

    return {
        "warning_letter_info": warning_letter_info,
        "inspection_summary": inspection_summary,
        "violations": violations,
        "conclusion": conclusion,
        "response_instructions": response_instructions,
    }

def process_with_llm(section, content, prompt):
    if not content.strip():
        return f"No content found for {section}."
    
    # Generate structured information using LLM
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"{prompt}\n\nContent:\n{content}",
        max_tokens=500,
        temperature=0.7
    )
    return response['choices'][0]['text'].strip()



# Define your prompts for each section
letter_info_prompt_template = """
Extract all available warning letter and inspection information from this section. Only include information that is explicitly stated.

###Example Input:
WARNING LETTER
Azurity Pharmaceuticals, Inc.MARCS-CMS 656489 — SEPTEMBER 20, 2024

Delivery Method:
Via Email
Product:
Drugs
Recipient:
Richard Blackburn
Chief Executive Officer
Azurity Pharmaceuticals, Inc.
8 Cabot Road
Woburn, MA 01801
United States
richard.blackburn@azurity.com
Issuing Office:
Center for Drug Evaluation and Research (CDER)
United States

WARNING LETTER
September 20, 2024
RE: 656489
Dear Mr. Blackburn:
The U.S. Food and Drug Administration inspected your drug manufacturing facility, Azurity Pharmaceuticals, Inc. FEI 3003395329, at 841 Woburn Street, Wilmington, MA, from February 2, 2023, to March 2, 2023. Based on our inspection and subsequent review of your firm's website, we found violations of the Federal Food, Drug, and Cosmetic Act (the FD&C Act).

Example Output:
{{
  "warning_letter_number": "656489",
  "marcs_cms_number": "MARCS-CMS 656489",
  "date_issued": "September 20, 2024",
  "recipient_name": "Richard Blackburn",
  "recipient_title": "Chief Executive Officer",
  "company_name": "Azurity Pharmaceuticals, Inc.",
  "company_address": "8 Cabot Road, Woburn, MA 01801, United States",
  "fei_number": "FEI 3003395329",
  "inspection_date": "February 2, 2023 - March 2, 2023",
  "issuing_office": "Center for Drug Evaluation and Research (CDER)"
}}

Now extract information from this text:
{text}

{format_instructions}
"""

summarize_prompt = """
Summarize the given text:
{text}

Return the output as valid JSON matching the following format:
{format_instructions}
"""
response_instructions_prompt = """
Provide the response instructions point wise to be carried out for the text provided include corresponding emailid and FEI numbers:
{text}

Return the output as valid JSON matching the following format:
{format_instructions}
"""

violations_prompt_template = """
You are the Director of 'Quality Assurance & Regulatory Affairs' in your organisation and you are an expert on the Title 21, Code of Federal Regulations.
You have received a warning letter, issued by the Center for Drug Evaluation & Research, primarily dealing with violations of the 'Current Good Manufacturing Practices',
with a number of violations that have detailed below.

In order for you to start addressing these violations, you need to be able to break them down in to the following format:

Analyse the context given below, and categorise at a high level, what are the major ways in which the violations below are sectioned out.
Typically, the content below is sectioned anywhere between 2 - 5 sections depending on the nature of the violations. 

Step 1: Categorize and Split up the section 
Step 2: Once the sections have been split, assign a heading 'Title' to each section. (This heading 'Title' will semantically capture the title summary of each of the previously split sections, in less than a sentence, up to a few words)
Step 3: Return the individual 'Title's (without repeating the section content back to me)


Step 4: For each of the section, you are then going to split it up further by drilling down and then split the section into subsections based on specific issues mentioned.
Step 5: For each of the subsection of issues, you are then going to do 2 things:
a) capture all the regulatory citations mentioned in the section pertaining to this specific issue
b) Without hallucinating, as a expert in Regulatory Affairs, what other related 'regulatory citations' can the list prepared in 'Step 5(a)' be augmented with. (Remember, you are an expert who will only give this additional citation as recommendation, if and only if you are sure about the legal application).

The thought process after completing Step #4 & Step #5 would then be:

VIOLATION # 1 - Heading
VIOLATION # 1 - Subsection A
VIOLATION # 1 - Subsection A - Regulatory Citations ['CITATIONS MENTIONED AS A LIST']

VIOLATION # 1 - Heading
VIOLATION # 1 - Subsection B
VIOLATION # 1 - Subsection B - Regulatory Citations ['CITATIONS MENTIONED AS A LIST')]

VIOLATION # 2 - Heading
VIOLATION # 2 - Subsection A
VIOLATION # 2 - Subsection A - Regulatory Citations ['CITATIONS MENTIONED AS A LIST')]

and so on. I hope the pattern is that is expected is clear. 

Step 6: Going back to the initial sections, your organisation's partners expect the information presented to them as a 'detail-rich-summary'. So your responsibility will also be to analyse and consolidate the information to be presented as a 'detail-rich-summary'.

VIOLATION # 1 - Heading
VIOLATION # 1 - Subsection A Header
VIOLATION # 1 - Subsection A Summary
VIOLATION # 1 - Subsection A - Regulatory Citations ['CITATIONS MENTIONED AS A LIST']

and the pattern continues as previously mentioned.

Step 7: Additionally, if the sections identified from the warning letter, mention any remediation requirements that can be mapped to the violations 'Sections',
then let us append the bullet points under each section after the regulatory citations.
The remediation requirements will be clearly mentioned as instructions & recommendations,
whether it is a change, or a review or a guidance document URL that needs to be downloaded and studied,
all of those are instructions by the FDA that are part and parcel of the required actions to
emediate the violations and hence will be considered as 'remediation_requirements'

VIOLATION # 1 - Heading
VIOLATION # 1 - Subsection A Header
VIOLATION # 1 - Subsection A Summary
VIOLATION # 1 - Subsection A - Regulatory Citations ['CITATIONS MENTIONED AS A LIST']
VIOLATION # 1 - This Section's Remediation Requirements

and the pattern continues as previously mentioned.

OUTPUT_STRUCTURE: Your objective is NOT to think out loud, but to carry out the above process on your own and just return the information in the format presented below:

{{
  "violations": [
    {{
      "violation_heading": "Title summarizing the violation section",
      "subsections": [
        {{
          "subsection_header": "Subsection A Header",
          "subsection_summary": "Detail-rich summary of the specific issues in this subsection",
          "regulatory_citations": [
            "CITATION_1",
            "CITATION_2",
            "...",
            "AUGMENTED_CITATION_IF_APPLICABLE"
          ],
          "remediation_requirements": [
            "Detailed remediation step 1",
            "Detailed remediation step 2",
            "..."
          ]
        }},
        {{
          "subsection_header": "Subsection B Header",
          "subsection_summary": "Detail-rich summary of the specific issues in this subsection",
          "regulatory_citations": [
            "CITATION_1",
            "CITATION_2",
            "...",
            "AUGMENTED_CITATION_IF_APPLICABLE"
          ],
          "remediation_requirements": [
            "Detailed remediation step 1",
            "Detailed remediation step 2",
            "..."
          ]
        }}
        // Additional subsections as needed
      ]
    }},
    {{
      "violation_heading": "Title summarizing another violation section",
      "subsections": [
        {{
          "subsection_header": "Subsection A Header",
          "subsection_summary": "Detail-rich summary of the specific issues in this subsection",
          "regulatory_citations": [
            "CITATION_1",
            "CITATION_2",
            "...",
            "AUGMENTED_CITATION_IF_APPLICABLE"
          ],
          "remediation_requirements": [
            "Detailed remediation step 1",
            "Detailed remediation step 2",
            "..."
          ]
        }}
        // Additional subsections as needed
      ]
    }}
    // Additional violations as needed
  ]
}}

------------------------WARNING LETTER BELOW:
{text}
    
{format_instructions}    
"""

letter_info_prompt = ChatPromptTemplate.from_template(
    template=letter_info_prompt_template,
    partial_variables={"format_instructions": letter_info_parser.get_format_instructions()}
)

letter_info_chain = letter_info_prompt | llm | letter_info_parser

violations_info_prompt = ChatPromptTemplate.from_template(
    template=violations_prompt_template,
    partial_variables={"format_instructions": violations_parser.get_format_instructions()}
)

violation_info_chain = violations_info_prompt | llm | violations_parser

summarizing_info_prompt = ChatPromptTemplate.from_template(
    template=summarize_prompt,
    partial_variables={"format_instructions": summary_parser.get_format_instructions()}
)

summarizing_info_chain = summarizing_info_prompt | llm | summary_parser

RI_info_prompt = ChatPromptTemplate.from_template(
    template=response_instructions_prompt,
    partial_variables={"format_instructions": summary_parser.get_format_instructions()}
)

summarizing_info_chain = RI_info_prompt | llm | summary_parser

def extract_letter_info(text: str) -> Dict:
    try:
        return letter_info_chain.invoke({"text": text})
    except Exception as e:
        print(f"Error extracting letter info: {e}")
        return {}

def extract_vioation_info(text: str) -> Dict:
    try:
        return violation_info_chain.invoke({"text": text})
    except Exception as e:
        print(f"Error extracting letter info: {e}")
        return {}

def create_summary(text: str) -> Dict:
    try:
        return summarizing_info_chain.invoke({"text": text})
    except Exception as e:
        print(f"Error extracting letter info: {e}")
        return {}

#"violations": "List the specific violations and associated CFR citations from the following content.",    
prompts = {
    "inspection_summary": "Summarize the inspection findings mentioned in the content.",
    "conclusion" : "Summarize the content provided",
    "response_instructions": "Provide a summary of the response instructions and required actions mentioned in the content."
}

def format_wlinfo(data):
    # Create a formatted dictionary
    formatted_data = {
        'Warning Letter Number': data.get('warning_letter_number', 'N/A'),
        'MARCS CMS Number': data.get('marcs_cms_number', 'N/A'),
        'Date Issued': data.get('date_issued', 'N/A'),
        'Recipient Name': data.get('recipient_name', 'N/A'),
        'Recipient Title': data.get('recipient_title', 'N/A'),
        'Company Name': data.get('company_name', 'N/A'),
        'Company Address': data.get('company_address', 'N/A'),
        'FEI Number': data.get('fei_number', 'N/A'),
        'Inspection Date': data.get('inspection_date', 'N/A'),
        'Issuing Office': data.get('issuing_office', 'N/A')
    }

    # Now create the formatted text to be returned
    formatted_text = ""
    for key, value in formatted_data.items():
        formatted_text += f"**{key}**: {value}\n\n"

    return formatted_text

def format_violations(warning_letter_data):
    """
    Parses and formats warning letter content into markdown-formatted text.
    
    :param warning_letter_data: Dictionary containing warning letter information.
    :return: A markdown-formatted string representing the warning letter content.
    """
    if not warning_letter_data.get("violations"):
        return "**No violations found in the data.**"

    result = []
    for violation in warning_letter_data["violations"]:
        result.append(f"### **{violation['violation_heading']}**\n")  # Heading for the violation
        for subsection in violation.get("subsections", []):
            result.append(f"#### **{subsection['subsection_header']}**\n")  # Subsection header
            result.append(f"**Summary:** {subsection['subsection_summary']}\n")  # Summary
            citations = ', '.join(subsection.get("regulatory_citations", []))
            result.append(f"**Regulatory Citations:** {citations if citations else 'None'}\n")  # Citations
            result.append("**Remediation Steps:**")
            for step in subsection.get("remediation_requirements", []):
                result.append(f"- {step}")
            result.append("")  # Add an extra newline after each subsection
        result.append("")  # Add a newline after each violation

    return "\n".join(result)

def format_summary(summary):
    if not summary:
        return "**Error:** No summary content found."
    
    # Format the content as markdown
    summary_content = summary.get("summary", "")
    formatted_content = f"{summary_content}"
    return formatted_content

def pass_output(pdf):
    # Provide the path to the PDF file
    #pdf_path = "C:/Users/user/Downloads/FDA/warning letters/Allen/MMC Healthcare 1.pdf"

    #Step 1: Extract content
    content = extract_warning_letter_content_new(pdf)
    
    # Step 2: Process Warning letter info using the LLM
    wl_info = extract_letter_info(content["warning_letter_info"])
    formatted_wl_data = format_wlinfo(wl_info)

    # Step 3: Process Inspection summary info using the LLM
    IS_info =  create_summary(content["inspection_summary"])
    formatted_IS_info = format_summary(IS_info)
    
    # Step 4: Process violation info using the LLM
    V_info = extract_vioation_info(content["violations"])
    formatted_violations = format_violations(V_info)
    
    # Step 5: Process Response instructions info using the LLM
    RI_info =  create_summary(content["response_instructions"])
    formatted_RI_info = format_summary(RI_info)
    
    # Step 6: Process violation info using the LLM
    conclusion_info =  create_summary(content["conclusion"])
    formatted_conclusion_info = format_summary(conclusion_info)

    return formatted_wl_data,formatted_IS_info,formatted_violations,formatted_RI_info,formatted_conclusion_info

    #partial_content = {'inspection_summary':content['inspection_summary'], 'conclusion':content['conclusion'], 'response_instructions':content['response_instructions']}
    #processed_data = {}
    #for section, content in partial_content.items():
    #    processed_data[section] = process_with_llm(section, content, prompts[section])
