import pandas as pd
import re
import os

# Define file paths
file_path = os.path.join(os.path.dirname(__file__), "source.txt")
output_csv = os.path.join(os.path.dirname(__file__), "base.csv")

# Define flexible patterns to extract key-value pairs
patterns = {
    "Organization": r"Organization[:\s]+(.+)",
    "Phone Numbers": r"Phone Numbers[:\s]+(.+)",
    "Toll-Free": r"Toll-Free[:\s]+(.+)",
    "Fax": r"Fax[:\s]+(.+)",
    "Email": r"Email[:\s]+(.+)",
    "Website": r"Website[:\s]+(.+)",
    "Address": r"Address[:\s]+(.+)",
    "Service Description": r"Service Description[:\s]+(.+)",
    "Eligibility": r"Eligibility / Target Population[:\s]+(.+)",
    "Application": r"Application[:\s]+(.+)",
    "Service Hours": r"Hours[:\s]+(.+)",
    "Languages": r"Languages[:\s]+(.+)",
    "Area Served": r"Area Served[:\s]+(.+)",
}

# Step 1: Read the text file
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Step 2: Extract structured data
entries = []
current_entry = {}
current_key = None  # Track multi-line fields

for line in lines:
    line = line.strip()
    
    # Check for new organization entry
    if line.startswith("Organization:"):
        if current_entry:
            entries.append(current_entry)  # Store previous entry before creating a new one
        current_entry = {"Organization": line.replace("Organization:", "").strip()}
        current_key = None  # Reset multi-line tracking
    
    # Capture key-value pairs
    else:
        found_match = False
        for key, pattern in patterns.items():
            match = re.search(pattern, line, re.MULTILINE)
            if match:
                current_entry[key] = match.group(1).strip()
                current_key = key  # Track multi-line fields
                found_match = True
                break
        
        # If the current line doesn't match a pattern, assume it's a continuation of the previous key
        if not found_match and current_key:
            current_entry[current_key] += " " + line

# Append the last entry
if current_entry:
    entries.append(current_entry)

# Step 3: Convert structured data into a DataFrame
df = pd.DataFrame(entries)

# Step 4: Convert to a question-answer format for Giskard
structured_data = []
questions = {
    "Organization": "What is the organization name?",
    "Phone Numbers": "What are the contact details?",
    "Service Description": "What is the service description?",
    "Eligibility": "What are the eligibility criteria?",
    "Application": "What are the application details?",
    "Service Hours": "What are the service hours?",
    "Languages": "What languages are supported?",
    "Website": "What is the website URL?"
}

for _, row in df.iterrows():
    for column, q in questions.items():
        if column in row and pd.notna(row[column]):
            structured_data.append({"question": q + f" ({row['Organization']})", "answer": row[column]})

# Convert structured data to DataFrame
kb_df = pd.DataFrame(structured_data)

# Step 5: Save as CSV
kb_df.to_csv(output_csv, index=False)

print(f"\n✅ Knowledge base saved as {output_csv} with {len(kb_df)} entries.")
