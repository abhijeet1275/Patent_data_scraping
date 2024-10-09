import os
import requests
import bs4
import json
import re

# Function to create a folder if it doesn't exist
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

# Function to fetch patent details and save them in a specified folder
def get_patent_details(patent_number, folder_name):
    # Ensure the folder exists
    create_folder(folder_name)
    
    url = "https://iprsearch.ipindia.gov.in/PublicSearch/PublicationSearch/PatentDetails"
    data = {
        "IP": "103.27.12.61:8080",
        "ConnectionName": "PublicationConnection",
        "ApplicationNumber": patent_number
    }
    
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()  # Raise an error for bad responses
    except requests.RequestException as e:
        return None
    
    soup = bs4.BeautifulSoup(response.text, "html.parser")
    
    # Create a dictionary to store patent details
    patent_details = {}

    # Find the main table with patent details
    table = soup.find("table", {"class": "table-striped"})
    
    # Check if the table exists
    if table:
        rows = table.find_all("tr")
        for row in rows:
            columns = row.find_all("td")
            if len(columns) == 2:
                key = columns[0].get_text(strip=True)
                value = columns[1].get_text(strip=True)
                patent_details[key] = value
    
    # Add abstract section
    abstract_tag = soup.find(string="Abstract:")
    if abstract_tag:
        abstract_content = ""
        for sibling in abstract_tag.parent.next_siblings:
            if sibling.name == "strong":  # Stop when encountering the next bold section
                break
            if isinstance(sibling, bs4.element.NavigableString):
                abstract_content += sibling.strip()
            elif sibling.name == "br":
                abstract_content += "\n"  # Handle new lines between paragraphs
        patent_details["Abstract"] = abstract_content.strip()
    
    # Extract complete specification (Description)
    details_section = soup.find(class_='tab-pane fade active in Action PatentDetails')
    if details_section:
        # Get the full text content
        text = details_section.text
        text_lines = text.split('\n')
        
        # Clean up the lines
        cleaned_lines = [line.strip() for line in text_lines if line.strip()]
        
        # Initialize a new list for the current patent
        patent_details["Description"] = []
        
        # Parse and organize the sections
        current_section = ""
        for line in cleaned_lines:
            if re.search(r"\b(DESCRIPTION|FIELD OF THE INVENTION|BACKGROUND OF THE INVENTION|SUMMARY OF THE INVENTION|BRIEF DESCRIPTION OF THE FIGURES|DETAILED DESCRIPTION OF THE INVENTION)\b", line.upper()):
                current_section = line.strip()
                patent_details["Description"].append(f"{current_section}")
            else:
                if current_section:
                    patent_details["Description"].append(f"\t{line}")
    
    # Find additional fields
    field_mapping = {
        "Invention Title": "Invention Title",
        "Publication Number": "Publication Number",
        "Application Number": "Application Number",
        "Filing Date": "Filing Date",
        "Publication Date": "Publication Date"
    }

    for field, field_name in field_mapping.items():
        field_tag = soup.find(string=field)
        if field_tag:
            patent_details[field_name] = field_tag.find_next("td").get_text(strip=True)
    
    # Extract Inventor and Applicant details
    tables = soup.find_all("table", {"class": "table-striped", "border": "1", "cellpadding": "4", "cellspacing": "0", "width": "100%"})

    # Initialize empty lists for inventors and applicants
    patent_details["Inventors"] = []
    patent_details["Applicants"] = []

    for i, table in enumerate(tables):
        rows = table.find_all("tr")
        headers = [header.get_text(strip=True) for header in rows[0].find_all("th")]  # Get the table headers
        
        # Loop through each row except the header
        for row in rows[1:]:
            columns = row.find_all("td")
            if len(columns) < len(headers):
                print(f"Warning: Row has fewer columns than headers for patent {patent_number}")
                continue  # Skip malformed rows
            
            row_data = {headers[j]: columns[j].get_text(strip=True) for j in range(len(columns))}
            # Differentiate between Inventor and Applicant using index
            if i % 2 == 1:  # Even-indexed tables are for Applicants
                patent_details["Applicants"].append(row_data)
            else:
                patent_details["Inventors"].append(row_data)

    # Save the dictionary to a JSON file in the specified folder
    patent_data = {patent_number: patent_details}
    json_filename = os.path.join(folder_name, f"patent_{patent_number}.json")
    
    with open(json_filename, "w") as json_file:
        json.dump(patent_data, json_file, indent=4)
    
    return json_filename

# Modify this code to fetch application numbers from the application_numbers.txt file
create_folder("Patent_dataset")
folder_name = "Patent_script"
list_file = os.path.join(folder_name, "application_numbers.txt")
with open(list_file, "r") as file:
    application_numbers = [line.strip() for line in file]

# Fetch details and save each patent
folder_name = "Patent_dataset"
for application_no in application_numbers:
    json_file_name = get_patent_details(application_no, folder_name)
    print(f"Saved {json_file_name}")

print("Saved Successfully")
