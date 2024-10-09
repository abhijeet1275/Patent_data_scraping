import requests
import bs4
import os

# Create a folder for storing files
folder_name = "Patent_script"
os.makedirs(folder_name, exist_ok=True)

def get_value(url):
    data = {
        "CurrentPage": 1,
        "TotalPages": 42,
        "ConnectionName": "PublicationConnection",
        "QueryString": "SELECT AD.[APPLICATION_NUMBER],AD.[PATENT_NUMBER],[TITLE_OF_INVENTION],CONVERT(VARCHAR(10),[DATE_OF_FILING],103) AS Application_Date FROM [dbo].[PAT_APPLICATION_DETAILS] (Nolock) AS AD WHERE Convert(date,AD.DATE_OF_FILING) BETWEEN '09/01/2024' AND '10/02/2024'",
        "TotalResult": 1049,
        "Title": ""
    }
    
    application_numbers = []
    
    # Loop through pages
    for page in range(1, 43):
        data.update({"Page": page})  # Update the page number in the request data
        
        # Send the POST request
        response = requests.post(url, data=data)
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        
        # Find all buttons for 'ApplicationNumber' on the page
        buttons = soup.find_all('button', {'class': 'btn btn-link', 'name': 'ApplicationNumber'})
        
        # Collect all application numbers from the buttons
        for button in buttons:
            application_number = button.get('value')
            if application_number:
                application_numbers.append(application_number)
        
        # Break if no buttons are found on the page
        if not buttons:
            break

    return application_numbers

# Example usage
url = "https://iprsearch.ipindia.gov.in/PublicSearch/PublicationSearch/PatentSearchResult"
application_numbers = get_value(url)

# Save application numbers to a text file inside the folder
application_numbers_file = os.path.join(folder_name, "application_numbers.txt")
with open(application_numbers_file, "w") as file:
    for number in application_numbers:
        file.write(f"{number}\n")

print(f"Found {len(application_numbers)} application numbers and saved them to {application_numbers_file}")
