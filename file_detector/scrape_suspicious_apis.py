import requests
from bs4 import BeautifulSoup
import json

URL = "https://malapi.io/"

def scrape_suspicious_apis():
    print("Fetching suspicious APIs from MalAPI...")
    r = requests.get(URL,timeout=10)
    if r.status_code != 200:
        raise Exception(f"Failed to fetch data from {URL} with status code {r.status_code}")
    
    soup = BeautifulSoup(r.text , "html.parser")
    api_elements = soup.select("table tbody tr td:first-child a")
    api_names = [api.text.strip() for api in api_elements]
    
    print(f"Found {len(api_names)} suspicious APIs.")
    with open("data/suspicious_api_list.json", "w") as f:
        json.dump(api_names, f, indent=2)
        
    return api_names

if __name__ == "__main__":
    apis = scrape_suspicious_apis()
    print(apis[:20])