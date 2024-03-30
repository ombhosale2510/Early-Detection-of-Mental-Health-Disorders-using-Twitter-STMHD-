import zipfile
import requests


url = 'https://zenodo.org/records/6409736/files/ptsd.zip'
server_path = '/data3/Zeinab/ Multiple Mental Health/ptsd.zip'

response = requests.get(url)

if response.status_code == 200:
    # Save the content of the response to a local file
    with open(server_path, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {url} and saved it to {server_path}")
else:
    print(f"Failed to download {url}. Status code: {response.status_code}")


extract_path = 'path/Extracted_ZIPs/'


with zipfile.ZipFile(server_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(f"Extracted the contents of {server_path} to {extract_path}")


