import subprocess

def download( target_url: str, token: str ):
    cmd = f'wget -e robots=off -m -np -R .html,.tmp -nH --no-check-certificate --cut-dirs=4 "{target_url}" --header "Authorization: Bearer {token}"'
    print(f"Using download command: '{cmd}'")
    proc = subprocess.Popen( cmd, shell=True, bufsize=-1 )
    print(f"Downloading url {target_url}")
    proc.wait()
    print(f"  **FINISHED Downloading url {target_url} **** ")

product= "MCDWD_L3_F2_NRT"
tile="h21v10"
day = 203

url = f"https://nrt4.modaps.eosdis.nasa.gov/api/v2/content/archives/{product}/allData/61//Recent/{product}.A2022{day}.{tile}.061.tif"
token = "dHBtYXh3ZWw6ZEdodmJXRnpMbTFoZUhkbGJHeEFibUZ6WVM1bmIzWT06MTYzMTgxNTYzOTo2ZjRjMTZjMmRiMjE1ZGZhNGIzNGYxNGQxZmQ1YWFhODg3ZWIwOTg1"

download( url, token )
