import requests

caption=input("enter the caption here:" )
url = "https://api.edenai.run/v2/text/generation"

payload = {
    "response_as_dict": True,
    "attributes_as_list": False,
    "show_original_response": False,
    "temperature": 0,
    "max_tokens": 1000,
    "providers": "openai",
    "text": "Generate 20 Hashatgs for this caption:"+caption
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiODlhZTdjYjctNjFiYi00MzZiLTliNzgtYTIyZjI3ZGYyNjQwIiwidHlwZSI6ImFwaV90b2tlbiJ9.A3DAnEg-gv2YvCZhrM_GEnZrXbKY8oXC7ma8Rmy6RNk"
}

response = requests.post(url, json=payload, headers=headers)

if response.status_code == 200:
    response_json = response.json()
    if 'openai' in response_json:
        if 'generated_text' in response_json['openai']:
            generated_text = response_json['openai']['generated_text'].strip()
            print(generated_text)
        else:
            print("generated_text not found in response.")
    else:
        print("openai not found in response.")
else:
    print("Error:", response.status_code, response.text)