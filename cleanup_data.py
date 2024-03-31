import json
import os
import re

files_to_data = json.loads(open("files_to_data.json").read())
new_files_to_data = {}
for file, data in files_to_data.items():
    ptr = {}
    starting_year = 2023
    if data["ptr"] != None:
        for i in range(len(data["ptr"][0].split("\n"))):
            ptr[starting_year - i] = data["ptr"][0].split("\n")[i]
    new_files_to_data[file] = {
        "ptr": ptr,
        "management_fee": re.findall(r"(\d+(?:\.\d+)?)%", data["management_fee"][0])[0],
    }

with open("files_to_data.json", "w") as f:
    json.dump(new_files_to_data, f, indent=4)
