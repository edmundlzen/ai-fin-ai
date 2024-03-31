import json
import os
import re
import camelot

files = os.listdir("pb_phs")

files_to_data = {}

for file in files:
    tables = camelot.read_pdf(
        f"pb_phs/{file}", pages="all", line_scale=100, split_text=True
    )
    ptr_data = None
    management_fee_data = None

    for table in tables:
        if (table.df[0].str.contains("PTR \(time\)", case=False)).any():
            for data in table.data:
                for item in data:
                    if "PTR" in item:
                        data.remove(item)
                        ptr_data = data
                        break

            print(ptr_data[0].split("\n"))
            continue

        if (table.df[0].str.contains("Management fee", case=False)).any():
            for data in table.data:
                for item in data:
                    if "Management fee" in item:
                        data.remove(item)
                        management_fee_data = data
                        break

            print(re.findall(r"(\d+(?:\.\d+)?)%", management_fee_data[0])[0])

    files_to_data[file] = {
        "ptr": ptr_data,
        "management_fee": management_fee_data,
    }

with open("files_to_data.json", "w") as f:
    json.dump(files_to_data, f, indent=4)
