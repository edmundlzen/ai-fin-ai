import json


pb_funds_data = json.loads(open("pb_funds_data.json").read())
unit_trusts_data = json.loads(open("unit_trusts.json").read())

new_data = {}

for key in pb_funds_data:
    unit_trust_data = None
    for unit_trust_key in unit_trusts_data:
        if unit_trust_key["fund_name"] == key:
            unit_trust_data = unit_trust_key
            break
    new_data[key] = pb_funds_data[key]
    new_data[key].update(unit_trust_data)

with open("merged_data.json", "w") as f:
    json.dump(new_data, f, indent=4)
