import json

filename = "submission.json"
with open(filename, 'r') as file:
    res = json.load(file)

new = []
for entry in res:
    entry["category_id"] = entry["category_id"][0]
    new.append(entry)

with open(filename, 'w') as file:
    json.dump(new, file)

