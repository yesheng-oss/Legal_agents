import json

with open('data/raw/train.json', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

print(f'Total records: {len(lines)}')
first = json.loads(lines[0])
print(f'Keys: {list(first.keys())}')
print(f'fact preview: {first["fact"][:200]}')
print(f'accusations: {first["meta"]["accusation"]}')
print(f'relevant articles: {first["meta"]["relevant_articles"]}')
