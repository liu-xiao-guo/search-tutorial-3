import json
from search import Search
es = Search()
with open('data.json', 'rt') as f:
    documents = json.loads(f.read())
es.insert_documents(documents)