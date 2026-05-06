import codecs
new_reqs = "\ngoogle-generativeai\n"
with codecs.open('reqiurements.txt', 'a', encoding='utf-16le') as f:
    f.write(new_reqs)
print("Added google-generativeai to reqiurements.txt")
