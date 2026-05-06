import codecs

# Add requirements to the end of the file
new_reqs = "\nstreamlit\npython-docx\nPyPDF2\n"

# Open as utf-16le and append
with codecs.open('reqiurements.txt', 'a', encoding='utf-16le') as f:
    f.write(new_reqs)
print("Added missing packages to reqiurements.txt")
