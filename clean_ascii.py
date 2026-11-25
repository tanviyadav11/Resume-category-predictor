import codecs

with codecs.open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all non-ASCII characters that break latin-1 REST transport
content = content.replace('—', '-')
content = content.replace('🔹', '*')
content = content.replace('→', '->')
content = content.replace('–', '-')
content = content.replace('✅', '*')
content = content.replace('❌', '*')
content = content.replace('💪', '*')
content = content.replace('⚠️', '*')
content = content.replace('🔴', '*')
content = content.replace('🟢', '*')
content = content.replace('🚀', '*')
content = content.replace('📥', '*')
content = content.replace('📊', '*')
content = content.replace('🤖', '*')
content = content.replace('📄', '*')
content = content.replace('🔍', '*')
content = content.replace('📝', '*')

# As a final catch-all, replace any remaining non-ascii characters just in case
# We only do this for the string literal part, but it's easier to just encode to ascii and decode
# Actually, let's just write the replaced content back.

with codecs.open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Cleaned non-ASCII characters from app.py")
