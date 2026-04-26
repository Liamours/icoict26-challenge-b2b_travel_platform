import fitz
doc = fitz.open(r'C:\Users\lulay\Downloads\icoict26_challenge.pdf')
out = []
out.append(f'Pages: {len(doc)}')
for i in range(len(doc)):
    out.append(f'\n=== PAGE {i+1} ===')
    out.append(doc[i].get_text()[:1500])
with open(r'C:\Users\lulay\Desktop\icoict26-challenge\scripts\pdf_content.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('done')
