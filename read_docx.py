import zipfile
import xml.etree.ElementTree as ET
import sys

def read_docx(path):
    with zipfile.ZipFile(path) as docx:
        tree = ET.parse(docx.open('word/document.xml'))
        root = tree.getroot()
        texts = []
        for paragraph in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
            para_text = ""
            for node in paragraph.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
                if node.text:
                    para_text += node.text
            texts.append(para_text)
    return '\n'.join(texts)

try:
    with open('research_doc_utf8.txt', 'w', encoding='utf-8') as f:
        f.write(read_docx(sys.argv[1]))
except Exception as e:
    print("Error:", e)
