import os
import requests
import xml.etree.ElementTree as ET


url = "https://raw.githubusercontent.com/venelink/ETPC/master/Corpus/paraphrase_types.xml"
r = requests.get(url)
xml_string1 = r.text
with open("./paraphrase_types.xml", "r") as f:
    xml_string2 = f.read()
    
print(xml_string1 == xml_string2)

