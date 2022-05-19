# -*- coding: utf-8 -*-
"""
Created on Thu May 19 19:00:39 2022

@author: Machine
"""
import os
import json
import csv

#CONVERTING FROM JSON TO CSV
dirname = "C:\\Users\\Machine\\Downloads\\autoriaNumberplateOcrRu-2021-09-01\\autoriaNumberplateOcrRu-2021-09-01\\train\\ann\\"

files = os.listdir(dirname)
cols = ['filename', 'words'] 
with open('/content/number_dataset.csv', 'a', encoding='utf-8') as f2:
      wr = csv.DictWriter(f2, fieldnames = cols) 
      wr.writeheader() 
       
      f2.close()
row = dict()
for file_ in files:
  file_path = os.path.join(dirname, file_)
  if file_.endswith(".json"):
    with open(file_path, 'r', encoding='utf-8') as f: 
      text_ = json.load(f) 
      f.close()

  
    row['filename'] = str(text_.get('name')) + '.png'
    row['words'] = str(text_.get('description'))
    print(row)
    with open('/content/number_dataset.csv', 'a', encoding='utf-8') as f2:
      wr = csv.DictWriter(f2, fieldnames = cols) 
      wr.writerow(row) 
      f2.close()
    row.clear()