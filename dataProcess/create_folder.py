import os

def create_folder(dic_name):
    if not os.path.exists(dic_name):
       os.makedirs(dic_name)