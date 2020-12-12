import easyocr

def activateModel(): 
    reader = easyocr.Reader(['en'],gpu=False) # need to run only once to load model into memory


#%%
