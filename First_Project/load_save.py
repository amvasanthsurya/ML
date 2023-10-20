import pickle

def save(name,val):
    with open('./saved_file/'+name+'.pkl','wb') as file:
        pickle.dump(val,file)

def load(name):
    with open('./saved_file/'+name+'.pkl','rb') as file:
        var=pickle.load(file)
    return var