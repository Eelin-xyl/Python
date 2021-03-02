class musictrack(object):
    
    musiclist = [{'id': 123, 'name': 'Little Apple', 'duration': 225}]

    def getname(self):
        lst = []
        for i in musictrack:
            lst.append(i.name)
        
        for j in lst:
            print(j)

    def getid(self):
        lst = []
        for i in musictrack:
            lst.append(i.id)
        
        for j in lst:
            print(j)

    def getduration(self):
        lst = []
        for i in musictrack:
            lst.append(i.duration)
        
        for j in lst:
            print(j)

class playlist(object):

    pllist = []

    def add_track(self, id):
        pllist.append(id)

    def add_track_index(self, id, index):
        pllist[index] = id

    def clear(self):
        pllist.clear()