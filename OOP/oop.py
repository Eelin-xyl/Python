class Screen(object):

    lst = []
    MIN = 0
    MAX = 100

    def __init__(self, blevel, element):

        Screen.lst.append(element)
        self.element = Screen.lst
        self._blevel = blevel
    
    
    def status(self):
        print('brightness level:', s.getblevel, '; elements:', s.lst)

    @property
    def getblevel(self):
        return self._blevel

    @getblevel.setter
    def setblevel(self, blevel):
        if type(blevel) != int:
            raise ValueError('The input must be an integer')

        elif blevel < Screen.MIN or blevel > Screen.MAX:
            raise ValueError('The input must between 0 and 100 (bounds included)')

        else:
            self._blevel = blevel

    def addElements(self, element):
        if len(self.lst) < 20:
            self.lst.append(element)

        else:
            print('The element list is full')

    def delElements(self, element):
        if len(self.lst) == 0:
            print('The element list is empty')

        elif element not in self.lst:
            print('The target is not in list')

        else:
            self.lst.remove(element)

    def clearElements(self):
        self.lst.clear()


if __name__ == "__main__":

    s = Screen(15, 'test')
    s.status()

    s.addElements(1)
    s.status()

    s.delElements(1)
    s.status()

    s.delElements('test')
    s.status()

    s.addElements(2)
    s.status()
    s.delElements(1)
    s.status()

    
    print('full test')
    s.clearElements()
    s.status()
    for i in range(20):
        s.addElements(1)
    s.status()
    s.addElements('test')

    s.setblevel = 2
    print(s.getblevel)
    s.status()

    s.setblevel = 1.1
    print(s.getblevel)