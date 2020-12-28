class Screen(object):

    lst = []
    MIN = 0
    MAX = 100

    def __init__(self, blevel, element):

        Screen.lst.append(element)
        self.element = Screen.lst
        self._blevel = blevel
    
    
    def status(self):
        prt = 'brightness level:' + str(self._blevel) + '; elements:' + str(self.lst)
        return prt

    @property
    def getblevel(self):
        return self._blevel

    @getblevel.setter
    def setblevel(self, blevel):
        if type(blevel) != int:
            raise ValueError('The input (' + str(blevel) + ') type must be int')

        elif blevel < Screen.MIN or blevel > Screen.MAX:
            raise ValueError('The input (' + str(blevel) + ') must between 0 and 100 (bounds included)')

        else:
            self._blevel = blevel

    def addEle(self, element):
        if len(self.lst) < 20:
            self.lst.append(element)

        else:
            print('The element list is full')

    def delEle(self, element):
        if len(self.lst) == 0:
            print('The element list is empty')

        elif element not in self.lst:
            print('The target is not in list')

        else:
            self.lst.remove(element)

    def clearEle(self):
        self.lst.clear()

if __name__ == "__main__":

    s = Screen(15, 'test')
    print(s.status())

    s.addEle(1)
    print(s.status())

    s.delEle(1)
    print(s.status())

    s.delEle('test')
    print(s.status())

    s.addEle(2)
    print(s.status())
    s.delEle(1)
    print(s.status())

    
    print('full test')
    s.clearEle()
    print(s.status())
    for i in range(20):
        s.addEle(1)
    print(s.status())
    s.addEle('test')

    s.setblevel = 2
    print(s.status())

    s.setblevel = 1.1
    print(s.status())