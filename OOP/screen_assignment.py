# Student Name: Xiangyu Xiao
# Student Number: 2034863

# Create an object called Screen
class Screen(object):

    # Define some variables
    l = 20
    elements = [] * l
    MINBRIGHTNESSLEVEL = 0
    MAXBRIGHTNESSLEVEL = 100
    number_of_screen = 0

    # Constructor
    def __init__(self, brightnessLevel, element):
        # self._brightnessLevel = None
        Screen.elements.append(element)
        self.element = Screen.elements
        Screen.number_of_screen += 1
        self.id = Screen.number_of_screen
        self._brightnessLevel = brightnessLevel

    # Use the property method to make sure the brightnessLevel won't be changed when we define it a new value in the main function
    @property
    def getBrightnessLevel(self):
        return self._brightnessLevel

    # Establish a setter for the property
    @getBrightnessLevel.setter
    def setBrightnessLevel(self, brightnessLevel):
        if not isinstance(brightnessLevel, int):
            raise ValueError('The input brightness level must be an integer!')
        if brightnessLevel < Screen.MINBRIGHTNESSLEVEL or brightnessLevel > Screen.MAXBRIGHTNESSLEVEL:
            raise ValueError('The input brightness level must be in the range of 0 and 100 (bounds included)!')
        self._brightnessLevel = brightnessLevel

    def addElements(self, element):
        if len(self.elements) < 20:
            self.elements.append(element)
        else:
            print('The list is full, you cannot add any element!')

    def delElements(self, element):
        if len(self.elements) > 0:
            self.elements.remove(element)
        else:
            print('The list is empty, you cannot delete any element!')

    def toString(self):
        return s.id, s.getBrightnessLevel, s.elements

if __name__ == "__main__":
    # Instantiate an object
    s = Screen(10, 'Daniel')
    print(s.toString())

    # Test the delElements function when there is no element in the list
    s.addElements('Maxim')
    print(s.toString())
    s.delElements('Maxim')
    print(s.toString())
    s.delElements('Daniel')
    print(s.toString())
    s.delElements('Daniel')
    print(s.toString())

    # Test the addElements function when there are full of elements in the list
    i = 1
    for i in range (1, 21):
        s.addElements('Maxim')
        i = i + 1
    print(s.toString())
    s.addElements('Daniel')

    # Test whether the brightness level would change when we define it a new value
    s.brightnessLevel = -1
    print(s.getBrightnessLevel)

    # Test the function if we input a illegal number as the brightness level
    s.setBrightnessLevel = 1.1
    print(s.getBrightnessLevel)
    # s.setBrightnessLevel = -1
    # print(s.getBrightnessLevel)
    # s.setBrightnessLevel = 101
    # print(s.getBrightnessLevel)
