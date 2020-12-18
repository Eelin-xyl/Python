class ComplexNumber:

    # I = ComplexNumber(0, 1)

    def __ini__(self, real, imaginary):
        self._real = real
        self._imaginary = imaginary

    def __str__(self):
        return self._real + ' + ' +  self._imaginary + 'i'

    def __eq__(self, other):
        return self._real == other._real