s = "abcabcbb"
s = list(s)
for i in range(len(s)):
    if s[i] == 'c':
        s[i] = '\b'
print(''.join(s))