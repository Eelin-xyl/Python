a = {'a': '1', 'b': '2', 'c': '3'}
for key in a:
    print(key+':'+a[key])
# a:1
# b:2
# c:3

for key in a.keys():
    print(key+':'+a[key])
# a:1
# b:2
# c:3

for value in a.values():
    print(value)
# 1
# 2
# 3

for kv in a.items():
    print(kv)
# ('a', '1')
# ('b', '2')
# ('c', '3')

for key, value in a.items():
    print(key+':'+value)
# a:1
# b:2
# c:3

for (key, value) in a.items():
    print(key+':'+value)
# a:1
# b:2
# c:3
