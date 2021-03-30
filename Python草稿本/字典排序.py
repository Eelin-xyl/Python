dic = {3: 2, 2: 1, 4: 0}

for i in sorted(dic):
    print((i, dic[i]), end=" ")

print(dic)
s = sorted(dic.items(), key=lambda i: i[1])
print(type(s[0]))
print(dic)
# k = []
# for i in dic:
#     k.append(i)
# k.sort()

# st = {}
# for j in k:
#     st[j] = dic[j]

# print(st)

# print(k)
# for i, k in enumerate(k):
#     print(i, k)
# print(k)
