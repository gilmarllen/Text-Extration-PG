from collections import Counter

f = open("raw.txt", "r")
contents = f.read()
res = Counter(contents)
print(str(res))
f.close()