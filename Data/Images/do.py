import os

file_names = os.listdir("./3/")

cnt = 1
for name in file_names:
	src = os.path.join("./3/", name)
	dst = str(cnt) + '.jpg'
	dst = os.path.join("./3/", dst)
	os.rename(src, dst)
	cnt += 1
