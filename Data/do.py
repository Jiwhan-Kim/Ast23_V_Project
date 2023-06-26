import os

cnt = 0
for i in range(10):
	spec = str(i) + '/'
	routes = os.path.join("./Images/", spec)
	file_names = os.listdir(routes)
	for name in file_names:
		paths = "./Images/" + str(i) + "/"
		src = os.path.join(paths, name)
		ass = str(cnt) + ".jpg"
		dst = os.path.join("./ImageStack", ass)
		os.rename(src, dst)
		cnt += 1

