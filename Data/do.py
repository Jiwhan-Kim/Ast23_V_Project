import os

for i in range(1, 271):
	names = str(i) + ".jpg"
	wavs = str(i) + ".wav"
	src = os.path.join("./Sounds/", names)
	dst = os.path.join("./Sounds/", wavs)
	os.rename(src, dst)

