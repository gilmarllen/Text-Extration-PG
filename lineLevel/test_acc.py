import textdistance

PATH_RESULT = "predicted.txt"
PATH_CORRECT = "correct.txt"

def calc_metric(x, y):
	return 1.0 - textdistance.levenshtein.normalized_distance(x, y)

f_res = open(PATH_RESULT,"r")
f_cor = open(PATH_CORRECT,"r")

lines_res = f_res.read().split('\n')
lines_cor = f_cor.read().split('\n')

f_res.close()
f_cor.close()

# print(len(lines_res))
# print(len(lines_cor))

acc_med = 0.0
for i in range(len(lines_res)):
	acc_med += calc_metric(lines_res[i].strip(), lines_cor[i].strip())
	# print("%s || %s"%(lines_res[i].strip(), lines_cor[i].strip()))

acc_med /= len(lines_res)
print('Acuraccy (char level): %f'%acc_med)