import textdistance

PATH_RESULT = "predicted.txt"
PATH_CORRECT = "correct.txt"

def calc_metric(pred, gt):
	return textdistance.levenshtein.distance(pred, gt)/len(gt)

f_res = open(PATH_RESULT,"r")
f_cor = open(PATH_CORRECT,"r")

lines_res = [x for x in f_res.read().split('\n') if len(x)>0]
lines_cor = [x for x in f_cor.read().split('\n') if len(x)>0]

f_res.close()
f_cor.close()

terr_med = 0.0
sample_count = 0
for i in range(len(lines_res)):
	if len(lines_res[i].strip())>0 and len(lines_cor[i].strip())>0:
		terr_med += calc_metric(lines_res[i].strip(), lines_cor[i].strip())
		sample_count += 1
		# print("%s || %s"%(lines_res[i].strip(), lines_cor[i].strip()))

terr_med /= sample_count
print('Character Error Rate: %f'%terr_med)
