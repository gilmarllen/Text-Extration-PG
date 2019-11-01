import textdistance
from nltk.translate.bleu_score import corpus_bleu

PATH_RESULT = "predicted.txt"
PATH_CORRECT = "correct.txt"

def calc_metric(pred, gt):
	# print(textdistance.damerau_levenshtein.distance(pred, gt)/len(gt))
	return textdistance.damerau_levenshtein.distance(pred, gt)/len(gt)

def CER(pred, gt, uncased=False):
	terr_med = 0.0
	sample_count = 0
	for i in range(len(pred)):
		if len(pred[i])>0 and len(gt[i])>0:
			x = pred[i]
			y = gt[i]
			if uncased:
				x = x.lower()
				y = y.lower()
			terr_med += calc_metric(x, y)
			sample_count += 1

	return terr_med/sample_count

def WER(pred, gt, uncased=False):
	terr_med = 0.0
	sample_count = 0
	for i in range(len(pred)):
		if len(pred[i])>0 and len(gt[i])>0:
			x = pred[i]
			y = gt[i]
			if uncased:
				x = x.lower()
				y = y.lower()
			x = x.split()
			y = y.split()
			all_tokens = set(x + y)
			map_token_char = {token: chr(i) for i, token in enumerate(all_tokens)}
			str_predicted = ''.join(map_token_char[token] for token in x)
			str_ground_truth = ''.join(map_token_char[token] for token in y)
			terr_med += calc_metric(str_predicted, str_ground_truth)
			sample_count += 1
	return terr_med/sample_count

def BLEU(pred, gt, uncased=False):
	pred_list = [[y for y in x.split(' ') if len(y)>0] for x in pred]
	gt_list = [[[y for y in x.split(' ') if len(y)>0]] for x in gt]

	if uncased:
		pred_list = [[y.lower() for y in x] for x in pred_list]
		gt_list = [[[z.lower() for z in y] for y in x] for x in gt_list]

	if len(pred_list)>0 and len(gt_list)>0 and len(pred_list) == len(gt_list):
		return corpus_bleu(gt_list, pred_list)
	else:
		return 1.0

# I/O
f_res = open(PATH_RESULT,"r")
f_cor = open(PATH_CORRECT,"r")

lines_res = [x.strip() for x in f_res.read().split('\n') if len(x.strip())>0]
lines_cor = [x.strip() for x in f_cor.read().split('\n') if len(x.strip())>0]

f_res.close()
f_cor.close()

print('Character Error Rate (cased): %f'%CER(lines_res, lines_cor, uncased=False))
print('Character Error Rate (uncased): %f'%CER(lines_res, lines_cor, uncased=True))
print('Word Error Rate (cased): %f'%WER(lines_res, lines_cor, uncased=False))
print('Word Error Rate (uncased): %f'%WER(lines_res, lines_cor, uncased=True))
print('BLEU score (cased): %f'%BLEU(lines_res, lines_cor, uncased=False))
print('BLEU score (uncased): %f'%BLEU(lines_res, lines_cor, uncased=True))