
## M3
from gensim.models import KeyedVectors
from scipy import stats
import numpy as np
import statistics as s
vecfile = 'C:/Users/Geoff/Desktop/Research/CCB/Code/Scalar/GoogleNews-vectors-negative300.bin'
vecs = KeyedVectors.load_word2vec_format(vecfile, binary=True)
path = 'C:/Users/Geoff/Desktop/Research/CCB/Code/Scalar/data/FINALDATA/'
files = ["oates_half","oates_full", "half_scale_3", "full_scale", "deMelo_half_OG", "deMelo_half_EXT", "deMelo_full_EXT", "deMelo_full_OG"]

def ise1(gold):
	inda = ""
	indb = ""
	mi = -100
	for i in gold:
		for j in gold:
			m = vecs.similarity(i,j)
			if mi == -100 or m < mi:
				mi = m
				inda = i
				indb = j
	return [inda, indb]



def ise2(gold, ex):
	mi = -100
	ind = ""
	for g in gold:
		m = vecs.similarity(g, ex)
		if mi == -100 or m < mi:
			mi = m
			ind = g
	return ind

def check_vocab(gold):
	for g in gold:
		if g not in vecs.vocab:
			return False
	return True




for f in files:
	for e in range(3):
		with open(path +f + ".txt", "r") as file:
	#every scale in file
			plus = 0
			minus = 0
			spearman = []
			gold_mega_scale = []
			predict_mega_scale = []
			for line in file:
				plus += 1
				l = line.split()

				if check_vocab(l):

					if e == 0:
						ab= ise1(l)
						a = ab[0]
						b = ab[1]
						gold = l.copy()
						gold.remove(a)
						gold.remove(b)
						predict = []
						al = []
						al_d = {}
						bl = []
						bl_d = {}
						for g in gold:
							v1 = vecs.similarity(g, a)
							v2 = vecs.similarity(g, b)
							if v1>v2:
								al.append(v1)
								if v1 in al_d:
									al_d[v1].append(g)
								else:
									al_d[v1] = [g]
							else:
								bl.append(v2)
								if v2 in bl_d:
									bl_d[v2].append(g)
								else:
									bl_d[v2] = [g]
						al.sort(reverse=True)
						bl.sort()
						predict.append(a)
						for x in al:
							for y in al_d[x]:
								predict.append(y)
						for x in bl:
							for y in bl_d[x]:
								predict.append(y)
						predict.append(b)
						if(len(l) == len(predict)):

							spear1 = stats.spearmanr(l, predict)
							predict.reverse()
							spear2 = stats.spearmanr(l, predict)
							if(spear2[0] > spear1[0]):
								predict_mega_scale += predict
							else:
								predict.reverse()
								predict_mega_scale += predict
							gold_mega_scale += l

					if e == 1:
						b = ise2(l[1:], l[0])
						gold = l[1:].copy()
						gold.remove(b)
						a = l[0]
						predict = []
						al = []
						al_d = {}
						bl = []
						bl_d = {}
						for g in gold:
							v1 = vecs.similarity(g, a)
							v2 = vecs.similarity(g, b)
							if v1>v2:
								al.append(v1)
								if v1 in al_d:
									al_d[v1].append(g)
								else:
									al_d[v1] = [g]
							else:
								bl.append(v2)
								if v2 in bl_d:
									bl_d[v2].append(g)
								else:
									bl_d[v2] = [g]
						al.sort(reverse=True)
						bl.sort()
						for x in al:
							for y in al_d[x]:
								predict.append(y)
						for x in bl:
							for y in bl_d[x]:
								predict.append(y)
						predict.append(b)

						if(len(l[1:]) == len(predict)):
							predict_mega_scale += predict
							gold_mega_scale += l[1:]


							# spear = stats.spearmanr(l[1:], predict)
		#every word in scale
					if e == 2:
						a = l[0]
						b = l[len(l)-1]
						gold = l[1:len(l)-1].copy()
						if len(gold) > 1:
							predict = []
							al = []
							al_d = {}
							bl = []
							bl_d = {}
							for g in gold:
								v1 = vecs.similarity(g, a)
								v2 = vecs.similarity(g, b)
								if v1>v2:
									al.append(v1)
									if v1 in al_d:
										al_d[v1].append(g)
									else:
										al_d[v1] = [g]
								else:
									bl.append(v2)
									if v2 in bl_d:
										bl_d[v2].append(g)
									else:
										bl_d[v2] = [g]
							al.sort(reverse=True)
							bl.sort()
							for x in al:
								for y in al_d[x]:
									predict.append(y)
							for x in bl:
								for y in bl_d[x]:
									predict.append(y)
							if(len(gold) == len(predict)):
								predict_mega_scale += predict
								gold_mega_scale += l[1:len(l)-1]

				else:
					minus += 1
				
			print(f + "E" + str(e) + "M3")
			if(len(gold_mega_scale) != len(predict_mega_scale)):
				print(len(gold_mega_scale))
				print(len(predict_mega_scale))
			spearman_t = stats.spearmanr(gold_mega_scale, predict_mega_scale)
			print(spearman_t[0])					