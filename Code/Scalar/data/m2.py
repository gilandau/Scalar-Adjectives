## M2
from gensim.models import KeyedVectors
from scipy import stats
import numpy as np
import statistics as s
vecfile = 'C:/Users/Geoff/Desktop/Research/CCB/Code/Scalar/GoogleNews-vectors-negative300.bin'
vecs = KeyedVectors.load_word2vec_format(vecfile, binary=True)
path = 'C:/Users/Geoff/Desktop/Research/CCB/Code/Scalar/data/FINALDATA/'
files = ["oates_half", "oates_full", "half_scale_3", "full_scale", "deMelo_half_OG", "deMelo_half_EXT", "deMelo_full_EXT", "deMelo_full_OG"]

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

						predict = [a]
						al = []
						bl = []

						current_a = a
						current_b = b


						for g in range(len(gold)):
							ad_arr = vecs.distances(vecs.get_vector(current_a), gold)
							bd_arr = vecs.distances(vecs.get_vector(current_b), gold)
							min_ina = np.argmin(ad_arr)
							min_inb = np.argmin(bd_arr)
							if(min(ad_arr) < min(bd_arr)):
								al.append(gold[min_ina])
								current_a = gold[min_ina]

								del gold[min_ina]

							else:
								bl.append(gold[min_inb])
								current_b = gold[min_inb]
								del gold[min_inb]
						bl.reverse()
						predict += al
						predict +=bl
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
						portion = len(gold)+1
						predict = []

						al = []
						bl = []

						current_a = a
						current_b = b


						for g in range(len(gold)):
							ad_arr = vecs.distances(vecs.get_vector(current_a), gold)
							bd_arr = vecs.distances(vecs.get_vector(current_b), gold)
							min_ina = np.argmin(ad_arr)
							min_inb = np.argmin(bd_arr)
							if(min(ad_arr) < min(bd_arr)):
								al.append(gold[min_ina])
								current_a = gold[min_ina]

								del gold[min_ina]

							else:
								bl.append(gold[min_inb])
								current_b = gold[min_inb]
								del gold[min_inb]
						bl.reverse()
						predict += al
						predict +=bl
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
						predict = []

						if len(gold) > 1:
							al = []
							bl = []

							current_a = a
							current_b = b


							for g in range(len(gold)):
								ad_arr = vecs.distances(vecs.get_vector(current_a), gold)
								bd_arr = vecs.distances(vecs.get_vector(current_b), gold)
								min_ina = np.argmin(ad_arr)
								min_inb = np.argmin(bd_arr)
								if(min(ad_arr) < min(bd_arr)):
									al.append(gold[min_ina])
									current_a = gold[min_ina]

									del gold[min_ina]

								else:
									bl.append(gold[min_inb])
									current_b = gold[min_inb]
									del gold[min_inb]
							bl.reverse()
							predict += al
							predict +=bl
							if(len(l[1:len(l)-1]) == len(predict)):
								predict_mega_scale += predict
								gold_mega_scale += l[1:len(l)-1]

				else:
					minus += 1
				
			print(f + "E" + str(e) + "M1")
			if(len(gold_mega_scale) != len(predict_mega_scale)):
				print(len(gold_mega_scale))
				print(len(predict_mega_scale))
			spearman_t = stats.spearmanr(gold_mega_scale, predict_mega_scale)
			print(spearman_t[0])