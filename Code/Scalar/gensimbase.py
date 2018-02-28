from gensim.models import KeyedVectors
from scipy import stats
import numpy as np
vecfile = 'GoogleNews-vectors-negative300.bin'
vecs = KeyedVectors.load_word2vec_format(vecfile, binary=True)
#
# What is the dimensionality of these word embeddings? Provide an integer answer.
#1 x 300?


# What are the top-5 most similar words to picnic (not including picnic itself)? (Use the function gensim.models.KeyedVectors.wv.most_similar)

# ('picnics', 0.740087628364563), ('picnic_lunch', 0.7213740348815918), ('Picnic', 0.7005340456962585), ('potluck_picnic', 0.6683276891708374), ('picnic_supper', 0.6518914103507996)

# According to the word embeddings, which of these words is not like the others? ['tissue', 'papyrus', 'manila', 'newsprint', 'parchment', 'gazette'] (Use the function gensim.models.KeyedVectors.wv.doesnt_match)

# Tissue

# Solve the following analogy: “leg” is to “jump” as X is to “throw”. (Use the function gensim.models.KeyedVectors.wv.most_similar with positive and negative arguments.)

# vecs.most_similar(positive=["leg", "throw"], negative=["jump"] )
# ('forearm', 0.4829462766647339)

# with open("closest_vecs.txt", "w",encoding="utf-8") as big_f:
#     word_list = ["good", "best" , "bad", "worst", "fast", "fastest", "slow", "slowest", "happy", "happiest", "sad", "saddest", "angry", "angriest"]
#     for word in word_list:
#         a = vecs.get_vector(word)
#         big_f.write(str(word) + "----------\n")
#         m = vecs.most_similar(positive=[a], topn=6)
#         for pair in m:
#             big_f.write(str(pair) + "\n")
#         big_f.write("+++++++++++++++++++++++++++++++++\n")
# big_f.close()
#
#
# with open("quartile_vecs.txt", "w",encoding="utf-8") as big_f:
#     first_x = ["furious","furious","terrible","cold","ugly","black","dark","sad"]
#     last_x =  ["happy","calm","terrific","hot","gorgeous","white","light","happy"]
#     for i in range(len(first_x)):
#         a = vecs.get_vector(first_x[i])
#         b = vecs.get_vector(last_x[i])
#         c2 = b / 2 + a / 2
#         c1 = (3*b / 4) + a / 4
#         c3 = b / 4 + (3 * a / 4)
#         m1 = vecs.most_similar(positive=[c1], topn=10)
#         m2 = vecs.most_similar(positive=[c2], topn=10)
#         m3 = vecs.most_similar(positive=[c3], topn=10)
#         m4 = vecs.most_similar(positive=[b], topn=10)
#         m0 = vecs.most_similar(positive=[a], topn=10)
#
#         big_f.write(first_x[i] + "----------\n")
#         for m in m0:
#             big_f.write(str(m) + "\n")
#         big_f.write("first-----------------\n")
#         for m in m1:
#             big_f.write(str(m) + "\n")
#         big_f.write("second-----------------\n")
#         for m in m2:
#             big_f.write(str(m) + "\n")
#         big_f.write("third-----------------\n")
#         for m in m3:
#             big_f.write(str(m) + "\n")
#         big_f.write(last_x[i] + "-----------------\n")
#         for m in m4:
#             big_f.write(str(m) + "\n")
#         big_f.write("+++++++++++++++++++++++++++++++++\n")
#
# big_f.close()

SPEARMAN_BUFFER = []

def calc_spearman(gold, test):
    s = stats.spearmanr(gold, test)
    if s[0] < 1.0:
        print(gold)
        print(test)
    return s


# Generate distance matrix
# Possibly NEGATIVE other values from search
def gen_matrix_twosided(scale):
    matrix = np.zeros((len(scale, len(scale))))
    for i in range(len(scale)):
        for j in range(len(scale)):
            matrix[i][j] = vecs.similarity(scale[i], scale[j])

    return matrix
# xlist = [min, max]
def two_sided(x_list, scale):
    oscale = scale
    final_scale = np.zeros(len(oscale), dtype=object)
    # sim_scale = gen_matrix_twosided(scale)
    x_scale = np.zeros((len(oscale), 2))
    for s in range(len(oscale)):
        for i in range(2):
            x_scale[s][i] = vecs.similarity(x_list[i], oscale[s])
    i1 = 0
    i2 = len(oscale) - 1
    while len(x_scale) != 0:
        close_to_x = np.argmax(x_scale, axis=0)
        # Pick best distance
        x1_d = x_scale[close_to_x[0]][0]
        x2_d = x_scale[close_to_x[1]][1]
        if x1_d > x2_d:
            final_scale[i1] = (oscale[close_to_x[0]], x_scale[close_to_x[0]][0], x_scale[close_to_x[0]][1])
            x_scale = np.delete(x_scale,close_to_x[0],0)
            del oscale[close_to_x[0]]
            i1 = i1+1
        else:
            final_scale[i2] = (oscale[close_to_x[1]], x_scale[close_to_x[1]][0], x_scale[close_to_x[1]][1])
            x_scale = np.delete(x_scale,close_to_x[1],0)
            del oscale[close_to_x[1]]
            i2 = i2 -1

    return final_scale


# if COMMA
# ASSUMES extremes are [0] and [-1]
def run_two_sided():
    with open("gold.txt", "r", encoding="utf-8") as gold, open("two_sided.txt", "w",encoding="utf-8") as predict:
        scale = []
        for line in gold:
            if line[0] == "=":
                test_scale = []
                # Scale of size two or 1
                if len(scale) <3:
                    for s in scale:
                        test_scale.append(s)
                    for s in test_scale:
                        predict.write(s + "\n");

                elif len(scale) ==3:
                    for s in scale:
                        test_scale.append(s)
                    predict.write(scale[0] + "\n")
                    d1 = vecs.distance(scale[0], scale[1])
                    d2 = vecs.distance(scale[-1], scale[1])
                    predict.write(scale[1] + "(" +  scale[0] +" -- " + str(d1) + ", " + scale[2] + " -- " + str(d2) + ")\n")
                    predict.write(scale[-1] + "\n")


                else:
                    p_scale = two_sided([scale[0], scale[-1]], scale[1:-1])
                    predict.write(scale[0] + "\n")
                    test_scale.append(scale[0])
                    for p in p_scale:
                        predict.write(p[0] + "(" + scale[0] + " -- " + str(p[1]) + ", " + scale[-1] + " -- " + str(
                            p[2]) + ")\n")
                        test_scale.append(p[0])
                    predict.write(scale[-1] + "\n")
                    test_scale.append(scale[-1])
                if len(test_scale) > 1 and len(scale) > 1:
                    SPEARMAN_BUFFER.append (calc_spearman(scale, test_scale))
                scale = []
                predict.write(line)

            else:
                # clean_up = line.split(",")
                scale.append(line.strip())
                # if "," in line:
                #     continue
                # else:
                # # clean_up = line.split(",")
                # # for c in clean_up:
                #     scale.append(line.strip())
    gold.close()
    predict.close()

def preprocess():
    with open("gold.txt", "r", encoding="utf-8") as gold, open("gold2.txt", "w", encoding="utf-8") as g2:
        for l in gold:
            if l[0] != "=":
                clean_up = l.split(",")
                for c in clean_up:
                    g2.write(c.strip() + "\n")
            else:
                g2.write(l)
    g2.close()
    gold.close()

# def one_sided(x):

# with open("quartile_vecs.txt", "r", encoding="utf-8") as q, open("new_q.txt", "w", encoding="utf-8") as q2:
#     for line in q:
#         if line[0] != "(":
#             q2.write(line)
#         else:
#             l = line.strip()[1:-1]
#             num = float(l.split(",")[1].strip())
#             new_line = (line.split(",")[0].strip()[1:] + ":" + "%.3f\n") % num
#             q2.write(new_line)
# q.close()
# q2.close()

# preprocess()
run_two_sided()
print(SPEARMAN_BUFFER)
total = 0.0
for s in SPEARMAN_BUFFER:
    total += s[0]
total = total/len(SPEARMAN_BUFFER)
print(total)

# Greedy Algorithm
# IGNORING COMMAS (93.9% avg spearman correlation)
# WITH COMMAS (85.4% avg spearman correlation)
# SELECTING SECOND VALUE OF COMMAS (.9311)
# SELECTING FIRST VALUE OF COMMAS (.934)
# NO CLEANUP (ERROR)
# WEIRD GOLD STANDARD------------
# possible
# realistic
# feasible
# practical
