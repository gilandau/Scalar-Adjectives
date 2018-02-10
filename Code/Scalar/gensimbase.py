from gensim.models import KeyedVectors
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


a = vecs.get_vector("good")
b = vecs.get_vector("best")
c = b + ((a-b)/2)
m = vecs.most_similar(positive=[c], topn=10)
print(m)


a = vecs.get_vector("bad")
b = vecs.get_vector("worst")
c = b + ((a-b)/2)
m = vecs.most_similar(positive=[c], topn=8)
print(m)




a = vecs.get_vector("slow")
b = vecs.get_vector("slowest")
c = b + ((a-b)/2)
m = vecs.most_similar(positive=[c], topn=8)
print(m)


a = vecs.get_vector("fast")
b = vecs.get_vector("fastest")
c = b + ((a-b)/2)
m = vecs.most_similar(positive=[c], topn=10)
print(m)


a = vecs.get_vector("happy")
b = vecs.get_vector("happiest")
c = b + ((a-b)/2)
m = vecs.most_similar(positive=[c], topn=10)
print(m)


a = vecs.get_vector("sad")
b = vecs.get_vector("saddest")
c = b + ((a-b)/2)
m = vecs.most_similar(positive=[c], topn=10)
print(m)


a = vecs.get_vector("angry")
b = vecs.get_vector("angriest")
c = b + ((a-b)/2)
m = vecs.most_similar(positive=[c], topn=10)
print(m)
# #################################################################33


a = vecs.get_vector("furious")
b = vecs.get_vector("happy")
c = b + ((a-b)/2)
1q = b + ((a-b)/2)
m = vecs.most_similar(positive=[c], topn=3)
3q =
print(m)


a = vecs.get_vector("furious")
b = vecs.get_vector("calm")
c = b + ((a-b)/2)
m = vecs.most_similar(positive=[c], topn=10)
print(m)


a = vecs.get_vector("good")
b = vecs.get_vector("best")
c = b + ((a-b)/2)
m = vecs.most_similar(positive=[c], topn=10)
print(m)
