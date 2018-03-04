from collections import defaultdict

nodes = {}
class ltNode:
    def __init__(self):
        word = ""
        intensity = 0
        gtNodes = set()

def store_files(file_name):
    with open(file_name, "r", encoding="utf-8") as gold_pairs:


        for line in gold_pairs:
            l = line.split(" ")
            if l[0] not in nodes:
                lt = ltNode()
                lt.word = l[0]
                lt.intensity = l[2]
                lt.gtNodes = set()
                nodes[l[0]] = lt


            if l[1] not in nodes:
                gt = ltNode()
                gt.word = l[1]
                gt.intensity = l[3]
                gt.gtNodes = set()

                nodes[l[0]].gtNodes.add(gt)
                nodes[l[1]] = gt
            else:
                nodes[l[0]].gtNodes.add(nodes[l[1]])



store_files("C:/Users/Geoff/Desktop/Research/CCB/Code/Scalar/adjectivePairs-SoCal/pos/pairsPos1")
store_files("C:/Users/Geoff/Desktop/Research/CCB/Code/Scalar/adjectivePairs-SoCal/pos/pairsPos2")
store_files("C:/Users/Geoff/Desktop/Research/CCB/Code/Scalar/adjectivePairs-SoCal/pos/pairsPos3")
store_files("C:/Users/Geoff/Desktop/Research/CCB/Code/Scalar/adjectivePairs-SoCal/pos/pairsPos4")
startNodes = list(nodes.keys())
notStartNodes = []

for s in startNodes:
    for n in nodes[s].gtNodes:
        notStartNodes.append(n.word)
for notstart in notStartNodes:
    if notstart in startNodes:
        startNodes.remove(notstart)
print(len(startNodes))

with open("gold_pairs.txt", "w", encoding="utf-8") as gold_pairs:
    for node in startNodes:
        n1 = nodes[node]
        paths = possible_paths(n1, [])
        for p in paths:
            gold_pairs.write("==============\n")
            for n in p:
                gold_pairs.write(n)
    gold_pairs.write("==============\n")
def possible_paths(start, []):
