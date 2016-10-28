def openData(doc):
    text = open(doc, 'r')
    text = text.read()
    return text
def saveData(doc, data):
    text = open(doc, 'w')
    text = text.write(data)
def parse(doc):
    text = openData(doc)
    lines = text.split('\n')
    heights = []
    weights = []
    for line in lines:
        components = line.split(' ')
        heights.append(components[1])
        weights.append(components[2])
    return heights, weights
parse('HeightsWeightsData.txt')
