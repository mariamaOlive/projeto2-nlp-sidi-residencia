import json

#Function reads a json file
def readJsonFile(path):
    f = open(path)
    data = json.load(f)
    f.close()
    return data

    
#Function returns a dataframe with the text of the pairs
def getJsonDocumentPair(dataPath, pairId):
    listIds = pairId.split('_')
    doc1Id = listIds[0]
    doc2Id = listIds[1]

    doc1Path = dataPath + doc1Id[-2:] + '/' + doc1Id + '.json' 
    doc2Path = dataPath + doc2Id[-2:] + '/' + doc2Id + '.json' 

    doc1Json = readJsonFile(doc1Path)
    doc2Json = readJsonFile(doc2Path)

    return (doc1Json, doc2Json)


