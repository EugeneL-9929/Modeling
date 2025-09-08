class AnchorLinkTreeNode():
    def __init__(self, layerNumber, serialNumber, parent=None):
        self.layerNumber = layerNumber
        self.serialNumber = serialNumber
        self.parent = parent
        self.children = []
        self.layerRepeated = []

    def addChild(self, childNode):
        self.children.append(childNode)

    def toJson(self):
        return {
            'layerNumber' : self.layerNumber,
            'serialNumber' : self.serialNumber,
            'parent' : self.parent,
            'children' : [i.toJson() for i in self.children]
        }

    @classmethod
    def loadJson(cls, jsonData):
        node = cls(jsonData['layerNumber'], jsonData['serialNumber'], jsonData['parent'])
        for childNode in jsonData['children']:
            node.addChild(cls.loadJson(childNode))
        return node