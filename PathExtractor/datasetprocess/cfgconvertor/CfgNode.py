class CfgNode(object):
    """控制流图中的节点，存储节点对应的源代码以及出度节点列表
    """

    def __init__(self, source_code=""):
        self.source_code = source_code
        self.children = set()
        self.parent = set()
        self.visited = False
        self.is_return = False

    def add(self, node):
        if isinstance(node, CfgNode):
            if self.is_return is False:
                self.children.add(node)
                node.parent.add(self)

    def __str__(self):
        """输出当前节点的源代码
        """
        return str(id(self)) + ": " + self.source_code
