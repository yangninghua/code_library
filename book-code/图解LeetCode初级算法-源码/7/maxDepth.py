        if root == None:
            return 0
        else:
            return 1 + max(self.maxDepth(root.right), self.maxDepth(root.left))