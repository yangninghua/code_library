class MaxHeap:
    """ 最大堆 """

    def __init__(self):
        self.max_heap_list = []

    def create_max_heap(self, new_list: list):
        """ 创建新的最大堆 """
        self.max_heap_list = new_list

        # 从第一个非子节点开始调整最大堆
        for i in range((len(self.max_heap_list) - 1) // 2, -1, -1):
            self.adjust_sub_max_heap(i)

    def insert_new_node(self, node: int):
        """ 插入新的节点值 """
        # 在数组的末尾插入新的节点值
        self.max_heap_list.append(node)

        # 调整该节点所在的父节点及其父节点的最大堆
        father_node_index = len(self.max_heap_list) - 1
        while father_node_index >= 0:
            father_node_index = (father_node_index - 1) // 2
            # 插入时，当该节点所在的最大堆不需要调整时，则不需要进一步调整其父节点的最大堆
            if not self.adjust_sub_max_heap(father_node_index):
                break

    def remove_max_node(self) -> int:
        """ 删除并返回最大的节点值 """
        # 删除顶点节点并保存最大的节点值
        max_node = self.max_heap_list[0]

        # 只有一个节点值时，清空最大堆并返回最大值
        if len(self.max_heap_list) == 1:
            self.max_heap_list.clear()
            return max_node

        # 将末尾的节点移至头结点,并删除尾结点
        self.max_heap_list[0] = self.max_heap_list[len(self.max_heap_list) - 1]
        del self.max_heap_list[len(self.max_heap_list) - 1]

        # 调整最大堆
        self.adjust_sub_max_heap(0)

        # 返回最大值
        return max_node

    def adjust_sub_max_heap(self, top_node_index: int) -> bool:
        """
        调整以某节点的顶点的最大堆
        :param top_node_index: 节点所在的序号
        :return: 是否调整过以该节点为顶点的最大堆
        """
        # 当编号小于0或者无子节点时，不需要调整以该节点为顶点的最大堆
        if top_node_index < 0 or top_node_index > (len(self.max_heap_list) - 1) // 2:
            return False

        left_node_index = 2 * top_node_index + 1
        right_node_index = 2 * top_node_index + 2

        # 获取顶节点、左节点和右节点中的最大数字节点的位置
        max_num_index = top_node_index
        if left_node_index < len(self.max_heap_list) and self.max_heap_list[max_num_index] < self.max_heap_list[left_node_index]:
            max_num_index = left_node_index
        if right_node_index < len(self.max_heap_list) and self.max_heap_list[max_num_index] < self.max_heap_list[right_node_index]:
            max_num_index = right_node_index

        # print("%s: %d, %d" % (str(self.max_heap_list), top_node_index, max_num_index))

        # 调整堆和子堆的位置，因为小的数字往下移，可能导致子堆不满足最大堆的规定
        if max_num_index != top_node_index:
            self.swap_node(max_num_index, top_node_index)
            self.adjust_sub_max_heap(max_num_index)
            return True

        return False

    def swap_node(self, a_node_index: int, b_node_index: int):
        """ 交换两个节点 """
        temp = self.max_heap_list[a_node_index]
        self.max_heap_list[a_node_index] = self.max_heap_list[b_node_index]
        self.max_heap_list[b_node_index] = temp


if __name__ == '__main__':
    unsorted_list = [0, 7, 5, 8, 3, 5, 2, 4, 1, 10]
    # 最大堆：父结点的键值总是大于或等于任何一个子节点的键值；
    # 最小堆：父结点的键值总是小于或等于任何一个子节点的键值
    # 创建最大堆
    max_heap = MaxHeap()
    max_heap.create_max_heap(unsorted_list)
    print(max_heap.max_heap_list)

    # 最大堆插入新值
    print("insert......")
    max_heap.insert_new_node(6)
    print(max_heap.max_heap_list)

    # 最大堆删除最大值
    print("delete......")
    print(max_heap.remove_max_node())
    print(max_heap.max_heap_list)