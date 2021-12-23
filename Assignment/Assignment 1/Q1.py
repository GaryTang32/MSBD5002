import pprint


class HashTree:
    def __init__(self, leaf_size=3, branch_size=3, item_len=3):
        self.leaf_size = leaf_size
        self.branch_size = branch_size
        self.item_len = item_len

    def build(self, data, index) -> list:
        # Base case
        if len(data) <= self.leaf_size:
            return data

        # Split list of itemsets into 3 branch
        output = [[] for _ in range(self.branch_size)]

        # insert the itemset into the child branch by its hash value
        # Hash value = 1 go to lef
        # Hash value = 2 go to middle
        # Hash value = 0 go to right
        for i in data:
            output[self.hash(i, index)].append(i)

        # Check whatever we need to further split the branch, or need to use the linkedlist
        for i in range(self.branch_size):
            # case 1 stil can split the branch
            if (len(output[i]) > self.leaf_size) and (index < self.item_len - 1):
                output[i] = self.build(output[i], index + 1)
            elif len(output[i]) > self.leaf_size:  # cannot split and need link list to store the data
                output[i] = [output[i][0], output[i][1], output[i][2:]]

        # Return the Hashtree
        return output

    def hash(self, n, index) -> int:
        # i have added "n[index] + 2", so that
        # 1, 4, 7 will go to left
        # 2, 5, 8 will go to middle
        # 3, 6, 9 will go to right
        temp = n.copy()
        return (temp[index] + 2) % self.branch_size


def sort_remove_duplicates(data):
    # According to the important node of assignment 1
    # [1, 3, 6] and [6, 1, 3] will map into exact the same bucket
    #
    # when they are at the same bucket. there is no reason to keep both [1, 3, 6] and [6, 1, 3]
    # at the same bucket to increase the number of comparison.
    # Because this is itemset. Set have no order concept.
    # When set have no order concept, this is the same result
    # comparing "[1,3,6] with [1,3,6]" and "[1,3,6] with [6,3,1]" Both return "Match"
    # So for same itemset, only 1 is needed to keep, another one can be discard.
    # There is no reason to compare it twice while the result is the same.
    #
    #
    # we need to sort the itemsets order before the hashing.
    #
    # So here will remove the duplicated itemsets
    output = list()
    for i in data:
        i.sort()
        if i not in output:
            output.append(i)
    return output


if __name__ == '__main__':
    data = [
        [1, 2, 3], [1, 4, 5], [1, 2, 4], [1, 2, 5], [1, 5, 9], [1, 3, 6],
        [2, 3, 4], [2, 5, 9],
        [3, 4, 5], [3, 5, 6], [3, 5, 9], [3, 8, 9], [3, 2, 6],
        [4, 5, 7], [4, 1, 8], [4, 7, 8], [4, 6, 7],
        [6, 1, 3], [6, 3, 4], [6, 8, 9], [6, 2, 1], [6, 4, 3], [6, 7, 9],
        [8, 2, 4], [8, 9, 1], [8, 3, 6], [8, 3, 7], [8, 4, 7], [8, 5, 1], [8, 3, 1], [8, 6, 2]
    ]

    data = sort_remove_duplicates(data)
    hashtree = HashTree()
    print("-" * 100, '\nThe Hash tree is:')
    pprint.pprint(hashtree.build(data, 0))
