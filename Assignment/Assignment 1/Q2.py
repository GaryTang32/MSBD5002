from copy import deepcopy


class Node:
    # Define the node class for the FP-Tree
    # next is the next node
    # next last is the next node with the same name
    def __init__(self, name, prev, cnt):
        self.name = name
        self.value = cnt
        self.next = None
        self.prev = prev
        self.next_last = None


# This is the linked list class for the whole FP-tree
class Linkedlist:
    def __init__(self):
        node = Node('root', None, 1)
        self.head = node
        self.item_link = list()

    # Using recursion to insert the itemsets into the tree.
    def insert(self, data, node, cnt):
        if not data:
            return
        item = data[0]
        # if the node have child, search the child with same name, if same name add the frequency
        # if not found add the new node into the tree and make the link with other same name node
        if node.next:
            for i in node.next:
                if i.name == item:
                    i.value += cnt
                    self.insert(data[1:], i, cnt)
                    return
        new_node = Node(item, node, cnt)
        if node.next:
            node.next.append(new_node)
        else:
            node.next = [new_node]
        self.link_to_last(new_node)
        self.insert(data[1:], new_node, cnt)

    # Link the node with the other node with the same name for constructing the itemsets
    def link_to_last(self, node):
        name = node.name
        for i in self.item_link:
            if i.name == name:
                last_node = self.find_last_node(i)
                last_node.next_last = node
                return
        self.item_link.append(node)

    # loop to the last same name node
    def find_last_node(self, i):
        while i.next_last != None:
            i = i.next_last
        return i


# This is the FP-tree class
class FPTree:
    def __init__(self, support):
        self.support_threshold = support

    # this function will build the whole tree and return the list of frequent itemsets.
    def build(self, file_path) -> list:
        # Variable declaration
        data = list()
        all_words = dict()

        # read the data from the CSV
        with open(file_path, 'r') as temp_f:
            for line in temp_f.readlines():
                words_token = line.split(',')[:-1]
                data.append(words_token)
                # Step 1: find out all the words' support.
                for j in words_token:
                    if j in all_words.keys():
                        all_words[j] += 1
                    else:
                        all_words[j] = 1
        # Step 1: if the words is infrequent, then store it.
        infrequent_words = list()
        for i in all_words:
            if all_words[i] < self.support_threshold:  # Here the support thresold is 2500
                infrequent_words.append(i)

        # Step 1: Remove infrequent words and sort the order of the records.
        # Make sure the records' item is in alphabetical order.
        for record in data:
            for item in record:
                if item in infrequent_words:
                    record.remove(item)
                record.sort()

        # Step 2 construct the FP-tree by inserting the Frequency itemset.
        tree = Linkedlist()
        for itemset in data:
            tree.insert(itemset, tree.head, 1)
        item_link = tree.item_link

        # Find out all the frequent items based on each appearing item eg: frequency item list based on 'Bread'
        # Find out all the list of itemsets by each item.
        # eg. the list of itemsets by "Bread"
        # { (bread:1, butter:1),
        #   (bread:1, Coffee Powder:1),
        #   (bread:1, lassi:1) }

        cond_frequent_item, fp_itemsets, single_frequent = self.frequent_items_and_fp_itemsets(item_link)

        # Construct the cond FP-tree based on the conditional pattern bases of each item
        # This function will generate the conditional pattern base, and conditional FP-tree
        cond_fp_tree = self.cond_fp_frequent_itemset(cond_frequent_item, fp_itemsets)

        # This function will take the conditional FP-Tree, and with the tree generate the final frequent itemsets
        # for each item
        # like this will return the list of frequent itemset found by the conditional FP-Tree of Bread,
        # and conditional FP-Tree of 'Lassi' etc.
        frequent_itemset = self.frequent_itemset(cond_fp_tree, single_frequent)

        # Finally return all the frequent itemset out
        return frequent_itemset

    def frequent_items_and_fp_itemsets(self, item_link):
        fp_itemsets, fp_cond_frequent_prep, fp_cond_frequent, single_frequent = dict(), dict(), dict(), dict()

        # Get all itemset from the FP-Tree
        for i in item_link:
            target = i.name
            fp_itemsets[target] = [self.get_list(i)]
            while i.next_last:
                i = i.next_last
                fp_itemsets[target].append(self.get_list(i))

        # find out the frequent items for each distinct item
        # Eg. find out the frequent item to "Bread"
        for target_item in fp_itemsets:
            fp_cond_frequent_prep[target_item] = dict()
            fp_cond_frequent[target_item] = dict()
            for row in fp_itemsets[target_item]:
                record, support = row
                for item in record:
                    if item in fp_cond_frequent_prep[target_item].keys():
                        fp_cond_frequent_prep[target_item][item] += support
                    else:
                        fp_cond_frequent_prep[target_item][item] = support

            # save the item if the support is more than 2500
            for item in fp_cond_frequent_prep[target_item]:
                if (fp_cond_frequent_prep[target_item][item] >= self.support_threshold) and (item != target_item):
                    fp_cond_frequent[target_item][item] = fp_cond_frequent_prep[target_item][item]
                elif item == target_item:
                    single_frequent[target_item] = fp_cond_frequent_prep[target_item][item]

        # return the result
        return fp_cond_frequent, fp_itemsets, single_frequent

    # This function is used to loop the FP-tree and return the itemset with support
    # Sample output: [['Bread', 'Butter', 'Lassi'], 423]
    # Where ['Bread', 'Butter', 'Lassi'] is the itemset
    # 423 is the support

    def get_list(self, node):
        output_itemset = list()
        output_times = 99999999
        pointer = node
        while pointer.name != 'root':
            output_itemset.append(pointer.name)
            output_times = min(output_times, pointer.value)
            pointer = pointer.prev
        return [output_itemset, output_times]

    # This function will return all the itemset with support of the conditional FP-tree
    # Sample output: [[['Milk', 'Ghee'], 685], [['Milk', 'Ghee'], 642], [['Milk', 'Ghee', 'Ghee'], 642], [['Milk'], 2503]]
    # Where ['Milk', 'Ghee'] is the itemset
    # 685 is the support of the itemset
    def get_full_list(self, item_link, item, item_f):
        output = list()
        for i in item_link:
            pointer = i

            # Base Case
            output_itemset = [[[item, pointer.name], min(item_f, pointer.value)]]
            pointer = pointer.prev
            while pointer.name != 'root':
                cur, support = deepcopy(output_itemset[-1])
                cur.append(pointer.name)
                support = min(support, pointer.value)
                output_itemset.append([cur, support])
                pointer = pointer.prev
            output = output + output_itemset

            # Iterative Case
            while i.next_last:
                i = i.next_last
                pointer = i
                output_itemset = [[[item, pointer.name], min(item_f, pointer.value)]]
                while pointer.name != 'root':
                    cur, support = deepcopy(output_itemset[-1])
                    cur.append(pointer.name)
                    support = min(support, pointer.value)
                    output_itemset.append([cur, support])
                    pointer = pointer.prev
                output = output + output_itemset

        # Return list of itemset and support
        return output

    # This function will loop and create all the frequency itemset
    def frequent_itemset(self, cond_fp_tree, single_frequent):
        output_itemsets, output = dict(), dict()
        # Loop thought all conditional FP-Tree
        # For each conditional FP-tree, get all the itemset and the support.
        # Aggregate the same itemset and support
        # Save itemset with support >= 2500
        # return the result
        for target_item in cond_fp_tree:
            output_itemsets[target_item] = dict()
            item_f = single_frequent[target_item]
            link = cond_fp_tree[target_item].item_link
            itemsets = self.get_full_list(link, target_item, item_f)

            # Check is the itemset duplicates, if so add the support.
            for itemset, support in itemsets:
                if tuple(itemset) in output_itemsets[target_item].keys():
                    output_itemsets[target_item][tuple(itemset)] += support
                else:
                    output_itemsets[target_item][tuple(itemset)] = support

            # Save the itemset only support more than support threshold
            for key in list(output_itemsets[target_item].keys()):
                if output_itemsets[target_item][key] < self.support_threshold:
                    del output_itemsets[target_item][key]
            output_itemsets[target_item][tuple([target_item])] = item_f

        # Make the output sorted and organized
        for i in list(output_itemsets.values()):
            output.update(i)
        output = {k: v for k, v in sorted(sorted(output.items()), key=lambda item: item[1], reverse=True)}
        return output

    def cond_fp_frequent_itemset(self, cond_frequent_item, fp_itemsets):
        fp_conditional_itemset, cond_fp_tree = dict(), dict()
        for key in fp_itemsets.keys():
            fp_conditional_itemset[key] = dict()
            frequent_item_list = set(cond_frequent_item[key].keys())

            # This part will create the conditional pattern base of each item
            # First loop the itemsets one by one and interact it with the frequnt item.
            # then aggregate the same pattern to form a conditinal pattern base
            for row in fp_itemsets[key]:
                itemset, support = row
                intersact = list(set(itemset).intersection(frequent_item_list))
                intersact.sort()
                if tuple(intersact):
                    if tuple(intersact) in fp_conditional_itemset[key].keys():
                        fp_conditional_itemset[key][tuple(intersact)] += support
                    else:
                        fp_conditional_itemset[key][tuple(intersact)] = support

            # After found the conditonal pattern base of each distinct item
            # Create another FP-tree call Conditional FP tree for each distinct item
            # insert the conditional pattern base to the conditional FP tree for each distinct item

            cond_fp_tree[key] = Linkedlist()
            for cond_itemset in fp_conditional_itemset[key]:
                cond_itemset_in = list(cond_itemset)
                cond_itemset_in.sort()
                cond_fp_tree[key].insert(cond_itemset_in, cond_fp_tree[key].head,
                                         fp_conditional_itemset[key][cond_itemset])

        # return the list of conditional FP-tree
        return cond_fp_tree


if __name__ == '__main__':
    # Create the FP-tree
    FPTree = FPTree(support=2500)

    # Define the file
    file_path = r'DataSetA.csv'

    # Construct and print the result
    result = FPTree.build(file_path)
    for i in result:
        print(i, ':', result[i])
    print('There are totally', len(result), 'frequent itemset(s).')
