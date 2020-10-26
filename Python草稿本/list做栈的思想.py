# ans = {}
# A = ["bella","label","roller"]
# while A != []:
#     a = A.pop()
#     print(a)

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        i = head
        while i:
            print(i.val)
            i = i.next
        stack = []
        pt = head
        while pt:
            tmp = pt.next
            pt.next = None
            stack.append(pt)
            pt = tmp
        print(stack)
        ans = ListNode(0)
        a = ans
        while stack:
            a.next = stack.pop()
            a = a.next
        # i = ans
        # while i:
        #     print(i.val)
        #     i = i.next
        return ans.next


node1 = ListNode(1)
node2 = ListNode(2)
node3 = ListNode(3)
node4 = ListNode(4)
node1.next = node2
node2.next = node3
node3.next = node4
s = Solution()
a = s.reverseList(node1)
i = a
while i:
    print(i.val)
    i = i.next