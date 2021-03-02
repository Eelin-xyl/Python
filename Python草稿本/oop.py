class Solution:
    def twoSum(self, nums, target):
        for i in range(len(nums)):
            if target - nums[i] in nums:
                if i != nums.index(target - nums[i]):
                    return i, nums.index(target - nums[i])

if __name__ == '__main__':
    so = Solution()
    result = so.twoSum([2, 7, 11, 15], 18)
    print(result)


class Solution:
    def __init__(self, a):
        self.a = a
    def twoSum(self, nums, target):
        for i in range(len(nums)):
            if target - nums[i] in nums:
                if i != nums.index(target - nums[i]):
                    return i, nums.index(target - nums[i])


if __name__ == '__main__':
    solution = Solution(5)
    result = solution.twoSum([1,2,3,4],5)
    print(result)