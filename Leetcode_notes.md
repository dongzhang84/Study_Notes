# Leetcode

## Sort

[reference1](https://www.jianshu.com/p/8c78c34a6409)

![img](https://upload-images.jianshu.io/upload_images/2666001-443d76ca7d579476.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

#### Bubble Short

https://www.youtube.com/watch?v=xli_FI7CuzA



![img](https://img-blog.csdn.net/20160316103848750)



#### Selection Sort

[reference](https://www.runoob.com/w3cnote/selection-sort.html)

https://www.youtube.com/watch?v=g-PGLbMth_g



![img](https://www.runoob.com/wp-content/uploads/2019/03/selectionSort.gif)



#### Insertion sort

https://www.youtube.com/watch?v=JU767SDMDvA

Pseudocode:

```
i ← 1
while i < length(A)
    j ← i
    while j > 0 and A[j-1] > A[j]
        swap A[j] and A[j-1]
        j ← j - 1
    end while
    i ← i + 1
end while
```





#### Quick Sort

[reference](https://www.techiedelight.com/quicksort/)

Recursive algorithm

https://www.youtube.com/watch?v=Hoixgm4-P4M

![Quicksort Algorithm – C++, Java, and Python Implementation – Techie Delight](https://www.techiedelight.com/wp-content/uploads/Quicksort.png)





#### Merge Sort

https://www.youtube.com/watch?v=4VqmGXwpLqc

![Merge Sort - GeeksforGeeks](https://media.geeksforgeeks.org/wp-content/cdn-uploads/Merge-Sort-Tutorial.png)

![Merge sort implementation questions in Java - Stack Overflow](https://i.stack.imgur.com/wk49i.png)

```
procedure mergesort( var a as array )
   if ( n == 1 ) return a

   var l1 as array = a[0] ... a[n/2]
   var l2 as array = a[n/2+1] ... a[n]

   l1 = mergesort( l1 )
   l2 = mergesort( l2 )

   return merge( l1, l2 )
end procedure

procedure merge( var a as array, var b as array )

   var c as array
   while ( a and b have elements )
      if ( a[0] > b[0] )
         add b[0] to the end of c
         remove b[0] from b
      else
         add a[0] to the end of c
         remove a[0] from a
      end if
   end while
   
   while ( a has elements )
      add a[0] to the end of c
      remove a[0] from a
   end while
   
   while ( b has elements )
      add b[0] to the end of c
      remove b[0] from b
   end while
   
   return c
	
end procedure
```





## Binary Search

Templete I

```python
def binarySearch(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    if len(nums) == 0:
        return -1

    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    # End Condition: left > right
    return -1
```



Templete II

```python
def binarySearch(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    if len(nums) == 0:
        return -1

    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid

    # Post-processing:
    # End Condition: left == right
    if left != len(nums) and nums[left] == target:
        return left
    return -1
```



Templete III



```python
def binarySearch(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    if len(nums) == 0:
        return -1

    left, right = 0, len(nums) - 1
    while left + 1 < right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid
        else:
            right = mid

    # Post-processing:
    # End Condition: left + 1 == right
    if nums[left] == target: return left
    if nums[right] == target: return right
    return -1
```



## Backtracking

Templete

```python
def backtrack(candidate):
    if find_solution(candidate):
        output(candidate)
        return
    
    # iterate all possible candidates.
    for next_candidate in list_of_candidates:
        if is_valid(next_candidate):
            # try this partial candidate solution
            place(next_candidate)
            # given the candidate, explore further.
            backtrack(next_candidate)
            # backtrack
            remove(next_candidate)
```



##### 77. Combination

https://leetcode.com/problems/combinations/

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        
        result = []
        
        def backtrack(remain, comb, last):
            
            if remain == 0:
                #print(remain, comb, last)
                result.append(list(comb))
            elif last == n:
                return 
            
            for i in range(last+1, n+1):
                comb.append(i)                
                backtrack(remain-1, comb, i)             
                comb.pop()
                
        
        backtrack(k, [], 0)
        
        return result
```



Dictionary

```python
dict.values()

list(dict.values())
```



##### 79. Word Search



![img](https://assets.leetcode.com/uploads/2020/11/04/word2.jpg)



```
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true
```

Sample code:

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        
        m = len(board)
        n = len(board[0])
        
        path = []
        
        def dfs(i,j,l):
            if l == len(word): return True
            if (i<0 or i>=m or j<0 or j>=n
               or word[l] != board[i][j] or (i,j) in path):
                return False
            
            path.append((i,j))
            res = (dfs(i+1,j,l+1) or
                   dfs(i-1,j,l+1) or
                   dfs(i,j+1,l+1) or
                   dfs(i,j-1,l+1)
                  )
            path.remove((i,j))
            
            return res
        
        for i in range(m):
            for j in range(n):
                if dfs(i,j,0): 
                    print(i,j)
                    return True
                
        return False
```





## Dynamic Programming

##### 300. Longest Increasing Subsequence

```
Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.
```



Code:

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        
        DP = [1] * len(nums)
        
        for i in range(len(nums)-2,-1,-1):
            for j in range(i+1, len(nums)):
                if nums[i] < nums[j]:
                    DP[i] = max(DP[i], 1+DP[j]) 
                    
        print(DP)
        
        return max(DP)
```





## Depth First Search & Breadth First Search

DFS pseudocode: 

![DFS_1.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/Leetcode/DFS_1.png?raw=true)





##### 127. Word Ladder

```
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5
Explanation: One shortest transformation sequence is "hit" -> "hot" -> "dot" -> "dog" -> cog", which is 5 words long.
```

![BFS_1.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/Leetcode/BFS_1.png?raw=true)



Code:

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        
        if endWord not in wordList: return 0
        
        nei = collections.defaultdict(list)
        wordList.append(beginWord)
        
        for word in wordList:
            for j in range(len(word)):
                pattern = word[:j] + "*" + word[j+1:]
                nei[pattern].append(word)
                
        #print(nei)
        
        visit = set([beginWord])
        q = deque([beginWord])
        res = 1
        
        while q:
            print(q,visit) # the printer
            for i in range(len(q)):
                word = q.popleft()
                if word == endWord:
                    return res
                for j in range(len(word)):
                    pattern = word[:j] + "*" + word[j+1:]
                    
                    for neiWord in nei[pattern]:
                        if neiWord not in visit:
                            visit.add(neiWord)
                            q.append(neiWord)
            res +=1
            
        return 0
```

Look at the printer, each layer gives q and visit:

```
deque(['hit']) {'hit'}
deque(['hot']) {'hot', 'hit'}
deque(['dot', 'lot']) {'dot', 'lot', 'hot', 'hit'}
deque(['dog', 'log']) {'lot', 'dog', 'hot', 'dot', 'hit', 'log'}
deque(['cog']) {'lot', 'dog', 'hot', 'dot', 'cog', 'hit', 'log'}
```



##### 207. Course Schedule (DFS)

https://leetcode.com/problems/course-schedule/

**Example 1:**

```
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0. So it is possible.
```

**Example 2:**

```
Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.
```

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        
        preMap = {i:[] for i in range(numCourses)}
        
        for crs, pre in prerequisites:
            preMap[crs].append(pre)
            
        
        # visited is all course along the DFS path
        visited = set()
        
        def dfs(crs):
            if crs in visited: return False
            if preMap[crs] == []: return True
            
            visited.add(crs)
            for pre in preMap[crs]:
                if dfs(pre) == False: return False
                
            visited.remove(crs)
            
            preMap[crs] = []
            return True
        
        for crs in range(numCourses):
            
            if dfs(crs) == False: return False
        
        return True
```



##### 200. Number of Islands

https://leetcode.com/problems/number-of-islands/

**Example 1:**

```
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
```

**Example 2:**

```
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
```

Code

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        
        m = len(grid)
        if m == 0: return 0
        n = len(grid[0])
        if n == 0: return 0
        
        
        def dfs(grid, i, j):

            grid[i][j] = "0"

            dirts = [[-1,0],[1,0],[0,1],[0,-1]]

            for direction in dirts:
                i1, j1 = i+direction[0], j+direction[1]
                if (i1>=0 and i1< m and j1>=0 and j1< n and grid[i1][j1] == '1'):
                    dfs(grid,i1,j1)
    
    
        
        res = 0
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    dfs(grid,i,j)                    
                    res +=1
                  
        print(grid)    
        return res  
```







## Listed Notes

##### 141. Linked List Cycle

https://leetcode.com/problems/linked-list-cycle/



```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        
        nodes_seen = set()
        while head is not None:
            if head in nodes_seen:
                return True
            nodes_seen.add(head)
            head = head.next
            
            #print(head.val)
        return False
```



##### 160. Intersection of Two Linked Lists

https://leetcode.com/problems/intersection-of-two-linked-lists/



```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        
        notes_B = set()
        
        while headB is not None:
            if headB not in notes_B:
                notes_B.add(headB)
                print(headB.val)
            headB = headB.next
            
        while headA is not None:
            if headA in notes_B:
                return headA
            headA = headA.next
```



##### 206. Reverse Linked List

![img](https://assets.leetcode.com/uploads/2021/02/19/rev1ex1.jpg)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        prev = None
        
        while head:
            temp = head
            head = head.next
            temp.next = prev
            prev = temp
            
        return prev
```



## Binary Tree

![Example Tree](https://media.geeksforgeeks.org/wp-content/cdn-uploads/2009/06/tree12.gif)

Depth First Traversals: 

(a) **Inorder** (Left, Root, Right) : 4 2 5 1 3 
(b) **Preorder** (Root, Left, Right) : 1 2 4 5 3 
(c) **Postorder** (Left, Right, Root) : 4 5 2 3 1



##### 144. Binary Tree Preorder Traversal

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        
        if root is None:
            return []
        
        stack, output = [root, ], []
        
        while stack:
            root = stack.pop()
            if root is not None:
                output.append(root.val)
                
                if root.right is not None:
                    stack.append(root.right)
                if root.left is not None:
                    stack.append(root.left)
                        
        return output
```





##### 94. Binary Tree Inorder Traversal

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        
        # recursively
        res = []
        self.helper(root, res)
        return res
    
    def helper(self, root, res):
        if root:
            self.helper(root.left, res)
            res.append(root.val)
            self.helper(root.right, res)
```



##### 145. Binary Tree Postorder Traversal

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        
        # recursively
        res = []
        self.helper(root, res)
        return res
    
    def helper(self, root, res):
        if root:
            self.helper(root.left, res)
            self.helper(root.right, res)
            res.append(root.val)
```



##### 101. Symmetric Tree

**Example 1:**

![img](https://assets.leetcode.com/uploads/2021/02/19/symtree1.jpg)

```
Input: root = [1,2,2,3,4,4,3]
Output: true
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2021/02/19/symtree2.jpg)

```
Input: root = [1,2,2,null,3,null,3]
Output: false
```



Code:

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        
        if root is None:
            return True
        else:
            return self.isMirror(root.left, root.right)
        
        
    def isMirror(self, left, right):
        if left is None and right is None: return True
        if left is None or right is None: return False
        
        if left.val == right.val:
            outPair = self.isMirror(left.left, right.right)
            inPair = self.isMirror(left.right, right.left)
            
            return outPair and inPair
        
        return False
```



