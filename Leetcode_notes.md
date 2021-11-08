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



#### Example: 77. Combination

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

