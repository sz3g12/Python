#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Basic solution - 2^n O
def fib(n):
    if n == 1 or n == 2:
        result = 1
    else:
        result = fib(n-1) + fib(n-2)
    return result


# In[8]:


fib(5)


# In[9]:


fib(35)
# already taking a few seconds


# In[12]:


# A memoized solution - 2n+1 O
def fib_2(n, memo):
    if memo[n] is not None:
        return memo[n]
    if n == 1 or n == 2:
        result = 1
    else:
        result = fib_2(n-1, memo) + fib_2(n-2, memo)
    memo[n] = result
    return result

def fib_memo(n):
    memo = [None] * (n + 1)
    return fib_2(n, memo)


# In[14]:


fib_memo(35)


# In[15]:


fib_memo(1000)


# In[18]:


fib_memo(5000)
# start to give RecursionError at 5000


# In[4]:


# A bottom-up solution - n O
def fib_bottom_up(n):
    if n == 1 or n == 2:
        return 1
    bottom_up = [None] * (n+1)
    bottom_up[1] = 1
    bottom_up[2] = 1
    for i in range(3, n+1):
        bottom_up[i] = bottom_up[i-1] + bottom_up[i-2]
    return bottom_up[n]


# In[21]:


fib_bottom_up(5000)
# handled easily

