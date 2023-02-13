#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:21:52 2023

@author: ali
"""
#https://stackoverflow.com/questions/31508574/semaphores-on-python
import threading
import time
sem1 = threading.Semaphore(0)
sem2 = threading.Semaphore(0)
sem3 = threading.Semaphore(0)

def fun1():
    while True:
        print(1)
        #sem1.release() #sem1=1
        time.sleep(0.50)

def fun2():
    while True:
        #sem1.acquire() #sem1=0
        print(2)
        sem2.release() #sem2=1
        time.sleep(0.50)
        

def fun3():
    while True:
        sem2.acquire() #sem2=0
        print(3)
        time.sleep(0.50)


if __name__=="__main__":
    print("Thread count: {}".format(threading.active_count()))
    print("threading.enumerate() :{}".format(threading.enumerate() ))
    t = threading.Thread(target = fun1)
    t.start()
    t2 = threading.Thread(target = fun2)
    t2.start()
    t3 = threading.Thread(target = fun3)
    t3.start()