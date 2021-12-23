#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 19:52:18 2021

@author: ali
"""
def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    
    txt=''.join(colors[x] for x in args) + f'{string}' + colors['end']
    print(txt)
    return txt



wow=colorstr('fuck you')
print(wow)


input = 'red','bold','fuck off'

*args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string

print(string)
print(*args)
colors = {'black': '\033[30m',  # basic colors
          'red': '\033[31m',
          'green': '\033[32m',
          'yellow': '\033[33m',
          'blue': '\033[34m',
          'magenta': '\033[35m',
          'cyan': '\033[36m',
          'white': '\033[37m',
          'bright_black': '\033[90m',  # bright colors
          'bright_red': '\033[91m',
          'bright_green': '\033[92m',
          'bright_yellow': '\033[93m',
          'bright_blue': '\033[94m',
          'bright_magenta': '\033[95m',
          'bright_cyan': '\033[96m',
          'bright_white': '\033[97m',
          'end': '\033[0m',  # misc
          'bold': '\033[1m',
          'underline': '\033[4m'}

txt2=''.join(colors[x] for x in args)+ f'{string}' + colors['end']
txt3 = ''.join(colors[x] for x in args)
print("txt3 =",txt3 )

print(txt2)

print('\033[31m\033[1m fuck your good \033[0m' )
print(' fuck your bad')
