import sys
works_list=[
#r'D:\HIWI\python-script\new_new_results\3.7/3.7.py',
#r'D:\HIWI\python-script\new_new_results\3.9/3.9.py',
#r'D:\HIWI\python-script\new_new_results\4.1/4.1.py',
#r'D:\HIWI\python-script\new_new_results\4.3/4.3.py',
#r'D:\HIWI\python-script\new_new_results\4.4/4.4a.py',
#r'D:\HIWI\python-script\new_new_results\4.4/4.4b.py',
#r'D:\HIWI\python-script\new_new_results\4.5/4.5.py',
#r'D:\HIWI\python-script\new_new_results\4.6/4.6a.py',
#r'D:\HIWI\python-script\new_new_results\4.6/4.6b.py',
r'D:\HIWI\python-script\new_new_results\4.7/4.7.py',
r'D:\HIWI\python-script\new_new_results\4.10/4.10.py',
r'D:\HIWI\python-script\new_new_results\4.12/4.12a.py',
r'D:\HIWI\python-script\new_new_results\4.13/4.13a.py',
r'D:\HIWI\python-script\new_new_results\4.14/4.14a.py',
r'D:\HIWI\python-script\new_new_results\4.14/4.14b.py',
r'D:\HIWI\python-script\new_new_results\4.15/4.15.py',
r'D:\HIWI\python-script\new_new_results\4.18/4.18a.py',
r'D:\HIWI\python-script\new_new_results\4.18/4.18b.py',
r'D:\HIWI\python-script\new_new_results\4.19/4.19a.py',
r'D:\HIWI\python-script\new_new_results\4.19/4.19b.py',
r'D:\HIWI\python-script\new_new_results\4.20/4.20.py',
r'D:\HIWI\python-script\new_new_results\4.21/4.21a.py',
r'D:\HIWI\python-script\new_new_results\4.21/4.21b.py',
r'D:\HIWI\python-script\new_new_results\4.22/4.22.py',
r'D:\HIWI\python-script\new_new_results\4.23/4.23a.py',
r'D:\HIWI\python-script\new_new_results\4.23/4.23b.py',

]

for work in works_list:
    execfile(work)
    print "finish "+work
    __reset__() -f