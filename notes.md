# Mastering Concurrency in Python

---

# Chapter 1 - Advanced Introduction to Concurrent and Parallel Programming

Concurrent programming vs. sequential programming

Topics:
- The concept of concurrency
- Why some programs cannot be made concurrent, and how to differentiate them from programs that can
- The history of concurrency in computer science: how it is used in the industry today, and what can be expected in the future
- The specific topics that will be covered in each section/chapter of the book
- How to set up a Python environment, and how to check out/download code from GitHub

## What is concurrency?

Concurrent programming: one of the most prominent ways to effectively process data
- It is estimated that the amount of data that needs to be processed by computer programs doubles every two years

## Concurrent versus sequential

Most obvious way to understand concurrent programming: Compare it to sequential programming
- Sequential program: 1 place at a time
- Concurrent program: Different components are in independent/semi-independent states
    - Components in different states can be executed independently
        - Can be executed at the same time (1 component's execution does not depend on the result of the other)

Diagram to illustrate:

![Difference between concurrent and sequential programs](image.png)

Advantage of concurrency: Improved execution time
- Since some tasks are independent, they can be completed at the same time, so less time is required for the computer to execute the whole program

## Example 1 – checking whether a non-negative number is prime

Sequential program:

```py
# Chapter01/example1.py
from timeit import default_timer as timer
# sequential
start = timer()
result = []
for i in input:
 if is_prime(i):
 result.append(i)
print('Result 1:', result)
print('Took: %.2f seconds.' % (timer() - start))
```

Results:
- Time passed: 3.41 seconds
- Computer performance: 83% idle


Concurrent program:

```py
# Chapter01/example1.py
from timeit import default_timer as timer
# sequential
start = timer()
result = []
for i in input:
 if is_prime(i):
 result.append(i)
print('Result 1:', result)
print('Took: %.2f seconds.' % (timer() - start))
```

Results:
- Time passed: 2.33 seconds
- Computer performance: 37% idle


## Concurrent versus parallel

Is concurrency and different than parallelism? Yes.
- Differences:
    - Parallel programs: A number of processing flows (CPUs/cores) working independently all at once
    - Concurrent programs: Might be different processing flows (mostly threads)
        - These threads use a shared resource at the same time

Diagram to explain:

![Difference between concurrency and parallelism](image-1.png)

Parallelism: Top, where cars are in their own lane and don't interact with each other

Concurrency: Bottom, where cars need to wait there turn to cross the street (ie. wait for others to finish before they can execute)

## A quick metaphor

Concurrency is difficult to grasp right away, so here is a metaphor ( to make concurrency and its differences from parallelism easier to
understand):

Assume that different parts of the human brain are responsible for performance separate/exclusive body part actions:
- Example: left hemisphere of brain controls right side of body, and vice versa
- Example: left hemisphere of brain controls speaking, the other controls writing
    - If you want to move your left hand, only the right side of your brain can process that command
        - The left side of the brain is then free to do something else, like speaking

Parallelism: Where different processes don't interact with ie. are independent of each other
- One hand eating, other hand snapping ie. left/right hands for independent tasks at the same time

Concurrency: Sharing the same resources
- Juggling ie. two hands perform different tasks at the same time, but interact with the same object
    - Some form of coordination/communicationt between the two hands is required

## Not everything should be made concurrent

Not all programs are created equal:
- Some program: can be made parallel or concurrent relatively easily
- Others: inherently sequential
    - cannot be executed concurrently or in parallel
- Others: embarrassingly parallel
    - little or no dependency, no need for communication

## Embarrassingly parallel

Example: 3D video rendering handled by a graphics processing unit (GPU)
- each frame/pixel can be processed with no interdependency

Example: Password cracking
- can easily be distributed on CPU cores

Example: Web scraping

## Inherently sequential

Opposite of embarrasingly parallel: These tasks heavily depend on the results of others ie. tasks are not independent (Cannot be made parallel/concurrent)
- If we tried to implement concurrency, it could cost us more execution time for the SAME results

Example: Chapter01/example1.py
- Assuming we want the output of prime numbers in order, this is what happened:
    - Method 1 (sequential): we went down the line in order
        - Output stays in order
    - Method 2 (concurrent): since we split the tasks into different groups, the
        - Requires a sort at the end, which could increase execution time 

This brings up the topic of pregnancy!

Pregnancy: A topic used to illustrate the innate sequentialy of some tasks
- The number of women will never reduce the length of pregnancy
    - Adding more processors will NOT improve execution time
- Examples:
    - iterative algorithms
    - iterative numerical approximation methods

## Example 2 – inherently sequential tasks

Sequential program:

```py
# Chapter01/example2.py
import concurrent.futures
from timeit import default_timer as timer
# sequential
def f(x):
    return x * x - x + 1

start = timer()
result = 3
for i in range(20):
    result = f(result)

print('Result is very large. Only printing the last 5 digits:', result % 100000)
print('Sequential took: %.2f seconds.' % (timer() - start))
```

Results:
- Time passed: 0.10 seconds


Concurrent program:

```py
# Chapter01/example2.py
import concurrent.futures
from timeit import default_timer as timer
# concurrent
def concurrent_f(x):
    global result
    result = f(result)

result = 3

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as exector:
    futures = [exector.submit(concurrent_f, i) for i in range(20)]

    _ = concurrent.futures.as_completed(futures)

print('Result is very large. Only printing the last 5 digits:', result % 100000)
print('Concurrent took: %.2f seconds.' % (timer() - start))
```

Results:
- Time passed: 0.19 seconds

Why did this occur, when both methods produce the same result?
- Every time a new thread from `ThreadPoolExecutor` was spwaned, the function `concurrent_f()` inside of that thread need to wait for `result` to be processed by the previous thread completely
    - The program therefore executes in a sequential manner, anyways

There was no actual concurrency in the 2nd method! (Not to mention, the overhead cost of spawning new threads contributes to worse execution time as well)

This is an example where concurrency/parallelism should NOT be applied, as it is a inherently sequential task.

## I/O bound

Another way to think about sequentiality: The CS concept of a condition called I/O bound
- Time it takes to complete a computation is mainly determined by time spent waiting for the input/output (I/O) operations to be completed
    - This condition arises when the rate at which data is requested is slower than the rate at which it is being consumed
        - In other words: More time is spent requesting data than processing it

- I/O bound state: CPU stalls its operation, waiting for data to be processed
    - What this mean: Even if the CPU gets faster at processing data, processes tend to not increase in speed (in proportion to the increased CPU speed) since it just gets more I/O-bound
    - New computers/processors are very fast, so I/O bound states are undesirable (although they are quite common in programs now)

Remember: Do not see concurrency as a golden ticket

## The history, present, and future of concurrency

The field of concurrent programming has enjoyed popularity since the early days of computer science, so let's go over how this has evolved/is evolving:

### The history of concurrency

Concurrency: been around for a long time now!
- idea started: early work on railroads/telegraphy in the 19th/20th centuries
    - some terms have survived ie. semaphore
        - semaphore: a variable that controls access to a shared resource, in a concurrent program
- first application: how to handle multiple trains on the same railroad system
    - need to avoid collisions
    - wanted to maximize efficiency
- second application: how to handle multiple transmissions over a set of wires in telegraphy
- 1959: academic study of concurrency begins
    - Dijkstra paper in 1965
    - no considerable interest after this
- 1970-2000: processors were doubling in execution speed every 18 months
    - programmers did not need to learn concurrent programming
- early 2000s: manufacturers started focusing on groups of smaller/slower processors
    - think multicore processor
- nowadays: average computer has more than 1 core
    - if you write a program to be non-concurrent in any way, you only use 1 core/thread to process data (rest of CPU sits idle!)
- another reason for increasing popularity of concurrency: graphical/multimedia, web-based application development
    - example: web development
        - each new request made by a user comes in as its own process (multi-processing) OR asynchronously coordinated with other requests (asynchronous programming)
        - if any of the requests need to share a resource (ie. database), concurrency should be considered

### The present

Present day: explosive growth of the internet and data sharing happens every second
- concurrency is more important than ever
- current use emphasis:
    - correctness
    - performance
    - robustness
- some concurrent systems (operating systems, database management systems) operate indefinitely
    - have automatic recovery from failure
    - use shared resources, so require a semaphore to control/coordinate access to the shared resource(s)
- examples where concurrency is present:
    - common programming languages: C++, C#, Erlang, Go, Java, Julia, JavaScript, Perl, Python, Ruby, Scala, and so on
    - almost every computer has multiple cores, so to take advantage of this computing power, need well-designed software
    - iphone 4s (2011): has a dual core CPU
    - Xbox360/PS3 are multicore/multi-CPU
    - on average, Google processes over 40,000 search queries per second
        - 3.5 billion per day
        - 1.2 trillion per year
        - concurrency is the best way to handle this!
- cloud: a large % of today's data and applications are stored in the cloud
    - cloud computing instances are smaller, so web applications have to be concurrent
        - need to process small jobs simultaneously
        - web apps with good design just need to utilize more servers
- GPUs: used as parallel computing engines
    - almost all Kaggle prize-winning solutions use GPU during training processes
    - concurrency is an effective solution for combing through all of this big data
    - example of using concurrency to increase model-training time: AI algos that break input data down into smaller portions and process them independently

### The future

Today: Users expect instant output for all applications
- developers struggle to provide better speed for applications
    - concurrency is a unique solutions to this problem
- some may argue that concurrent programming may become more standard in academia
    - concurrency/parallelism are covered in CS
        - this is only the beginning
- more skeptical view: that concurrency is about dependency analysis
    - combination of low number of programmers who understand concurrency, and possibility of automating concurrency design, makes for decreased interest in learning
    - may be a push for compilers, with support from operating systems, to implement concurrency into the programs they compile
        - compiler will look at program, analyze statements/instructions, produce a dependency graph, and apply concurrency/parallelism
- time will tell!
- concurrent programming is very complicated and hard to get right
    - knowledge gained is beneficial

## A brief overview of mastering concurrency in Python

Python: one of the most popular programming languages out there
-  pros: comes with numerous libraries and frameworks that facilitate highperformance computing
-  cons: Global Interpreter Lock (GIL)
    - difficulty of implementing concurrent/parallel programs
    - concurrency and parallelism do behave differently in Python than in other common
    programming languages
    - it is still possible for programmers to implement programs that run concurrently or in parallel
- this book: provide a detailed overview of how concurrency and parallelism are being used in
real-world applications
    - theoretical analysis
    - practical examples

## Why Python?

Python: has GIL (Global Interpreter Lock)
- mutex that protects access to Python objects
    - prevents multiple threads from executing Python byte codes at once
    - necessary, as CPython's memory management is not thread-safe
        - thread-safe: a function is thread-safe when it can be invoked or accessed concurrently by multiple threads without causing unexpected behavior, race conditions, or data corruption
        - CPython uses reference counting, which can cause incorrect handling of data
- addressing problem with GIL
    - lock allows only 1 thread access to Python code and objects
    - this means to implement multithreading, you need to be aware of the GIL and work around it
- why work with Python at all if it has the GIL?
    - GIL is only a bottleneck for multithreaded programs that spend significant time in the GIL
        - prevents multithreadeded programs from taking full advantage of multiprocessor systems 
            - blocking operations ie. I/O, image processing, NumPy number crunching happen outside of the GIL
                - multiprocessing applications that do not share any common resources among processes, such as I/O, image processing, or NumPy number crunching, can work seamlessly with the GIL
        - other forms of concurrent programming do not have this problem (that multithreading does)

- why Python:
    - user friendly syntax
    - overall readability
    - development can be 10x faster than C/C++ code
    - strong and growing support community
    - sheer number of development tools available
        - vicious circle of Python. David Robinson, chief data scientist at DataCamp, wrote a blog (https://stackoverflow.blog/2017/09/06/incredible-growthpython/) about the incredible growth of Python, and called it the most popular programming language.

- other cons of Python:
    - slow (slower than other languages)
        - dynamically typed/interpreted language
            - values are stored in scattered objects, not dense buffers
                - direct result of having readability

    - luckily, we can use concurrency to and other options to speed up your programs

---

# Chapter 2 - Amdahl's Law

Amdahl's Law: explains the theoretical speedup of the execution of a program, when using concurrency

## Amdahl's Law

How to balance between parallelizing a sequential program (increasing # of processors) and optimizing the execution speed of the sequential program?
- Option 1: 4 processors run program at 40% of its execution
- Option 2: 2 processors run program, but for twice as long
    - this tradeoff in concurrent programming is analyzed via Amdahl's Law

Notes:
- Concurrency/parallelism are powerful, but not able to speed up any non-sequential architecture
    - Important to know its limits
        - Amdahl's Law helps with that!


## Terminology

Amdahl's Law: Provides a mathematical formula that calculates the potential improvement (in speed) of a concurrent program by increasing resources (# of available processors)
- this law applies to potential speedup when executing tasks in parallel
- speed: time for program to execute in full
- speedup: benefit of executing a computation in parallel
    - time to execute in serial (w/ 1 processor), divided by time to execute in parallel (w/ 2+ processors)

![The formula for speedup](image-2.png)

## Formula and interpretation

Let's assume we have N workers working on a job that is fully parallelizable:
- job is divided into N equal sections
    - N workers will do 1/N work
    - it will take 1/N time as 1 worker doing all of the work

Note: Most computer programs are NOT 100% parallelizable (some parts are inherently sequential)

## The formula for Amdahl's Law

![The formula for Amdahl's Law](image-3.png)

## A quick example

![A quick example](image-4.png)

## Implications

Gene Amdahl (1967):
- sequential overhead nature of a program sets an upper boundary on the possible speedup
- as the number of resources increases (ie. # of available processors), speed of execution increases
    - however: this does not mean to always use as many processors as possible!
        - speedup decreases eventually (as we add more processors for our concurrent program, we will obtain less and less improvement in execution time.)

upper limit of the execution time improvement:

![upper limit of the execution time improvement](image-5.png)

## Amdahl's Law's relationship to the law of diminishing returns

Diminishing returns: Popular concept in economics
- Only a special case of applying Amdahl's law: It depends on the order of improvement
    - Optimal method: First applying improvements that result in the greatest speedups
    - Reverse: Improve the less optimal components of the program first
        - Can be more beneficial since optimal components are usually complex

- Another similarity: Improvement of speedup via adding more processors
    - Fixed-size task: New processor added to the system offers less usable computation power than the previous processor
        - Remember: Throughput has an upper boundary
- Also need to keep in mind bottlenecks:
    - Memory bandwidth
    - I/O bandwidth
        - These don't usually scale with processors, so adding processors gives lower return

## How to simulate in Python

In this section, we will look at the results of Amdahl's Law through a Python program.

What this function does: checks for prime numbers

```py
# ch2/example1.py

from math import sqrt

import concurrent.futures
import multiprocessing

from timeit import default_timer as timer


def is_prime(x):
    if x < 2:
        return False

    if x == 2:
        return x

    if x % 2 == 0:
        return False

    limit = int(sqrt(x)) + 1
    for i in range(3, limit, 2):
        if x % i == 0:
            return False

    return x

```

The next part of the code: Indicates the number of processors (workers) we will be utilizing to concurrently solve the problem

```py
def concurrent_solve(n_workers):
    print('Number of workers: %i.' % n_workers)

    start = timer()
    result = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:

        futures = [executor.submit(is_prime, i) for i in input]
        completed_futures = concurrent.futures.as_completed(futures)

        sub_start = timer()

        for i, future in enumerate(completed_futures):
            if future.result():
                result.append(future.result())

        sub_duration = timer() - sub_start

    duration = timer() - start
    print('Sub took: %.4f seconds.' % sub_duration)
    print('Took: %.4f seconds.' % duration)


input = [i for i in range(10 ** 13, 10 ** 13 + 1000)]

```

Finally: We loop from one to the maximum number of processors available in our
system, and we will pass that number to the preceding concurrent_solve() function

```py
for n_workers in range(1, multiprocessing.cpu_count() + 1):
    concurrent_solve(n_workers)
    print('_' * 20)
```

Note: You can the number of available processors from your computer with this call in your terminal (I got 8)

```bash
python
import multiprocessing
multiprocessing.cpu_count()
exit()

```

Note: You may need to guard in order to avoid the following error:

```bash
RuntimeEntityError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...
```

How to ensure main guard and use freeze_support():

```bash
import multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()
```

Why is this: the multiprocessing module in Python needs a special setup on Windows.
- freeze_support() is called to ensure that when a new process is started on Windows, it doesn’t run into issues with recursive process creation
- Unlike on Unix-based systems, where the fork method is used to start child processes, Windows uses the spawn method
    - spawn method starts a fresh Python interpreter process
        - Due to this, the code running in the child process needs to be properly guarded by `if __name__ == '__main__':` to prevent unintended code execution when the module is imported

```py
import multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()
    for n_workers in range(1, multiprocessing.cpu_count() + 1):
        concurrent_solve(n_workers)
        print('_' * 20)
```

Let's run the program:

```bash
cd mastering_concurrency_in_python
python Mastering-Concurrency-in-Python/Chapter02/example1.py
```

Here is my output:

```bash
Number of workers: 1.
Sub took: 10.1639 seconds.
Took: 10.4939 seconds.
____________________
Number of workers: 2.
Sub took: 5.2571 seconds.
Took: 5.5624 seconds.
____________________
Number of workers: 3.
Sub took: 3.7803 seconds.
Took: 4.1331 seconds.
____________________
Number of workers: 4.
Sub took: 3.1190 seconds.
Took: 3.4789 seconds.
____________________
Number of workers: 5.
Sub took: 3.0214 seconds.
Took: 3.4106 seconds.
____________________
Number of workers: 6.
Sub took: 2.5960 seconds.
Took: 3.0392 seconds.
____________________
Number of workers: 7.
Sub took: 2.6030 seconds.
Took: 3.1581 seconds.
____________________
Number of workers: 8.
Sub took: 3.0615 seconds.
Took: 3.5715 seconds.
____________________
```

A few things to note:
1. In each iteration: Subtask took almost as long as entire program
- ie. the concurrent computation was the majority of the program
    - Makes sense, since prime checking is the only other real computation

2. We can see hardly an improvement from 3-4 processors
- 1-2 had considerable improvement
- 2-3 had considerable improvement
- 3-4, 4-5, etc. saw almost the same numbers
    - some took longer (this is probably due to overhead processing ie. work and resources used by a system or application that are not directly related to the primary tasks the system is meant to perform)
    - this aligns with our learing earlier of Amdahl's Law + the law of diminishing returns

Here is a graph that shows the relationship between Number of processors and speedup, based on the portion of the program that is parallel:

![Speedup curves with different parallel portions](image-6.png)

## Practical applications of Amdahl's Law

Using Amdahl's law, we can estimate the upper limit of potential speed improvements from parallel computing. From there, we can decide whether the increase in computing power is worth it.
- You apply it when you have a concurrent program that is a mixture of BOTH sequentially and executed-in-parallels instructions
- How we use it: To determine speedup through each incrementation in # of cores available

Going back to the initial problem at the beginning of the chapter, there is a trade-off between increasing # of processors vs. increasing in how long parallelism can be applied. Here is our example:
- Current state: We are developing a program with 40% of its instructions parallelizable, and we have 2 choices to increase the speed of the program:
    1. Have 4 processors
    2. Have 2 processors, but increase parallelizable portion to 80%

We can compare these two choices using Amdahl's Law:

![Amdahl's Law can assist us when comparing two choices, when determining the one that will produce the best speed](image-7.png)

We can calculate this with Python:

```bash
python
```

```py
b = (1 - 0.4) # 40% parallelizable
j = 4 # processors
s = 1 / (b + ((1 - b) / j))
print(f"speedup for choice 1: {s}")

b = (1 - 0.8) # 40% parallelizable
j = 2 # processors
s = 1 / (b + ((1 - b) / j))
print(f"speedup for choice 2: {s}")

```

As you can see, the second option (which has fewer processors than the first) is actually the
better choice to speed up our specific program.

Final note: Remember that this law makes some of assumptions, and by no means is it a be-all-end-all.
- Examples of important factors that it ignores:
    - Overhead of parallelism
    - Speed of memory

Best way to think about and use Amdahl's Law? We must actually measure the speedup by implementing in practice.

## Summary

Summary:
- Amdahl's Law gives us a way to estimate the potential speedup in execution time
- When only the number of processors increases, Amdahl's Law resembles the law of diminishing returns (The speedup curve flattens out)
- Improvement through concurrency and parallelism is NOT always what we want when designing an efficient concurrent program

Next chatper: We will discuss the tools that Python provides for us to implement concurrency, more specifically threads!

---

# Chapter 3 - Working with Threads in Python

In this chapter:
- We will be introduced to the formal definition of a thread, and the `threading` library in Python.
- A number of ways to work with threads in Python:
    - Creating new threads
    - Synchronizing threads
    - Working with multithreaded priority queues
- Learn about the concept of "a lock in thread synchronization"
- Implementing a lock-based multithreaded application
    
The following topics will be covered in this chapter:
- The concept of a thread in the context of concurrent programming in computer
science
- The basic API of the `threading` module in Python
- How to create a new thread via the `threading` module
- The concept of a lock and how to use different locking mechanisms to synchronize threads
- The concept of a queue in the context of concurrent programming, and how to use the `Queue` module to work with queue objects in Python

## The concept of a thread

In CS, a "thread of execution" is the smallest unit of programming commands (ie. code) that a scheduler (part of an OS) can process/manage.
- Depending on the OS, the implementation of threads and processes varies
- In general: A thread is an element/component of the process

## Threads vs. Processes

More than one thread can be implemented within the same 1 process, often executing concurrently and accessing/sharing the same resources (ie. memory)
- Separate processes do not do this
- Threads in the same process share:
    - The latter's instructions (code)
    - Context (the values that its variable reference at any given moment)

Key difference:
- Thread: Independent component of process of computation
    - Typically a component of a process
    - Allow for shared resources (Memory/data)
        - 
- Process:
    - Usually don't allow shared resources (it is rare)
    - Can include multiple threads
        - These threads can execute simultaneously
            - Can share address space/data

Example:

![A process with two threads of execution running on one processor
alt text](image-8.png)

## Multithreading

In Computer Science:
- Single-threading: Traditional sequential processing
    - 1 single command at any given time
- Multithreading: Implements 2+ threads to exist/execute 1 process
    - Allow access shared resources/contexts
    - Helps applications gain speed in execution of independent tasks

Multithreading can be achieved in 2 ways:
1. Single-processor systems
- Time Slicing: A technique that allows the CPU to switch between different software running on different threads

![An example of a time slicing technique called round-robin schedulingalt text](image-9.png)

2. Multiple-processor systems
- Systems with multiple processors/cores can easy implement multithreading
- Each thread is in a separate process/core, at the same time
- Time slicing is an option
    - Not good practice, as these multicore systems can only have 1 processor/core to switch between tasks

Advantaged of Multithreaded applications:
- Faster execution time: If threads are sufficiently independent of each other, you can execute them concurrently/in parallel
- Responsiveness: By using separate threads, you can take in different user input simultaneously
    - In single-threaded programs, if the main execution threads blocks on a long-running task (ie. heavy computational task), the whole program will not be able to continue with other input
- Efficiency in resource consumption: Serve and process many client requests for data concurrently
    - Since multiple threads can share/access the same resources, it takes less resources because you can process data requests concurrently
        - Fewer resources used = quicker communication between threads

And their disadvantages:
- Crashes: 1 illegal operation within 1 thread can affect the processing of all threads + crash the entire program
- Synchronization: Careful consideration is needed to "share" the resources
    - Usually: Must coordinate threads in a systematic manner (so that shared data is computed and manipulated correctly)
    - Unintuitive problems you may run into:
        - Deadlocks
        - Livelocks
        - Race conditions
            - We will discuss these more later!
        

## An example in Python

To illustrate the concept of running multiple threads in the same process, let's look at a
quick example in Python:

```py
# ch3/my_thread.py

import threading
import time


class MyThread(threading.Thread):
    def __init__(self, name, delay):
        threading.Thread.__init__(self)
        self.name = name
        self.delay = delay

    def run(self):
        print('Starting thread %s.' % self.name)
        thread_count_down(self.name, self.delay)
        print('Finished thread %s.' % self.name)

def thread_count_down(name, delay):
    counter = 5

    while counter:
        time.sleep(delay)
        print('Thread %s counting down: %i...' % (name, counter))
        counter -= 1

```

What is going on here:
- the `threading` module is the foundation of `MyThread` class
- Each object of this class has:
    - name
    - delay
    - `run()` method
        - calls `thread_count_down()` (This function counts down from the number 5 to the number 0, while sleeping between iterations for a number of seconds, specified by the delay parameter.)

Point of this example: Show concurrent nature of running 2+ threads in the same program/process
- This is achieved by starting more than 1 object of `MyThread`
    - My quick thought: If we start more than 8 threads, we may have an issue (My computer has 8 cores only)

This function is more of a class to be imported into other function, but you can run it if you want with this call:

```bash
cd mastering_concurrency_in_python
python Mastering-Concurrency-in-Python/Chapter03/my_thread.py
```

Next, let's look at the Chapter3/example1.py file:

```py
# ch3/example1.py

from my_thread import MyThread


thread1 = MyThread('A', 0.5)
thread2 = MyThread('B', 0.5)

thread1.start()
thread2.start()

thread1.join()
thread2.join()


print('Finished.')

```

What is going on here:
- Init 2 threads and run them together

Let's run the program:

```bash
cd mastering_concurrency_in_python
python Mastering-Concurrency-in-Python/Chapter03/example1.py
```

Our output looks something like this:

```bash
C:\Users\Myles\mastering_concurrency_in_python>python Mastering-Concurrency-in-Python/Chapter03/example1.py
Starting thread A.
Starting thread B.
Thread A counting down: 5...
Thread B counting down: 5...
Thread B counting down: 4...
Thread A counting down: 4...
Thread B counting down: 3...
Thread A counting down: 3...
Thread B counting down: 2...
Thread A counting down: 2...
Thread A counting down: 1...
Thread B counting down: 1...
Finished thread A.
Finished thread B.
Finished.
```

As you can see, both Thread A and Thread B were running simultaneously/executed concurrently.
- A sequential program would have had to count down for Thread A, then do Thread B next
-   We see almost a 2x improvement in speed
 - This does not account for overhead and declarations

Notes:
- See how the countdown of thread B actually got ahead of thread A in execution, even though we know that thread A was initialized and started before thread B
    - This phenomenon is a direct result of concurrency via multithreading
        - since the two threads were initialized and started almost simultaneously, it was quite likely for one thread to get ahead of the other in execution
        - if you executed this script many times, you would see varying outputs (I did it again and this time, Thread A printed out 4 first)

## An overview of the threading module

There are a lot of choices when it comes to implementing multithreaded programs in Python.
- One of the most common ways to work with threads in Python is through the `threading` module
- Before we dive into the module's usage and its syntax, first, let's explore the `thread` model
    - was previously the main thread-based development module in Python

## The thread module in Python 2

Before the `threading` module became popular, the primary thread-based development module was `thread`.
- If you are using an older version of Python 2, it is possible to use the module as it is.
    - However, according to the module documentation page, the thread module was renamed `_thread in Python 3

The main feature of the thread module is its fast and sufficient method of creating new
threads to execute functions: the thread.start_new_thread() function.
- Aside from this, the module only supports a number of low-level ways to work with multithreaded
primitives and share their global data space.
- Additionally, simple lock objects (for example, mutexes and semaphores) are provided for synchronization purposes

Note: The old thread module has been considered deprecated by Python developers for a long
time, mainly because of its rather low-level functions and limited usage.

## The threading module in Python 3

The `threading` module is built on top of the `thread` module, providing easier ways to work with threads through powerful, higher-level APIs.

Main difference between `thread` and `threading`:
- `thread`: considers each thread a function
    - when the `thread.start_new_thread()` is called, it actually takes in a separate function as its
main argument, in order to spawn a new thread
- `threading`: treats each thread that is created as an object
    - designed to be user-friendly for those that come from the object-oriented software development paradigm

In addition, `threading` supports a number of extra methods:
- `threading.activeCount()`: This function returns the number of currently active thread objects in the program
- `threading.currentThread()`: This function returns the number of thread objects in the current thread control from the caller
- `threading.enumerate()`: This function returns a list of all of the currently active thread objects in the program

Following the object-oriented software development paradigm, the threading module also provides a Thread class that supports the object-oriented implementation of threads. The following methods are supported in this class:
- `run()`: This method is executed when a new thread is initialized and started
- `start()`: This method starts the initialized calling thread object by calling the run() method
- `join()`: This method waits for the calling thread object to terminate before continuing to execute the rest of the program
- `isAlive()`: This method returns a Boolean value, indicating whether the calling thread object is currently executing
- `getName()`: This method returns the name of the calling thread object
- `setName()`: This method sets the name of the calling thread object

## Creating a new thread in Python

As mentioned previously, the `threading` module is most likely the most common way of working with threads in Python. Specific situations require use of the thread module and maybe other tools, as well, and it is important for us to be able to differentiate those situations.

### Starting a thread with the thread module

In the thread module, new threads are created to execute functions concurrently. As we have mentioned, the way to do this is by using the `thread.start_new_thread()` function:

```py
thread.start_new_thread(function, args[, kwargs])
```

When this function is called, a new thread is spawned to execute the function specified by
the parameters, and the identifier of the thread is returned when the function finishes executing.
- arguments:
    - function parameter: the name of the function to be executed
    - args parameter: includes the arguments to be passed to the specified function
        - has to be a list or a tuple
    - optional kwargs argument: a separate dictionary of additional keyword arguments

When the thread.start_new_thread() function returns, the thread also terminates silently.

Let's look at an example:

```py
# ch3/example2.py

import _thread as thread
from math import sqrt

def is_prime(x):
    if x < 2:
        print('%i is not a prime number.' % x)

    elif x == 2:
        print('%i is a prime number.' % x)

    elif x % 2 == 0:
        print('%i is not a prime number.' % x)

    else:
        limit = int(sqrt(x)) + 1
        for i in range(3, limit, 2):
            if x % i == 0:
                print('%i is not a prime number.' % x)
                return

        print('%i is a prime number.' % x)

my_input = [2, 193, 323, 1327, 433785907]

for x in my_input:
    thread.start_new_thread(is_prime, (x, ))

a = input('Type something to quit: \n')
print('Finished.')

```

Run the program:

```bash
cd mastering_concurrency_in_python
python Mastering-Concurrency-in-Python/Chapter03/example2.py

```

There is a line of code to take in the user's input at the end:

```py
a = input('Type something to quit: \n')
```

- If you comment this out, the program terminates without printing out any output
    - This is because the entire program finishes before the threads can finish executing
        - When a new thread is spawned via `thread.start_new_thread()`, the program continues, and by the time it reaches the end, any thread that has not finished executing is just terminated (therefore ignored)
            - In this case: All of them!
                - Sometimes, you will see 1 thread finish in time (I was not able to re-produce this)

- This last line of code is a workaround for the `thread` module:
    - It prevents the program from exiting until the user presses a key
        - Strategy: Wait for program to finish executing all threads, then quit manually
    - As you can see, the "Type something to quit:" line was printed out before the output from the is_prime() function
        - this is consistent with the fact that that line is executed before any of the other threads finish executing

`thread` needs unintuitive workarounds, which is part of why it is not preferred.

### Starting a thread with the threading module

To create and customize a new thread using the `threading` module, these are the steps:
1. Define a subclass of the `threading.Thread` class in your program
2. Override the default __init__(self [,args]) method inside of the subclass
    - this adds custom arguments for the class
3. Override the default run(self [,args]) method inside of the subclass
    - this customizes the behavior of the `thread` class when a new thread is initialized and started

(You actually saw an example of this in the first example of this chapter)

In our next example, we will look at the problem of determining whether a specific number
is a prime number.
- This time, we will be implementing a multithreaded program through the `threading` module.

```py
# ch3/example3.py

import threading
from math import sqrt

def is_prime(x):
    if x < 2:
        print('%i is not a prime number.' % x)

    elif x == 2:
        print('%i is a prime number.' % x)

    elif x % 2 == 0:
        print('%i is not a prime number.' % x)

    else:
        limit = int(sqrt(x)) + 1
        for i in range(3, limit, 2):
            if x % i == 0:
                print('%i is not a prime number.' % x)
                return

        print('%i is a prime number.' % x)

class MyThread(threading.Thread):
    def __init__(self, x):
        threading.Thread.__init__(self)
        self.x = x

    def run(self):
        print('Starting processing %i...' % x)
        is_prime(self.x)

my_input = [2, 193, 323, 1327, 433785907]

threads = []

for x in my_input:
    temp_thread = MyThread(x)
    temp_thread.start()

    threads.append(temp_thread)

for thread in threads:
    thread.join()

print('Finished.')

```

What is going on here:
- `MyThread` builds upon `threading.Thread`
    - It takes input of an integer, `x`
- Each instance of `MyThread` is spawned, takes in `x`, and does the following:
    - run(self) is called via `.start()`
        - A message prints that thread `x` is starting
        - `is_prime(self.x)` is called
    - thread is added to the list threads
        - list of threads is needed to join them
- Iterate over the list of threads and call `.join()` on each thread
    - This successfully makes sure each thread finished executing

Run the program:

```bash
cd mastering_concurrency_in_python
python Mastering-Concurrency-in-Python/Chapter03/example3.py

```

Note: Unlike with the `thread` module where we had to use a workaround, we used the `join()` method

## Synchronizing threads

As you saw in the previous examples, the threading module has many advantages over
its predecessor, the thread module, in terms of functionality and high-level API calls. Even
though some recommend that experienced Python developers know how to implement
multithreaded applications using both of these modules, you will most likely be using the
threading module to work with threads in Python. In this section, we will look at using
the threading module in thread synchronization.

### The concept of thread synchronization

Before we jump into an actual Python example, let's explore the concept of synchronization
in computer science.

Sometimes, it is undesirable to have all portions parts of a program run in parallel.
- Most contemporary concurrent programs have sequential portions AND concurrent portions
    - Inside of concurrent portions, there has to be coordination between different threads/processes as well

**Thread/Process Synchronization**: A concept in Computer Science that makes sure that no more than 1 concurrent thread/process can process and execute a particular program portion at a time
- This is known as "critical section"
    - Will discuss more later in Chapter 12 (Starvation) + Chapter 13 (Race Conditions)
    - When a threading is accessing the critical section of the program, the other threads have to wait until that thread is done
- Goal of thread synchronization: Avoid data discrepancies
    - By allowing only ` thread to execute the critical section, you guarantee no data conflicts

### The threading.Lock class

One of the most common ways to apply thread synchronization: locking mechanisms
- In the `threading` module, the `threading.Lock` class provides a simple/intuitive approach to working with locks.

Its main usage includes the following methods:
- threading.Lock(): initializes and returns a new lock object.
- acquire(blocking): When this method is called, all of the threads will run synchronously (that is, only one thread can execute the critical section at a time)
    - The optional argument `blocking` allows us to specify whether the current thread should wait to acquire the lock
        - When `blocking` = 0, the current thread does NOT wait for the lock
            - it simply returns 0 if the lock cannot be acquired by the thread, or 1 otherwise
        - When `blocking` = 1, the current thread blocks and waits for the lock to be released
            - acquires it afterwards
- release(): When this method is called, the lock is released.

### An example in Python

In this example, we will be looking at the Chapter03/example4.py file.
- We will go back to the thread example of counting down from five to one, which we looked at at the beginning of this chapter

In this example, we will be tweaking the MyThread class, as follows:

```py
# ch3/example4.py

import threading
import time

class MyThread(threading.Thread):
    def __init__(self, name, delay):
        threading.Thread.__init__(self)
        self.name = name
        self.delay = delay

    def run(self):
        print('Starting thread %s.' % self.name)
        thread_lock.acquire()
        thread_count_down(self.name, self.delay)
        thread_lock.release()
        print('Finished thread %s.' % self.name)

def thread_count_down(name, delay):
    counter = 5

    while counter:
        time.sleep(delay)
        print('Thread %s counting down: %i...' % (name, counter))
        counter -= 1
```

What is different here:
- The `MyThread` class has a lock object (`thread_lock`) inside of its run function
    - How this works:
        - lock object is acquired before the `thread_count_down()` function is called (ie. when the countdown begins)
        - lock object is released right after it ends
- What we expect to see now:
    - Program will execute the thread separately (ie. the countdowns will take place 1 after another)
        - Before: The executed their countdowns simultaneously

Here is the rest of the logic:

```py
thread_lock = threading.Lock()

thread1 = MyThread('A', 0.5)
thread2 = MyThread('B', 0.5)

thread1.start()
thread2.start()

thread1.join()
thread2.join()


print('Finished.')
```

In Summary: We are initializing the `thread_lock` variable and running 2 separate instances of the `MyThread` class.

Let's run it and look at the output:

```bash
cd mastering_concurrency_in_python
python Mastering-Concurrency-in-Python/Chapter03/example4.py

```

```bash
Starting thread A.
Starting thread B.
Thread A counting down: 5...
Thread A counting down: 4...
Thread A counting down: 3...
Thread A counting down: 2...
Thread A counting down: 1...
Finished thread A.
Thread B counting down: 5...
Thread B counting down: 4...
Thread B counting down: 3...
Thread B counting down: 2...
Thread B counting down: 1...
Finished thread B.
Finished.

```

## Multithreaded priority queue

Queue: Abstract data structure that is a collection of elements in a specific order
- This is a computer science concept that is widely used in both non-concurrent and concurrent
programming

### A connection between real-life and programmatic queues

Queues: Intuitive concept that we can relate to our everyday life
- Example: Standing in line to board a plane
    - Enter in at 1 end, exit at the other end
    - If person A enters the line before person B, person A will also leave the line before person B
        - (unless person B has more priority)
    - Once everyone has boarded the plane, there will be no one left in the line.
        - In other words, the line will be empty
- In computer science terms now:
    - Elements can be added to the end of the queue; this task is called `enqueue`
    - Elements can also be removed from the beginning of the queue; this task is called `dequeue`
    - In a First In First Out (FIFO) queue, the elements that are added first will be removed first (hence, the name FIFO).
        - This is contrary to another common data structure in computer science, called stack, in which the last element that is added will be removed first. This is known as Last In First Out (LIFO).
            - Think of a stack of dishes
    - If all of the elements inside of a queue have been removed, the queue will be empty and there will be no way to remove further elements from the queue.
        - Similarly, if a queue is at the maximum capacity of the number of elements it can hold, there is no way to add any other elements to the queue

![A visualization of the queue data structurealt text](image-10.png)

### The queue module

The `queue` module in Python provides a simple implementation of the queue data structure. Each queue in the `queue.Queue` class can hold a specific amount of element, and can have the following methods as its high-level API:
- `get()`: returns the next element of the calling queue object (and removes it from the queue object)
- `put()`: adds a new element to the calling queue object
- `qsize()`: returns the number of current elements in the calling `queue` object
    - ie. its size
- `empty()`: Boolean, indicates whether the calling queue object is empty
- `full()`: Boolean, indicaties whether the calling queue object is full

## Queuing in concurrent programming

The concept of a queue is even more prevalent in of concurrent programming (especially when we need to implement a fixed number of threads in our program to interact with a varying number of shared resources.)

Previous examples: We have assigned a specific task to a new thread
- What this means: The number of tasks that need to be processed dictates the number of threads our program spawns
    - Example: in `Chapter03/example3.py`, we spawned 5 threads for each of the 5 input numbers

Sometimes: It is undesirable to have as many threads as tasks we need to process
- If we have a lot of tasks, then spawning 1000 threads is inefficient
    - Better answer: Spawn a fixed # of threads to work through the tasks in a cooperative manner
        - This is a "thread pool"

Here is when the concept of a queue comes in!

How we design our structure:
- Pool of threads does not hold any information on the tasks they should execute
- Tasks are stored in a queue (the task queue)
- Items in the task queue are fed to the threads in the pool of threads
- When a task is completed by a member of the pool thread, that worker is freed up
    - If the task queue still has work to be done/elements to be processed, that next element in the queue is sent to the thread/worker that just became available

This diagram further illustrates this setup:

![Queuing in threading](image-11.png)

Let's look at an example in Python - in this example, we will print out all positive factors of a number in a list of positive integers.

We have adjusted the `MyThread` class once again:

```py
class MyThread(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):
        print('Starting thread %s.' % self.name)
        process_queue()
        print('Exiting thread %s.' % self.name)
```

What has changed:
- When a new instance of `MyThread` is initialized and started, the method `process_queue()` is called
    - This attempts to grab the next value of the queue object (`my_queue`)
        - Usually: `.get(block=True)` is a blocking call
            - You will wait at that line of code until a value is entered/is available in the queue
        - `block=False` makes sure that it does not block
            - will raise `queue.Empty` if the queue is empty, and then moves on (given that you handle this error/exception)

Let's look at the rest of the code:

```py
# ch3/example5.py

def process_queue():
    while True:
        try:
            x = my_queue.get(block=False)
        except queue.Empty:
            return
        else:
            print_factors(x)

        time.sleep(1)

def print_factors(x):
    result_string = 'Positive factors of %i are: ' % x
    for i in range(1, x + 1):
        if x % i == 0:
            result_string += str(i) + ' '
    result_string += '\n' + '_' * 20

    print(result_string)


# setting up variables
input_ = [1, 10, 4, 3]

# filling the queue
my_queue = queue.Queue()
for x in input_:
    my_queue.put(x)


# initializing and starting 3 threads
thread1 = MyThread('A')
thread2 = MyThread('B')
thread3 = MyThread('C')

thread1.start() # Start the thread; this calls the run method
thread2.start()
thread3.start()

# joining all 3 threads
thread1.join()
thread2.join()
thread3.join()

print('Done.')

```

What is going on here:
- `process_queue()`: defines logic to look in the queue `my_queue`
    - does 1 of 2 things:
    1. grabs value from queue in non-blocking way and prints the factors, then waits 1 second before trying to retrieve the next item in the queue
    OR
    2. returns nothing, effectively ending the loop and the thread's run method
- `print_factors()`: takes an integer, prints string of factors
- the queue is filled via looping through a basic list
- 3 separate threads are initialized, started via `thread.start()`, and joined via `thread.join()`
    - We chose a fixed number of threads to simulate the design of processing a queue whose size can change independently
    - Note: The class's `run()` method specifies what the thread does when it is started
        - Important: The run method does not execute during the initialization of the class instance (`__init__`)
    - it is the method that is called when the thread's `.start()` method is invoked.

Let's run the code to see what happens:

```bash
cd mastering_concurrency_in_python
python Mastering-Concurrency-in-Python/Chapter03/example5.py

```

Run the program and you will see the following output:

```bash
# > python example5.py
Starting thread A.
Starting thread B.
Starting thread C.
Positive factors of 1 are: 1
____________________
Positive factors of 10 are: 1 2 5 10
____________________
Positive factors of 4 are: 1 2 4
____________________
Positive factors of 3 are: 1 3
____________________
Exiting thread C.
Exiting thread A.
Exiting thread A.
Done.
```

Note: I actually got this... (hmmmm)

```bash
Starting thread A.
Starting thread B.
Positive factors of 1 are: 1
____________________
Positive factors of 10 are: 1 2 5 10
____________________
Starting thread C.
Positive factors of 4 are: 1 2 4
____________________
Positive factors of 3 are: 1 3 
____________________
Exiting thread B.
Exiting thread C.
Exiting thread A.
Done.
```

### Multithreaded priority queue

Priority queue: Abstract data structure similar to the queue + stack
- Each element in the queue has a "priority" associated with it
    - When an element is added to the priority queue, its priority needs to be specified
    - Elements with higher priorities are processed before those with lower priorities
        - Unlike in regular queues, where dequeuing removes the element at the front of the queue/line
- Examples of use: applications that use a definite scoring system/function to determine the priority of its elements
    - bandwidth management
        - prioritized traffic, such as real-time streaming, is processed with the least delay and the least likelihood of being rejected
    - Dijkstra's algorithm
    - best-first search algorithms
        -  implemented to keep track of unexplored routes
            - routes with shorter estimated path lengths are given higher priorities in the queue

## Summary

Summary:
- A thread of execution is the smallest unit of programming commands
- The `threading` module in Python 3 provides an efficient, powerful, and high-level API to work with threads
- Queuing and priority queuing are important data structures; and they are essential concepts in concurrent and parallel programming

---

# Chapter 4 - Using the with Statement in Threads

The `with` statement can cause confusion for notices and experienced programmers alike.
- This chapter explains the idea behind the `with` statement as a context manager
- It also epxlains its usage in concurrent/parallel programming
    - Specifically: Regarding the use of locks while synchronizing threads

The following topics will be covered in this chapter:
- The concept of context management and the options that the with` statement
    - provides as a context manager, specifically in concurrent and parallel programming
- The syntax of the `with` statement and how to use it effectively and efficiently
- The different ways of using the `with` statement in concurrent programming

## Context management

`with` statement: most commonly used as a context manager that manages resources
- essential in concurrent and parallel programming (since resources are shared across different entities in the concurrent or parallel application)

## Starting from managing files

As an experienced Python user, you have probably seen the `with` statement being used to
open and read external files inside Python programs.
- At a lower level: The operatoin of opening an external file in Python consumes a resource
    - resource: file descriptor
    - operating system sets a limit on this resource
        - What this means: upper limit on how many files a single process can open simultaneously

Let's look at a quick example to illustrate:

```py
# ch4/example1.py

n_files = 254
files = []

# method 1
for i in range(n_files):
    files.append(open('output1/sample%i.txt' % i, 'w'))
```

Run the program:

```bash
cd mastering_concurrency_in_python
cd Mastering-Concurrency-in-Python
cd Chapter04
python example1.py

```

It runs, and there are no prints or anything. OK...

Now, try running it with `n_files` = 10000! This is similar to what you should see:

```bash
C:\Users\Myles\mastering_concurrency_in_python\Mastering-Concurrency-in-Python\Chapter04>python example1.py
Traceback (most recent call last):
  File "C:\Users\Myles\mastering_concurrency_in_python\Mastering-Concurrency-in-Python\Chapter04\example1.py", line 
8, in <module>
OSError: [Errno 24] Too many open files: 'output1/sample8189.txt'
```

What is going on here:
-  File descriptor leakage: Your laptop/device can only handle a certain amount of opened files at the same time
    - On LINUX/UNIX-like systems, print `ulimit -n` to see how many files (I got 1024 via my Ubuntu on WSL)
- Can lead to a number of problems:
    -  unauthorized I/O operations on open files

Example 2 takes care of this properly too. (Run the code if you'd like to see)

## The with statement as a context manager

In real-life applications: It is easy to mismanage opened files in your programs (ie. by forgetting to close them)
- It can also be impossible to tell whether the program has processed a file
    - Makes it difficult to close the file appropriately
    - This situation is even more common in concurrent and parallel programming, where the
order of execution between different elements changes frequently

Solution 1: use a `try...except...finally` block every time we want to interact with an
external file
-  still requires the same level of management and significant overhead
- does not provide a good improvement in the ease and readability of our
programs either

Solution 2 (better): `with` statement
- gives us a simple way of ensuring that all opened files are properly managed and cleaned up when the program finishes using them
    - most notable advantage: even if the code is successfully executed or it returns an error, the with statement always handles and manages the opened files appropriately (via context)

Let's look at an example:

```py
for i in range(n_files):
    with open('output1/sample%i.txt' % i, 'w') as f:
        files.append(f)

```

Another pro: the with statement helps us indicate the scope of certain variables
- in this case, the variables that point to the opened files—and hence, their context
- in this example: `f` indicates the current opened file within the `with` block at each iteration of the `for` loop
    - as soon as program exits the `with` block, you can no longer access `f`
        - guarantees that all cleanup associated with a file descriptor happens appropriately
            - hence why it is called a context manager

## The syntax of the with statement

Purpose: wrapping the execution of a block with methods defined by a context manager

```bash
with [expression] (as [target]):
    [code]
```

Note: `as [target]` is not required
- Another note: `with` statement can handle more than 1 item on the same line
- Specifically, the context managers created are treated as if multiple with statements were nested inside one another

Look at this example:

```bash
with [expression1] as [target1], [expression2] as [target2]:
    [code]
```

This is interpreted as follows:

```bash
with [expression1] as [target1]:
    with [expression2] as [target2]:
        [code]

```

## The with statement in concurrent programming

These are simple examples - opening and closing files does not resemble concurrency much at all.

As a context manager, is not only used to manage file descriptors, but most resources in general.
-  if you actually found managing lock objects from the `threading.Lock()` class similar to managing external files while going through Chapter 2 - Amdahl's Law, then this is where the comparison comes in handy

Refresher: locks are used in concurrent and parallel programming to synchronize threads in a multithreaded application
- goal: prevent one thread from accessing the critical session at the same time as another
- unfortunately, locks are a common source of deadlock
    - deadlock: when a thread acquires a lock but never releases it (due to an unhandled occurrence)
        - this stops the entire program!

## Example of deadlock handling

Let's take a look at this example:

```py
# ch4/example2.py

from threading import Lock

my_lock = Lock()

# induces deadlocks
def get_data_from_file_v1(filename):
    my_lock.acquire()

    with open(filename, 'r') as f:
        data.append(f.read())

    my_lock.release()

# handles exceptions
def get_data_from_file_v2(filename):
    with my_lock, open(filename, 'r') as f:
        data.append(f.read())

data = []

try:
    get_data_from_file_v1('output2/sample0.txt')
    #get_data_from_file_v2('output2/sample0.txt')
except FileNotFoundError:
    print('File could not be found...')

my_lock.acquire()
print('Lock can still be acquired.')

```

Run the code:

```bash
cd mastering_concurrency_in_python
cd Mastering-Concurrency-in-Python
cd Chapter04
python example2.py

```

What is going on here:
- We declare a lock `my_lock`
- We run a function `get_data_from_file_v1` try to read a file (that doesn't exist)
    - Lock is acquired via `my_lock.acquire()` ie. the thread takes over this lock
    - Error occurs reading file (since it doesn't exist)
    - Lock does not get released via `my_lock.release()` due to error
        - We know this because the print statement at the end of the program never runs
            - Deadlock was induced!

My program is stuck - look at this:

![Stuck program](image-12.png)

Let's try with `with` - Comment out line 24 ie. `get_data_from_file_v1('output2/sample0.txt')` and uncomment line 25 and you will now get the following:

```bash
File could not be found...
Lock can still be acquired.
```

Since Lock objects are context managers, simply using `with my_lock:` ensures that
the lock object is acquired and released appropriately
- even if an exception is encountered inside the block!

## Summary

The `with` statement in Python offers an intuitive/convenient way to manage resources, while still ensuring that errors and exceptions are handled correctly.
- This ability to manage resources is even more important in concurrent and parallel programming
    - various resources are shared across different entities
        - specifically, by using the `with` statement with `threading.Lock` objects that are used to synchronize different threads in a multithreaded application.

Aside from better error handling and guaranteed cleanup tasks, the `with` statement also
provides extra readability from your programs
- one of the strongest features that Python offers its developers

In the next chapter, we will be discussing one of the most popular uses of Python at the
moment: web-scraping applications.
- We will look at the concept and the basic idea behind web scraping, the tools that Python provides to support web scraping, and how concurrency will significantly help your web-scraping applications.

---

# Chapter 5 - Concurrent Web Requests

This chapter will focus on the application of concurrency in making web requests.
- Intuitively: making requests to a web page to collect information about it is independent to applying the same task to another web page.
- Concurrency, specifically threading in this case, therefore can be a powerful tool that provides a significant speedup in this process.

In this chapter, we will learn the fundamentals of web requests and how to interact with
websites using Python.

We will also see how concurrency can help us make multiple requests in an efficient way.

Finally, we will look at a number of good practices in web requests.

In this chapter, we will cover the following concepts:
- The basics of web requests
- The requests module
- Concurrent web requests
- The problem of timeout
- Good practices in making web requests

## The basics of web requests

