import fileinput
import codecs
import os
import numpy as np

from os.path import join

from itertools import ifilter, imap, izip, islice


class Producer(object):
    def __init__(self, iterable):
        self.it = iterable.__iter__()

    def __iter__(self):
        return self.it
        # # try:
        # while True:
        #     yield self.next()
        # # except Exception:
        # #     raise StopIteration

    def next(self):
        return self.it.next()

    def plug_to(self, transducer):
        return transducer.plug_in(self)

    def write_in(self, filename, codec='utf-8'):
        FileWriter(filename, codec).plug_in(self)

    def write_to(self, out):
        Writer(self, out).plug_in(self)

    def zip(self, other):
        return Producer(izip(self, other))

    def filter(self, requirement):
        return Producer(ifilter(requirement, self))

    def map(self, f):
        return Producer(imap(f, self))

    def fold(self, zero, combine):
        acc = zero
        for x in self:
            acc = combine(acc, x)
        return x

    def then(self, other):
        def iter():
            for x in self:
                yield x
            for x in other:
                yield x
        return Producer(iter())

    def extend(self, other):
        return self.then(other)

    def foreach(self, f):
        for x in self:
            f(x)

    def flatmap(self, f):
        def flat():
            for x in self:
                for y in f(x):
                    yield y

        return Producer(flat())

    def to_list(self):
        return list(self)

    def to_dict(self):
        return self.plug_to(DictWriter(dict()))

    def take(self, n):
        return Producer(islice(self, n))

    def drop(self, n):
        return Producer(islice(self, n, None))

    def distincts(self):
        return Producer(DistinctPlugger().plug_in(self))

# class MultipleProducer(Producer):
#     def __init__(self, iterables):
#         self.n = len(iterables)
#         self.its = map(lambda it: it.__iter__(), iterables)

#     def next(self):
#         return [it for it in self.its]

#     def map(self, fs):
#         return MultipleProducer([imap(f, it) for f, it in zip(fs, self.its)])

#     def plug_to(self, processors):
#         return MultipleProducer([p1.plug_in(p0) for (p0, p1) in zip(self.iterables, processors)])

#     def zip(self, other):
#         return Producer(izip(self.it, other.it))

#     def filter(self, requirement):
#         return Producer(ifilter(requirement, self.it))

#     def __map(self, f, *args):
#         pass


class LoopProducer(Producer):
    def __init__(self, l):
        super(LoopProducer, self).__init__()
        self.l = l
        self.i = 0
        self.n = len(l)

    def __iter__(self):
        while(True):
            for x in self.l:
                yield x

    def next(self):
        x = self.l[self.i]
        self.i = (self.i + 1) % self.n
        return x


def RandProducer(n):
    return StillProducer(lambda: np.random.randint(n))


class StillProducer(Producer):
    def __init__(self, next):
        self.next = next
        super(StillProducer, self).__init__(self)

    def __iter__(self):
        while(True):
            yield self.next()


def FileReader(filename, codec='utf-8', prefix=''):
    if prefix is not '':
        filename = join(prefix, filename)
    return Producer(codecs.open(filename, 'r', codec)).map(lambda s: s[:-1])


def FilesReader(filenames, codec='utf-8', prefix=''):
    if prefix is not '':
        filenames = map(lambda f: join(prefix, f), filenames)
    f = fileinput.FileInput(filenames, openhook=fileinput.hook_encoded(codec))
    return Producer(f).map(lambda s: s[:-1])


def DirectoryReader(dirname, recursive=False, codec='utf-8'):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(dirname):
        files += map(lambda f: join(dirpath, f), filenames)
        if not recursive:
            break
    return FilesReader(files, codec)


class Stream(Producer):
    def __init__(self, iterable):
        self.iterable = iterable
        self.it = self.iterable.__iter__()

    def reset(self):
        self.it = self.iterable.__iter__()


def Stdin():
    def read_stdin():
        while True:
            l = raw_input()
            print l
            if l == chr(4):  # EOT character
                break
            yield l
    return Producer(read_stdin())


class Transducer(object):
    def plug_in(self, producer):
        return producer

    def plug_to(self, consumer):
        return consumer.chain_before(self)

    def process_all(self, iterable):
        return self.plug_in(Producer(iterable))

    def write_in(self, filename, codec='utf-8'):
        return self.plug_to(FileWriter(filename, codec))

    def write_to(self, out):
        return self.plug_to(Writer(self, out))

    def zip(self, other):
        return Plugger(lambda ps: self.plug_in(ps[0]).zip(other.plug_in(ps[1])))

    def filter(self, requirement):
        return Plugger(lambda producer: self.plug_in(producer).filter(requirement))

    def chain(self, other):
        return Plugger(lambda producer: other.plug_in(self.plug_in(producer)))

    def map(self, f):
        return Plugger(lambda producer: self.plug_in(producer).map(f))

    def distincts(self):
        return self.chain(DistinctPlugger())


class Plugger(Transducer):
    def __init__(self, plug_in):
        super(Plugger, self).__init__()
        self.plug_in = plug_in


class Processor(Transducer):
    def __init__(self, process):
        self.process = process

    def __call__(self, *args):
        return self.process(*args)

    def plug_in(self, producer):
        return producer.map(self.process)

    def map(self, f):
        return Processor(lambda x: f(self.process(x)))

    def chain(self, other):
        if isinstance(other, Processor):
            return Processor(lambda x: other.process(self.process(x)))
        else:
            return Plugger(lambda producer: other.plug_in(self.plug_in(producer)))


class DistinctPlugger(Transducer):
    """docstring for DistinctPlugger"""
    def __init__(self):
        self.distincts = set([])
        super(DistinctPlugger, self).__init__()

    def plug_in(self, xs):
        for x in xs:
            if x not in self.distincts:
                self.distincts.add(x)
                yield x


def Switcher(switch, processors):
    return Processor(lambda x: processors[switch(x)].process(x))


def BeforeAndAfter(before, mapper, after):
    return Processor(lambda x: after(x, mapper.process(before(x))))


def Pipeline(*transducers):
    def plug_in(producer):
        for t in transducers:
            producer = producer.plug_in(t)
        return producer
    return Plugger(plug_in)


class Consumer(object):
    """docstring for Consumer"""
    def close(self):
        pass

    def plug_in(self, producer):
        raise NotImplementedError

    # def consume(self, x):
    #     raise NotImplementedError


# class ChainedBefore(Consumer):
#     def __init__(self, transducer, consumer):
#         super(ChainedBefore, self).__init__()
#         self.preproc = transducer
#         self.consumer = consumer

#     def plug_in(self, producer):
#         p = self.preproc.plug_in(producer)
#         for x in p:
#             self.consume(x)
#         return self.close()

#     def consume(self, x):
#         self.consumer.consume(x)

#     def close(self):
#         return self.consumer.close()


class UnitConsumer(Consumer):
    def __init__(self, consume):
        super(UnitConsumer, self).__init__()
        self.consume = consume

    def plug_in(self, producer):
        for x in producer:
            self.consume(x)
        return self.close()

    def map(self, f):
        return UnitConsumer(lambda x: self.consume(f(x)))

    def filter(self, f):
        def consume(x):
            if f(x):
                self.consume(x)
        return UnitConsumer(consume)


class PluggableConsumer(Consumer):
    def __init__(self, plug_in):
        super(PluggableConsumer, self).__init__()
        self.plug_in_f = plug_in

    def plug_in(self, producer):
        self.plug_in_f(producer)
        self.close()


class Dispatcher(UnitConsumer):
    def __init__(self, switch, consumers, common=None):
        self.switch = switch
        self.consumers = consumers
        self.common = common
        super(Dispatcher, self).__init__(self.consume)

    def consume(self, x):
        if self.common is not None:
            self.consumers[self.switch(x)].consume(self.common(x))
        else:
            self.consumers[self.switch(x)].consume(x)

    def close(self):
        for out in self.consumers:
            out.close()


class AutoDispatcher(UnitConsumer):
    def __init__(self, switch, init=None, common=None):
        self.switch = switch
        self.consumers = {}
        self.init = init if init is not None else lambda f: FileWriter(f)
        self.common = common
        super(AutoDispatcher, self).__init__(self.consume)

    def consume(self, x):
        f = self.switch(x)
        if f not in self.consumers:
            self.consumers[f] = self.init(f)
        if self.common is not None:
            x = self.common(x)
        self.consumers[f].consume(x)

    def close(self):
        for out in self.consumers.itervalues():
            out.close()


class Writer(UnitConsumer):
    def __init__(self, out, write):
        self.out = out
        self.write = write
        super(Writer, self).__init__(self.consume)

    def consume(self, data):
        self.write(data, self.out)

    def close(self):
        self.out.close()


def FileWriter(filename, codec='utf-8', before='', after='\n', write=None):
    if write is None:
        def write2(x, out):
            out.write(before + unicode(x) + after)
    else:
        def write2(x, out):
            out.write(before)
            write(x, out)
            out.write(after)

    return Writer(codecs.open(filename, 'w', codec), write2)


def ListWriter(l):
    return UnitConsumer(lambda x: l.append(x))


class DictWriter(UnitConsumer):
    def __init__(self, d):
        self.d = d
        super(DictWriter, self).__init__(self.consume)

    def consume(self, x):
        self.d[x[0]] = x[1]

    def close(self):
        return self.d


class WordCounter(UnitConsumer):
    def __init__(self, d={}):
        self.d = d
        super(WordCounter, self).__init__(self.consume)

    def consume(self, line):
        for w in line.split():
            if w in self.d:
                self.d[w] += 1
            else:
                self.d[w] = 1

    def close(self):
        return self.d


class VocabReader(UnitConsumer):
    def __init__(self):
        self.vocab = set([])
        super(VocabReader, self).__init__(self.consume)

    def consume(self, line):
        for w in line.split():
            self.vocab.add(w)

    def close(self):
        return self.vocab

