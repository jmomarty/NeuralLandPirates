import numpy as np
import cPickle as pickle
import sys
from DCNN import CNN
import gensim

MIKOLOV = "test2-sg-hs-30"
# MIKOLOV = "data/words_glove/vectors.840B.300d"
# MIKOLOV = "save-model/test2_miko"


def norm(v):
    return np.linalg.norm(v)


def pickle_mikolov(f, out, sep=' '):
    dico = {}
    for line in f:
        parts = line.strip().split(sep)
        if len(parts) > 1:
            dico[parts[0]] = np.array(parts[1:]).astype(float)
    pickle.dump(dico, out, -1)
    return dico

# pickle_mikolov(open(MIKOLOV + '.txt', 'r'), open(MIKOLOV + '.pkl', 'wb'), '\t')


def load_mikolov(filename):
    return gensim.models.Word2Vec.load_word2vec_format(filename + '.bin', binary=True)


def rejected(vec, base):
    res = np.copy(vec)
    for b in base:
        res -= vec.dot(b) * b
    return res


def projected(vec, base):
    a = np.zeros(vec.shape)
    for b in base:
        a += vec.dot(b) * b
    return a


def average(vecs):
    a = np.zeros(vecs[0].shape)
    for v in vecs:
        a += v
    a /= len(vecs)
    return a


def negate(base):
    return []


def intersect(base1, base2):
    inter = []
    for b in base1:
        inter.append(projected(b, base2))
    for b in base2:
        inter.append(projected(b, base1))
    return construct_base(inter)


def add(base1, base2):
    for v in base1:
        base2 = orthonormalize(v, base2)
    return base2


def orthonormalize(vec, base=None):
    if not base:
        return [vec / norm(vec)]
    else:
        vec = rejected(vec, base)
        n = norm(vec)
        if n > 0.0000001:
            base.append(vec / n)
        return base


def construct_base(vectors):
    base = []
    for v in vectors:
        base = orthonormalize(v, base)
    return base


def embed(coordinates, base):
    a = np.zeros(base[0].shape)
    for x, v in zip(coordinates, base):
        a += v * x
    return a


def coordinates(vec, base):
    a = np.zeros(len(base))
    for i, b in enumerate(base):
        a[i] += vec.dot(b)
    return a


def find_orthogonal(miko, base, n=1):
    score = [(-1, "__not_found__")]
    for (w, v) in miko.iteritems():
        sim = norm(rejected(v, base)) / norm(v)
        if sim > score[0][0]:
            score.append((sim, w))
            score.sort()
            score = score[-n:]
    return score


def find_closest_words(miko, v0, n=1):
    score = [(-1, "__not_found__")]
    v0_norm = norm(v0)
    for (w, v1) in miko.iteritems():
        sim = v0.dot(v1) / norm(v1) / v0_norm
        if sim > score[0][0]:
            score.append((sim, w))
            score.sort()
            score = score[-n:]
    return score


def find_neighbor(miko, word, n=10):
    v = miko.get(word)
    if v is None:
        print "Unknown word:", word
        return []
    else:
        return find_closest_words(miko, v, n)


def print_score(score):
    for s, w in score[::-1]:
        print w, '\t\t', s


def evaluate_on_relations(miko, relations, capitalize=False):
    (score, total) = (0, 0)
    with open(relations, 'r') as lines:
        for line in lines.readlines():
            line = line.strip()
            if capitalize:
                line = line.title()
            sc = evaluate_on_relation(miko, line.split())
            score += sc[0]
            total += sc[1]

    res = 100.0 * score / total
    print "Score on", relations, "= %4f" % res
    return res


def evaluate_on_relation(miko, words):
    vectors = map(lambda x: miko.get(x), words)
    if all(map(lambda x: x is not None, vectors)):
        a = vectors[0] - vectors[1] + vectors[3]
        b = find_closest_words(miko, a)[-1][1]
        if b == words[2]:
            print "right for:", words
            return (1, 1)
        else:
            print "wrong for:", words
            return (0, 1)
    else:
        print "skipped:", words
        return (0, 0)


def get_arg(expr):
    i = expr.find("(")
    j = expr.rfind(")")
    return expr[i + 1:j].strip()


def split_strip(expr, sep):
    return map(str.strip, expr.split(sep))


def parse_on(expr, operators, opening='(', closing=')', unary=''):
    for op in operators:
        opened = 0
        for i, c in enumerate(expr):
            if opened == 0 and i > 0 and expr.startswith(op, i):
                return expr[:i], op, expr[i + len(op):]
            elif c in opening:
                opened += 1
            elif c in closing:
                opened -= 1

    for o, c in zip(opening, closing):
        if expr.startswith(o) and expr.endswith(c):
            return '', o + c, expr[1:-1].strip()

    for op in unary:
        if expr.startswith(op, 0):
            return '', op, expr[len(op):]

    return '', '', expr


def split_on(expr, sep, opening='(', closing=')'):
    opened = 0
    res = []
    i0 = 0
    for i, c in enumerate(expr):
        if opened == 0 and expr.startswith(sep, i):
            res.append(expr[i0:i])
            i0 = i + len(sep)
        elif c in opening:
            opened += 1
        elif c in closing:
            opened -= 1
    res.append(expr[i0:])
    return res


def getOrThrow(mem, key):
    x = mem.get(key)
    if x is None:
        raise ParseException("key not found: " + key)
    return x


class ParseException(Exception):
    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return "ParseException(%s)" % self.expr


class Word2VecRepl(object):
    def __init__(self, dico):
        self.dico = dico
        self.mem_base = {}
        self.mem_vec = {}
        self.n_neighbors = 10
        self.combinator = None

    def get(self, word):
        try:
            return self.dico[word]
        except Exception:
            return None

    def set_combinator(self, c):
        self.combinator = c

    def combine(self, x, y):
        if self.combinator is None:
            return (x + y) / 2
        else:
            return self.combinator.combine(x, y)

    def print_vec(self, v):
        print "norm:", norm(v)
        print_score(find_closest_words(self.dico, v, self.n_neighbors))

    def print_base(self, b):
        print "base of dim:", len(b)
        for vec in b:
            print_score(find_closest_words(self.dico, vec, 1))
        print "orthogonal candidates:"
        print_score(find_orthogonal(self.dico, b, self.n_neighbors))

    def read_base(self, expr):
        expr = expr.strip()
        print "reading base in:", expr
        left, op, right = parse_on(expr, ['+', '&'], '([', ')]', '!$')
        if left == '':
            if op == '()':
                return self.read_vec(right)
            elif op == '[]':
                return construct_base(map(self.read_vec, split_on(right, ',')))
            elif op == '!':
                return negate(self.read_base(right))
            elif op == '$':
                return getOrThrow(self.mem_base, right)
            else:
                raise ParseException(expr)
        else:
            if op == '+':
                return construct_base(self.read_base(left) + self.read_base(right))
            else:
                raise ParseException(expr)

    def read_vec(self, expr):
        expr = expr.strip()
        print "reading vec in:", expr
        left, op, right = parse_on(expr, list('+-/_') + ['%', '>>'], '({[', ')}]', '-$')
        if left == '':
            if op == '()':
                return self.read_vec(right)
            elif op == '{}':
                return average(map(self.read_vec, split_on(right, ',')))
            elif op == '-':
                return - self.read_vec(right)
            elif op == '$':
                return getOrThrow(self.mem_vec, right)
            elif op == '' and right.find(' ') < 0:
                return getOrThrow(self.dico, right)
            else:
                raise ParseException(expr)
        elif op == '+':
            return self.read_vec(left) + self.read_vec(right)
        elif op == '-':
            return self.read_vec(left) - self.read_vec(right)
        elif op == '%':
            r = self.read_base_or_vec(right)
            l = self.read_vec(left)
            return rejected(l, r)
        elif op == '/':
            return projected(self.read_vec(left), self.read_base_or_vec(right))
        elif op == '_':
            return self.combine(self.read_vec(left), self.read_vec(right))
        elif op == '>>':
            return embed(self.read_array(left), self.read_base(right))
        else:
            raise ParseException(expr)

    def read_array(self, expr):
        expr = expr.strip()
        print "reading array in:", expr
        left, op, right = parse_on(expr, '*', '([', ')]', '')
        if left == '':
            if op == '()':
                return self.read_vec(right)
            if op == '[]':
                return np.array(split_strip(right, ',')).astype(float)
            else:
                raise ParseException(expr)
        else:
            if op == '*':
                return coordinates(self.read_vec(left), self.read_base(right))
            else:
                raise ParseException(expr)

    def evaluate_expr(self, expr):
        if expr.find('=') > 0:
            name, expr = expr.split('=')
            self.read_assign(expr, name=name.strip())
        else:
            self.read_assign(expr, verbose=True)

    def read_assign(self, expr, name=None, verbose=False):
        try:
            b = self.read_base(expr)
            if name is not None:
                self.mem_base[name] = b
                print "stored as base in", name
            elif verbose:
                self.print_base(b)
            return b
        except ParseException:
            try:
                print "No base found"
                v = self.read_vec(expr)
                if name is not None:
                    self.mem_vec[name] = v
                    print "stored as vec in", name
                elif verbose:
                    self.print_vec(v)
                return v
            except ParseException:
                print "No vec found"
                a = self.read_array(expr)
                if name is not None:
                    self.mem_vec[name] = a
                    print "stored as vec in", name
                elif verbose:
                    self.print_vec(a)
                return a

    def read_base_or_vec(self, expr):
        try:
            b = self.read_base(expr)
            return b
        except ParseException:
            print "No base found"
            v = self.read_vec(expr)
            return [v / norm(v)]

    def run(self):
        done = False
        while not done:
            sys.stdout.write('> ')
            l = raw_input()
            try:
                if l == 'q()' or l == 'quit()':
                    done = True
                    break
                elif l.startswith("neighbors("):
                    self.n_neighbors = int(get_arg(l))
                    print "set the numbers of neighbors to display to:", self.n_neighbors
                elif l.startswith("construct_base("):
                    base = self.read_base(get_arg(l))
                    words = [get_arg(l)]
                    while len(base) < 10:
                        s, w = find_orthogonal(self.dico, base, 1)[0]
                        base = orthonormalize(self.read_vec(w), base)
                        words.append(w)
                        print words
                    self.print_base(base)
                # elif l.startswith("predict("):
                #     words = get_arg(l).split(" ")
                #     words = sentence2matrix(words)


                elif l == '':
                    pass
                else:
                    self.evaluate_expr(l)

            except Exception, e:
                print e

# repl(self.dico)


def main(filename, evaluate=None):
    print "loading mikolov ..."
    dico = load_mikolov(MIKOLOV)
    print "done"

    if evaluate is None:
        Word2VecRepl(dico).run()
    else:
        evaluate_on_relations(dico, evaluate)

if __name__ == '__main__':
    vocab = ['orange', 'bleu', 'jus', 'pomme', 'sfr']
    dico = dict(map(lambda x: (x, np.random.randint(-2, 3, 5)), vocab))
    # filename = 'data/words_mikolov/dict_vectors'
    # dico = load_mikolov(filename)

    repl = Word2VecRepl(dico)
    repl.run()
