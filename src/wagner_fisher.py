# Copyright (c) 2013-2016 Kyle Gorman
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# wagnerfischer.py: efficient computation of Levenshtein distance and
# all optimal alignments with arbitrary edit costs. The algorithm for
# computing the dynamic programming table used has been discovered many
# times, but is described most clearly in:
#
# R.A. Wagner & M.J. Fischer. 1974. The string-to-string correction
# problem. Journal of the ACM, 21(1): 168-173.
#
# Wagner & Fischer also describe an algorithm ("Algorithm Y") to find the
# alignment path (i.e., list of edit operations involved in the optimal
# alignment), but it it is specified such that in fact it only generates
# one such path, whereas many such paths may exist, particularly when
# multiple edit operations have the same cost. For example, when all edit
# operations have the same cost, there are two equal-cost alignments of
# "TGAC" and "GCAC":
#
#     TGAC     TGxAC
#     ss==     d=i==
#     GCAC     xGCAC
#
# However, all such paths can be generated efficiently, as follows. First,
# the dynamic programming table "cells" are defined as tuples of (partial
# cost, set of all operations reaching this cell with minimal cost). As a
# result, the completed table can be thought of as an unweighted, directed
# graph (or FSA). The bottom right cell (the one containing the Levenshtein
# distance) is the start state and the origin as end state. The set of arcs
# are the set of operations in each cell as arcs. (Many of the cells of the
# table, those which are not visited by any optimal alignment, are under
# the graph interpretation unconnected vertices, and can be ignored. Every
# path between the bottom right cell and the origin cell is an optimal
# alignment. These paths can be efficiently enumerated using breadth-first
# traversal. The trick here is that elements in deque must not only contain
# indices but also partial paths. Averaging over all such paths, we can
# come up with an estimate of the number of insertions, deletions, and
# substitutions involved as well; in the example above, we say S = 1 and
# D, I = 0.5.
#
# Thanks to Christoph Weidemann (ctw@cogsci.info), who added support for
# arbitrary cost functions.


import collections
import doctest
import pprint


# Default cost functions.


def INSERTION(A, cost=1):
    return cost


def DELETION(A, cost=1):
    return cost


def SUBSTITUTION(A, B, cost=1):
    return cost


Trace = collections.namedtuple("Trace", ["cost", "ops"])


class WagnerFischer(object):

    """
    An object representing a (set of) Levenshtein alignments between two
    iterable objects (they need not be strings). The cost of the optimal
    alignment is scored in `self.cost`, and all Levenshtein alignments can
    be generated using self.alignments()`.
    Basic tests:
    >>> WagnerFischer("god", "gawd").cost
    2
    >>> WagnerFischer("sitting", "kitten").cost
    3
    >>> WagnerFischer("bana", "banananana").cost
    6
    >>> WagnerFischer("bana", "bana").cost
    0
    >>> WagnerFischer("banana", "angioplastical").cost
    11
    >>> WagnerFischer("angioplastical", "banana").cost
    11
    >>> WagnerFischer("Saturday", "Sunday").cost
    3
    IDS tests:
    >>> WagnerFischer("doytauvab", "doyvautab").IDS() == {"S": 2.0}
    True
    >>> WagnerFischer("kitten", "sitting").IDS() == {"I": 1.0, "S": 2.0}
    True
    Detect insertion vs. deletion:
    >>> thesmalldog = "the small dog".split()
    >>> thebigdog = "the big dog".split()
    >>> bigdog = "big dog".split()
    >>> sub_inf = lambda A, B: float("inf")
    # Deletion.
    >>> wf = WagnerFischer(thebigdog, bigdog, substitution=sub_inf)
    >>> wf.IDS() == {"D": 1.0}
    True
    # Insertion.
    >>> wf = WagnerFischer(bigdog, thebigdog, substitution=sub_inf)
    >>> wf.IDS() == {"I": 1.0}
    True
    # Neither.
    >>> wf = WagnerFischer(thebigdog, thesmalldog, substitution=sub_inf)
    >>> wf.IDS() == {"I": 1.0, "D": 1.0}
    True
    """

    # Initializes pretty printer (shared across all class instances).
    pprinter = pprint.PrettyPrinter(width=75)

    def __init__(self, A, B, insertion=INSERTION, deletion=DELETION,
                 substitution=SUBSTITUTION):
        # Stores cost functions in a dictionary for programmatic access.
        self.costs = {"I": insertion, "D": deletion, "S": substitution}
        # Initializes table.
        self.asz = len(A)
        self.bsz = len(B)
        self._table = [[None for _ in range(self.bsz + 1)] for
                       _ in range(self.asz + 1)]
        # From now on, all indexing done using self.__getitem__.
        ## Fills in edges.
        self[0][0] = Trace(0, {"O"})  # Start cell.
        for i in range(1, self.asz + 1):
            self[i][0] = Trace(self[i - 1][0].cost + self.costs["D"](A[i - 1]),
                               {"D"})
        for j in range(1, self.bsz + 1):
            self[0][j] = Trace(self[0][j - 1].cost + self.costs["I"](B[j - 1]),
                               {"I"})
        ## Fills in rest.
        for i in range(len(A)):
            for j in range(len(B)):
                # Cleans it up in case there are more than one check for match
                # first, as it is always the cheapest option.
                if A[i] == B[j]:
                    self[i + 1][j + 1] = Trace(self[i][j].cost, {"M"})
                # Checks for other types.
                else:
                    costD = self[i][j + 1].cost + self.costs["D"](A[i])
                    costI = self[i + 1][j].cost + self.costs["I"](B[j])
                    costS = self[i][j].cost + self.costs["S"](A[i], B[j])
                    min_val = min(costI, costD, costS)
                    trace = Trace(min_val, set())
                    # Adds _all_ operations matching minimum value.
                    if costD == min_val:
                        trace.ops.add("D")
                    if costI == min_val:
                        trace.ops.add("I")
                    if costS == min_val:
                        trace.ops.add("S")
                    self[i + 1][j + 1] = trace
        # Stores optimum cost as a property.
        self.cost = self[-1][-1].cost

    def __repr__(self):
        return self.pprinter.pformat(self._table)

    def __iter__(self):
        for row in self._table:
            yield row

    def __getitem__(self, i):
        """
        Returns the i-th row of the table, which is a list and so
        can be indexed. Therefore, e.g.,  self[2][3] == self._table[2][3]
        """
        return self._table[i]

    # Stuff for generating alignments.

    def _stepback(self, i, j, trace, path_back):
        """
        Given a cell location (i, j) and a Trace object trace, generate
        all traces they point back to in the table
        """
        for op in trace.ops:
            if op == "M":
                yield i - 1, j - 1, self[i - 1][j - 1], path_back + ["M"]
            elif op == "I":
                yield i, j - 1, self[i][j - 1], path_back + ["I"]
            elif op == "D":
                yield i - 1, j, self[i - 1][j], path_back + ["D"]
            elif op == "S":
                yield i - 1, j - 1, self[i - 1][j - 1], path_back + ["S"]
            elif op == "O":
                return  # Origin cell, so we"re done.
            else:
                raise ValueError("Unknown op {!r}".format(op))

    def alignments(self):
        """
        Generate all alignments with optimal-cost via breadth-first
        traversal of the graph of all optimal-cost (reverse) paths
        implicit in the dynamic programming table
        """
        # Each cell of the queue is a tuple of (i, j, trace, path_back)
        # where i, j is the current index, trace is the trace object at
        # this cell, and path_back is a reversed list of edit operations
        # which is initialized as an empty list.
        queue = collections.deque(self._stepback(self.asz, self.bsz,
                                                 self[-1][-1], []))
        while queue:
            (i, j, trace, path_back) = queue.popleft()
            if trace.ops == {"O"}:
                # We have reached the origin, the end of a reverse path, so
                # yield the list of edit operations in reverse.
                yield path_back[::-1]
                continue
            queue.extend(self._stepback(i, j, trace, path_back))

    def IDS(self):
        """
        Estimates insertions, deletions, and substitution _count_ (not
        costs). Non-integer values arise when there are multiple possible
        alignments with the same cost.
        """
        npaths = 0
        opcounts = collections.Counter()
        for alignment in self.alignments():
            # Counts edit types for this path, ignoring "M" (which is free).
            opcounts += collections.Counter(op for op in alignment if op != "M")
            npaths += 1
        # Averages over all paths.
        return collections.Counter({o: c / npaths for (o, c) in
                                    opcounts.items()})


if __name__ == "__main__":
    doctest.testmod()