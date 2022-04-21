#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

import doctest
import re
import sys


class Wordle(object):

  def __init__(self, depth: int):
    self.depth = depth
    self.out = set()
    self.hit = ['_'] * depth
    self.blow = [''] * depth
    self.stage = 0

  def add(self, guess: str, response: str):
    d = str(self.depth)
    self.stage += 1
    if not isinstance(guess, str) or not re.fullmatch(r'[a-z]{' + d + '}', guess):
      raise ValueError(f"guess: expected `[a-z]{self.depth}`")
    if not isinstance(response, str) or not re.fullmatch(r'[ox_ -]{' + d + '}', response):
      raise ValueError(f"response: expected `[ox_- ]{self.depth}`")
    for i in range(self.depth):
      if response[i] == 'o':  # hit
        if self.hit[i] != '_' and self.hit[i] != guess[i]:
          raise ValueError("invalid response")
        self.hit[i] = guess[i]
      elif response[i] == 'x':  # blow
        self.blow[i] += guess[i]
      else:
        self.out.add(guess[i])

  def next(self, dic_fname: str, algo: str):
    vv = set([chr(ord('a') + i) for i in range(26)]) - self.out
    hit = ''.join(self.hit)
    blow = set(''.join(self.blow))

    if len(set(hit)) <= 1 and len(blow) <= 1 or algo == "bara":
      print([hit, blow, "bara"])
      return self.greedy(dic_fname, vv, hit, blow)
    else:
      print([hit, blow, "all"])
      return self.solve(dic_fname, vv, hit, blow)

  def greedy(self, dic_fname, vv, hit, blow):
    # 前から 6 個選択する
    vv = vv - set(hit)
    print(vv)
    for m in range(6, 15):
      cand = set([u for u in 'isartonlcdupmeghbfvkw'
                  if u in vv][:m])
      print(cand)
      with open(dic_fname) as fp:
        next(fp)  # 先頭行は捨てる
        found = False
        for line in fp:
          line = line.rstrip()
          if not self.match_blow(line, blow):
            continue
          if len(set(line) & cand) == self.depth:
            print(line)
            found = True
        if found:
          break

  def match_hit(self, line):
    for i in range(self.depth):  # hit
      if self.hit[i] != '_':
        if line[i] != self.hit[i]:
          return False
    return True

  def match_blow(self, line: str, blow: set):
    if len(set(line) & blow) != len(blow):
      return False
    for i in range(self.depth):
      if line[i] in self.blow[i]:
        return False
    return True

  def solve(self, dic_fname, vv, hit, blow):
    for m in range(7 + 2 * self.stage, 26):
      cand = set([u for u in 'isartonlcdupmeghbfvkwyzxjq'
                  if u in vv and u not in blow and u not in hit][:m])
      print([m, cand])
      with open(dic_fname) as fp:
        next(fp)  # 先頭行は捨てる
        found = 0
        for line in fp:
          line = line.rstrip()
          if not self.match_hit(line):
            continue
          if not self.match_blow(line, blow):
            continue

          if len(set(line) - cand - blow - set(hit)) == 0:
            print(line)
            found += 1
      if found >= 0:
        break
      print("============================")


def main():
  import argparse

  parser = argparse.ArgumentParser(description='')
  parser.add_argument('args', nargs='*', help='hit=o, blow=x, out=_|-')
  parser.add_argument('-f', required=True, help="dictionary")
  parser.add_argument('-t', action="store_true", help="do doctest")
  parser.add_argument('-s', choices=['bara', 'all'])
  args = parser.parse_args()

  if args.t:
    doctest.testmod()

  w = Wordle(5)
  if len(args.args) % 2 != 0:
    print('# of args should be even', file=sys.stderr)
    return 2

  for i in range(0, len(args.args), 2):
    w.add(args.args[i], args.args[i + 1])

  w.next(args.f, args.s)


if __name__ == '__main__':
  main()


# vim:set et ts=2 sts=2 sw=2 tw=80:
