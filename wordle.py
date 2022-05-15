#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

import doctest
import os
import re
import sys


class Wordle(object):

  freq_char = 'aesoriltnudcymphbgkfwvzjxq'

  difficult = ['sails', 'bails', 'jails',
               'eases', 'saxes', 'eaves', 'vases', 'saves', 'babes', 'bases',
               'sabes', 'gages',
               'safes', 'gases', 'sages', 'sades', 'sakes', 'zaxes', 'hajes',
               'faxes', 'haves', 'jades', 'jakes', 'bakes', 'hazes',
               'dazes', 'james', 'gazes', 'fazes', 'hades', 'hakes', 'gades',
               'fades', 'fakes', 'mazes', 'hames', 'mages', 'kames', 'dames',
               'makes', 'games', 'fames',
               'sanes', 'banes', 'fanes', 'janes', 'kanes',
               'names', 'naves', 'manes', 'vanes',
               'bange'
               ]

  def dprint(self, v, lv=0):
    print(f'<<{self.stage+1}>> {v}')

  def __init__(self, depth: int):
    self.depth = depth
    self.dic = []
    self.quiet = False
    self.max_stage = 6
    self.reset()

  def _init_costs(self):
    n = 26  # 文字種数
    self.cost = [0] * n
    self.costs = [0] * self.depth
    for i in range(self.depth):
      self.costs[i] = [0] * n

  def reset(self):
    self.out = set()
    self.hit = ['_'] * self.depth
    self.blow = [''] * self.depth
    self.stage = 0
    self._init_costs()

  def clone(self):
    w = Wordle(self.depth)
    w.stage = self.stage
    w.hit = self.hit.copy()
    w.blow = self.blow.copy()
    w.out = self.out.copy()
    w.dic = self.dic
    return w

  def _can_solve(self, answer) -> bool:
    while self.stage < self.max_stage:
      v = self.next(None)
      assert len(self.cand) > 0
      c = self.cand[-1]
      r = self.response(answer, c)
      self.add(c, r)
    return not v

  def can_solve(self, cand) -> bool:
    for idx, c in enumerate(cand):  # 正解が c だったときに解けるか
      w = self.clone()
      w.dic = cand
      r = self.response(c, cand[-1])
      w.add(cand[-1], r)
      if not w._can_solve(c):
        return False
    return True

  def muri(self, hit, blow):
    """このままいくと詰むかも.

    o-ooo のようなときに候補が 7 個あると詰む
    """
    if len(self.cand) <= 1:  # クリア
      return False
    if self.stage <= 1:  # 判断する状況にないのでは?
      return False
    if len(self.cand) + self.stage <= self.max_stage:
      # 総当りでなんとかなる
      self.dprint(f"muri: cand={len(self.cand)}")
      return False
    if self.stage >= self.max_stage - 1:
      # 最後なのに複数候補．運勝負
      self.giveup = True
      return False
    self.dprint(f"muri: cand[{len(self.cand)}]={self.cand}")

    # 普通に解いたら最大あと何手？
    if self.can_solve(self.cand.copy()):
      return False

    hnum = 0
    for i in range(self.depth):
      if self.hit[i] != '_':
        hnum += 1
    self.dprint(f"hit={hnum}, blow={len(blow)}, bh={len(blow|set(hit)-set('_'))}")
    if hnum >= self.depth - 2:
      return True
    if len(blow | set(hit) - set('_')) >= self.depth - 2:
      return True

    return False

  def reduce_cand(self, vvv, hit, blow):
    """このままいくと詰みそうなので，まだ使用していない文字を優先する
    """
    uu = set()
    ww = set(self.cand[0])
    for i in range(len(self.cand)):
      ww = ww & set(self.cand[i])
      for j, ch in enumerate(self.cand[i]):
        if self.hit[j] == '_':
          uu.add(ch)
    uu = uu - ww
    if len(uu) > 10:
      vv = uu - ww
    else:
      vv = (vvv | uu) - ww
    self.dprint(f"too bad {vv} ({len(vvv)} | {len(uu)}={len(vvv | uu)}) - {len(ww)} => {len(vv)}")
    cand = self.cand
    self.cand = []
    num_b = max([len(uu), self.depth]) - self.stage
    chars = (''.join(list(uu)) +
             ''.join([x for x in self.freq_char if x not in uu]))
    for j in range(2):
      for vvv in [vv, set(self.freq_char)]:
        ret = self.greedy(vvv, '______', blow,
                          freq_char=chars, num_a=1, num_b=num_b,
                          prio=uu, prio_min=self.depth - 2 - j)
        self.dprint(f"greedy end j={j}, ret={ret}, vv={vvv}")
        if ret != 0:
          break
      if ret != 0:
        break
    if ret == 0:  # 候補見つからず
      self.cand = cand
    else:
      self.cmp_cost()
      self.cand.sort(key=lambda x: self.weight2(x, uu))
      for c in self.cand:
        self.dprint(f"reduce_cand() {c}, {self.weight2(c, uu)}, {set(c) & uu}")

  def add_dict(self, fname):
    with open(fname) as fp:
      next(fp)  # 先頭行は捨てる
      for line in fp:
        line = line.rstrip()
        self.dic.append(line)

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

  def next(self, algo: str):
    vv = set([chr(ord('a') + i) for i in range(26)]) - self.out
    hit = ''.join(self.hit)
    blow = set(''.join(self.blow))
    self.cand = []

    self.giveup = False
    if self.stage <= 2 and len(set(hit)) <= 1 and len(blow) <= 1 or algo == "bara":
      self.dprint([hit, blow, "bara"])
      ret = self.greedy(vv, hit, blow)
    else:
      ret = 0
    if ret == 0:
      self.dprint([hit, blow, "all"])
      self.solve(vv, hit, blow)

    # 評価する
    self.cmp_cost()
    self.cand.sort(key=lambda x: self.weight(x))
    if self.muri(hit, blow):
      # このままいくと無駄打ちな可能性があるので...
      self.reduce_cand(vv, hit, blow)
    return self.giveup

  def cmp_cost(self):
    self._init_costs()
    for c in self.cand:
      for s in set(c):
        self.cost[ord(s) - ord('a')] += 1
      for i, ci in enumerate(c):
        self.costs[i][ord(ci) - ord('a')] += 1
    # out なものはゴミでしょう
    for c in range(26):
      a = chr(c + ord('a'))
      if a in self.out:
        self.cost[c] *= 0.4

  def weight(self, x):
    """良さげな候補を探したい"""
    xx = set(x)
    w0 = len(xx)  # 文字種別数
    w1 = 0
    for s in xx:  # 数字の優先度
      w1 += self.cost[ord(s) - ord('a')]
    w2 = 0
    for i, s in enumerate(x):
      w2 += self.costs[i][ord(s) - ord('a')]
    return (w0, w1, w2)

  def weight2(self, x, prio):
    """良さげな候補を探したい

    prio が優先度の高い文字
    """
    xx = set(x)
    w0 = len(xx)  # 文字種別数
    w1 = len(xx & prio)
    w2 = 0
    for s in xx:
      w2 += self.cost[ord(s) - ord('a')]
    return (w0, w1, w2)

  def greedy(self, vv, hit: str, blow,
             freq_char: str = None,  # 優先度の高い文字
             num_a: int = 3,  # 候補文字数の計算式 a * stage + b
             num_b: int = 26,
             prio: set = set(), prio_min: int = 0):
    vv = vv - set(hit)
    num_b = 26
    if prio_min > len(prio):
      prio_min = len(prio)
    if freq_char is None:
      freq_char = self.freq_char
    self.dprint(f"greedy vv={vv}, prio={prio}/{prio_min}, chars={freq_char}")
    for m in range(min([self.stage * num_a + num_b, len(vv) - 1]), len(vv)):
      cand = set([u for u in freq_char
                  if u in vv][:m])
      print([m, len(vv), num_a, self.stage, num_b, cand], file=sys.stderr)
      self.dprint(f"greedy m={m}, cand={cand}")
      for line in self.dic:
        if not self.match_blow(line, blow) and prio_min <= 0:
          continue
        if len(prio & set(line)) < prio_min:
          continue
        if len(set(line) & cand) == self.depth:
          self.cand.append(line)
      if len(self.cand) > 0:
        break
    return len(self.cand)

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

  def solve(self, vv, hit, blow: set):
    """

    blow := set(self.blow)
    """
    ret = 0
    print([self.blow, blow])
    for m in range(min([7 + 5 * self.stage, 26]), 28, 3):
      cand = set([u for u in self.freq_char
                  if u in vv and u not in blow and u not in hit][:m])
      for line in self.dic:
        if not self.match_hit(line):
          continue
        if not self.match_blow(line, blow):
          continue

        if len(set(line) - cand - blow - set(hit)) == 0:
          if not self.quiet:
            print(line)
          ret += 1
          self.cand.append(line)
      if ret > 0:
        break
      self.dprint(f" {m}/{len(self.cand)}/{len(cand)} =======================")
    return ret

  @classmethod
  def response(cls, problem: str, guess: str) -> str:
    """guess と answer から response を構築する

    >>> Wordle.response('fugab', 'gabfu')
    'xxxxx'
    >>> Wordle.response('accur', 'bpcpc')
    '--o-x'
    >>> Wordle.response('accur', 'bpccc')
    '--ox-'
    >>> Wordle.response('accur', 'cpcpa')
    'x-o-x'
    >>> Wordle.response('metal', 'raise')
    '-x--x'
    >>> Wordle.response('metal', 'leant')
    'xox-x'
    >>> Wordle.response('metal', 'tepal')
    'xo-oo'
    >>> Wordle.response('metal', 'tapal')
    'x--oo'
    >>> Wordle.response('metal', 'tepaa')
    'xo-o-'
    """
    if len(problem) != len(guess):
      raise ValueError()
    cnt = {}
    for i, p in enumerate(problem):
      if p != guess[i]:
        cnt[p] = cnt.get(p, 0) + 1

    r = ''
    for i, g in enumerate(guess):
      if problem[i] == g:
        r += 'o'
      elif cnt.get(g, 0) == 0:
        r += '-'
      else:
        r += 'x'
        cnt[g] -= 1
    return r


def optim(word_dic, problems, d=5):
  probs = []
  if os.path.exists(problems):  # ファイルがあったら読み込む
    with open(problems) as fp:
      for line in fp:
        line = line.rstrip()
        if len(line) == d:
          probs.append(line)
  else:  # なかったら，問題のcsvだと思う
    for line in problems.split(','):
      line = line.rstrip()
      if len(line) == d:
        probs.append(line)

  w = Wordle(d)
  w.add_dict(word_dic)
  w.quiet = True
  dic = w.dic
  score = [0] * (w.max_stage + 1)
  ng = 0
  ngword = None
  for i, p in enumerate(probs):
    if p in w.difficult:
      continue
    w.reset()
    w.dic = dic
    found = False
    rr = ''
    for c in range(1, w.max_stage + 1):  # 6回試せる設定
      w.next(None)
      # print([len(w.cand), None if len(w.cand) == 0 else w.cand[0], p, c, rr])
      if len(w.cand) == 0:
        # 辞書になかったということでしょう?
        assert False, f"{p} is not in dic? {p in dic}"
        continue
      if w.cand[-1] == p:
        found = True
        break
      rr = w.response(p, w.cand[-1])
      print(f"{w.cand[-1]}, {rr} [problem={p}], {c}, {w.stage}")
      w.add(w.cand[-1], rr)
    if not found:
      c += 1
      ngword = p
      for cc in w.cand:
        print([cc, w.weight(cc)])
      assert False, [p, w.blow, w.hit, w.stage, c, w.max_stage, found]
    score[c - 1] += 1
    print(["score", score, ng, ngword, c, p])


def main():
  import argparse

  parser = argparse.ArgumentParser(description='')
  parser.add_argument('args', nargs='*', help='hit=o, blow=x, out=_|-')
  parser.add_argument('-f', required=True, help="dictionary")
  parser.add_argument('-p', help="problem")
  parser.add_argument('-t', action="store_true", help="do doctest")
  parser.add_argument('-s', choices=['bara', 'all'])
  args = parser.parse_args()

  if args.t:
    v = doctest.testmod()
    return 3 if v.failed > 0 else 0
  if args.p:
    optim(args.f, args.p)
    return

  w = Wordle(5)
  if len(args.args) % 2 != 0:
    print('# of args should be even', file=sys.stderr)
    return 2

  for i in range(0, len(args.args), 2):
    w.add(args.args[i], args.args[i + 1])

  w.add_dict(args.f)
  w.next(args.s)
  for c in w.cand:
    print(f'{c}, {w.weight(c)}')


if __name__ == '__main__':
  sys.exit(main())


# vim:set et ts=2 sts=2 sw=2 tw=80:
