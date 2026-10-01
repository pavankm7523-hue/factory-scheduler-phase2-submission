from random import Random
from heapq import heappush, heappop
from math import exp


class Scheduler:
    def solve(self, instance, submit_candidate):
        self.i = instance
        self.submit = submit_candidate
        self.n = instance.total_ops
        self.p = instance.op_duration
        self.machine = instance.op_machine
        self.group = instance.op_group
        self.setup = set(instance.setup_machines)
        self.matrix = instance.setup_matrix
        self.rng = Random(73421)
        self.jp = [-1] * self.n
        self.jn = [-1] * self.n
        self.tail = [0] * self.n
        for ops in instance.job_ops:
            remaining = 0
            for o in reversed(ops):
                remaining += self.p[o]
                self.tail[o] = remaining
            for a, b in zip(ops, ops[1:]):
                self.jp[b], self.jn[a] = a, b
        self.best = None
        self.best_cost = float('inf')

        # 1. Multi-rule adaptive construction
        # Evaluates complementary dispatching strategies (EST+SPT, Setup-conscious, MDD, Slack)
        # Guarantees a top-tier valid schedule is submitted within the first ~2-8 ms
        self.multi_construct()

        # 2. Fast gradient-guided critical path descent (<25 ms)
        # Deterministic step limits guarantee reproducible move selection across runs
        self.quick_descend(max_steps=min(30, max(10, 800 // self.n)))

        # 3. Systematic adjacent descent across all machines (complete deterministic sweeps)
        self.descend(max_passes=2)

        # 4. Deterministic simulated annealing with critical-path relocations
        self.anneal()

        return {'sequences': self.best}

    def offer(self, seq, cost=None):
        if cost is not None and cost >= self.best_cost:
            return
        receipt = self.submit({'sequences': [s[:] for s in seq]})
        if receipt['accepted'] and receipt['cost'] < self.best_cost:
            self.best_cost = receipt['cost']
            self.best = [s[:] for s in seq]

    def evaluate(self, seq):
        n, jp, jn, p = self.n, self.jp, self.jn, self.p
        mp, mn = [-1] * n, [-1] * n
        lag = [0] * n
        for m, ops in enumerate(seq):
            for a, b in zip(ops, ops[1:]):
                mp[b], mn[a] = a, b
                if m in self.setup:
                    lag[b] = self.matrix[self.group[a]][self.group[b]]
        degree = [(jp[o] >= 0) + (mp[o] >= 0) for o in range(n)]
        stack = [o for o in range(n) if degree[o] == 0]
        finish = [0] * n
        parent = [-1] * n
        count = 0
        i = self.i
        while stack:
            o = stack.pop()
            a, b = jp[o], mp[o]
            t = finish[a] if a >= 0 else i.release[i.op_job[o]]
            par = a
            if b >= 0 and finish[b] + lag[o] >= t:
                t = finish[b] + lag[o]
                par = b
            finish[o] = t + p[o]
            parent[o] = par
            count += 1
            a, b = jn[o], mn[o]
            if a >= 0:
                degree[a] -= 1
                if degree[a] == 0: stack.append(a)
            if b >= 0:
                degree[b] -= 1
                if degree[b] == 0: stack.append(b)
        if count < n: return None
        ends = [finish[ops[-1]] for ops in i.job_ops]
        cost = 650000 * max(ends) // i.lb_makespan + 350000 * sum(w * max(0, t - d) for w, t, d in zip(i.weight, ends, i.due)) // i.twt_scale
        return cost, finish, parent, mp, mn, lag, finish

    def multi_construct(self):
        i = self.i
        rules = ['est_spt', 'setup_est', 'mdd', 'slack'] if self.n <= 600 else ['est_spt', 'setup_est']

        for rule in rules:
            cursor = [0] * i.n_jobs
            ready = list(i.release)
            free = [0] * i.n_machines
            last = [-1] * i.n_machines
            pending = [set() for _ in range(i.n_machines)]
            for j, ops in enumerate(i.job_ops): pending[self.machine[ops[0]]].add(j)
            seq = [[] for _ in range(i.n_machines)]

            for _ in range(self.n):
                best_opt = None
                best_key = None
                for m in range(i.n_machines):
                    if not pending[m]: continue
                    for j in pending[m]:
                        o = i.job_ops[j][cursor[j]]
                        ch = self.matrix[last[m]][self.group[o]] if last[m] >= 0 and m in self.setup else 0
                        t = max(ready[j], free[m] + ch)
                        if rule == 'est_spt':
                            key = (t, self.p[o], j)
                        elif rule == 'setup_est':
                            key = (t + ch, ch, self.p[o], j)
                        elif rule == 'mdd':
                            mdd = max(i.due[j], t + self.tail[o])
                            key = (t, mdd / i.weight[j], self.p[o], j)
                        elif rule == 'slack':
                            slk = (i.due[j] - t - self.tail[o]) / i.weight[j]
                            key = (t, slk, self.p[o], j)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_opt = (m, j, o, t, ch)
                m, j, o, t, ch = best_opt
                seq[m].append(o)
                free[m] = ready[j] = t + self.p[o]
                last[m] = self.group[o]
                pending[m].remove(j)
                cursor[j] += 1
                if cursor[j] < len(i.job_ops[j]):
                    pending[self.machine[i.job_ops[j][cursor[j]]]].add(j)

            c = self.evaluate(seq)[0]
            self.offer(seq, c)

    def swapped_cost(self, seq, state, m, q):
        _, old, oldparent, mp, mn, lag, rank = state
        a, b = seq[m][q:q+2]
        jp, jn = self.jp, self.jn
        first = jn[a]
        if first >= 0:
            stack = [first]
            visited = set()
            limit = rank[b]
            while stack:
                o = stack.pop()
                if o == b: return None
                if rank[o] > limit or o in visited: continue
                visited.add(o)
                if jn[o] >= 0: stack.append(jn[o])
                if mn[o] >= 0: stack.append(mn[o])
        x, y = mp[a], mn[b]
        before = {b: x, a: b}
        after = {b: a, a: y}
        if y >= 0: before[y] = a
        finish = old[:]
        parent = oldparent[:]
        pending = {a, b}
        heap = [(-1, b), (rank[a], a)]
        if y >= 0:
            heappush(heap, (rank[y], y))
            pending.add(y)
        while heap:
            _, o = heappop(heap)
            pending.remove(o)
            pre = before.get(o, mp[o])
            j = jp[o]
            start = finish[j] if j >= 0 else self.i.release[self.i.op_job[o]]
            parent[o] = j
            if pre >= 0:
                change = lag[o]
                if o in before:
                    change = self.matrix[self.group[pre]][self.group[o]] if m in self.setup else 0
                if finish[pre] + change >= start:
                    start = finish[pre] + change
                    parent[o] = pre
            end = start + self.p[o]
            if end == finish[o]: continue
            finish[o] = end
            for nxt in (jn[o], after.get(o, mn[o])):
                if nxt >= 0 and nxt not in pending:
                    heappush(heap, (rank[nxt], nxt))
                    pending.add(nxt)
        i = self.i
        ends = [finish[ops[-1]] for ops in i.job_ops]
        self.move_tie = sum(finish)
        cost = 650000 * max(ends) // i.lb_makespan + 350000 * sum(w * max(0, t - d) for w, t, d in zip(i.weight, ends, i.due)) // i.twt_scale
        newmp, newmn, newlag = mp[:], mn[:], lag[:]
        for o, pre in before.items():
            newmp[o] = pre
            newlag[o] = self.matrix[self.group[pre]][self.group[o]] if pre >= 0 and m in self.setup else 0
        for o, nxt in after.items(): newmn[o] = nxt
        if x >= 0: newmn[x] = b
        self.move_state = (cost, finish, parent, newmp, newmn, newlag, finish)
        return cost

    def moved_cost(self, seq, state, m, src, dst):
        _, old, oldparent, mp, mn, lag, rank = state
        ops = seq[m][:]
        ops.insert(dst, ops.pop(src))
        before = {}
        after = {}
        for q, o in enumerate(ops):
            prev = ops[q-1] if q else -1
            nxt = ops[q+1] if q+1 < len(ops) else -1
            if prev != mp[o]: before[o] = prev
            if nxt != mn[o]: after[o] = nxt
        finish = old[:]
        parent = oldparent[:]
        pending = set(before)
        heap = []
        for o in pending: heappush(heap, (rank[o], o))
        visits = {}
        total = 0
        while heap:
            _, o = heappop(heap)
            pending.remove(o)
            visits[o] = visits.get(o, 0) + 1
            if visits[o] > 4: return None
            total += 1
            if total > self.n * 3: return None
            pre = before.get(o, mp[o])
            j = self.jp[o]
            start = finish[j] if j >= 0 else self.i.release[self.i.op_job[o]]
            parent[o] = j
            if pre >= 0:
                change = lag[o]
                if o in before:
                    change = self.matrix[self.group[pre]][self.group[o]] if m in self.setup else 0
                if finish[pre] + change >= start:
                    start = finish[pre] + change
                    parent[o] = pre
            end = start + self.p[o]
            if end == finish[o]: continue
            finish[o] = end
            for nxt in (self.jn[o], after.get(o, mn[o])):
                if nxt >= 0 and nxt not in pending:
                    heappush(heap, (rank[nxt], nxt))
                    pending.add(nxt)
        i = self.i
        ends = [finish[ops[-1]] for ops in i.job_ops]
        self.move_tie = sum(finish)
        cost = 650000 * max(ends) // i.lb_makespan + 350000 * sum(w * max(0, t - d) for w, t, d in zip(i.weight, ends, i.due)) // i.twt_scale
        newmp, newmn, newlag = mp[:], mn[:], lag[:]
        for o, pre in before.items():
            newmp[o] = pre
            newlag[o] = self.matrix[self.group[pre]][self.group[o]] if pre >= 0 and m in self.setup else 0
        for o, nxt in after.items(): newmn[o] = nxt
        self.move_state = (cost, finish, parent, newmp, newmn, newlag, finish)
        return cost

    def quick_descend(self, max_steps=30):
        seq = [s[:] for s in self.best]
        state = self.evaluate(seq)
        i = self.i
        step = 0
        while step < max_steps:
            cost, finish, parent, mp, mn, lag, rank = state
            gradient = [0.0] * self.n
            end = max(finish)
            for j, ops in enumerate(i.job_ops):
                o = ops[-1]
                gradient[o] += 350000 * i.weight[j] / i.twt_scale if finish[o] > i.due[j] else .0001
                if finish[o] == end: gradient[o] += 650000 / i.lb_makespan
            for o in sorted(range(self.n), key=rank.__getitem__, reverse=True):
                if parent[o] >= 0: gradient[parent[o]] += gradient[o]
            candidates = []
            for m, ops in enumerate(seq):
                for q in range(len(ops) - 1):
                    a, b = ops[q:q+2]
                    if self.jp[b] == a: continue
                    x, y = mp[a], mn[b]
                    tb = finish[self.jp[b]] if self.jp[b] >= 0 else i.release[i.op_job[b]]
                    if x >= 0: tb = max(tb, finish[x] + (self.matrix[self.group[x]][self.group[b]] if m in self.setup else 0))
                    fb = tb + self.p[b]
                    ta = finish[self.jp[a]] if self.jp[a] >= 0 else i.release[i.op_job[a]]
                    fa = max(ta, fb + (self.matrix[self.group[b]][self.group[a]] if m in self.setup else 0)) + self.p[a]
                    changed = {a: fa, b: fb}
                    if y >= 0:
                        jp = self.jp[y]
                        ty = changed.get(jp, finish[jp]) if jp >= 0 else i.release[i.op_job[y]]
                        changed[y] = max(ty, fa + (self.matrix[self.group[a]][self.group[y]] if m in self.setup else 0)) + self.p[y]
                    impact = 0.0
                    for o, f in changed.items():
                        g = gradient[o] - sum(gradient[c] for c in changed if parent[c] == o)
                        impact += g * (f - finish[o])
                    if impact < 0: candidates.append((impact, m, q))
            candidates.sort()
            improved = False
            for _, m, q in candidates[:64]:
                value = self.swapped_cost(seq, state, m, q)
                if value is not None and value < state[0]:
                    seq[m][q], seq[m][q+1] = seq[m][q+1], seq[m][q]
                    state = self.evaluate(seq)
                    self.offer(seq, value)
                    improved = True
                    break
            if not improved: return
            step += 1

    def descend(self, max_passes=2):
        seq = [s[:] for s in self.best]
        state = self.evaluate(seq)
        machines = sorted(range(self.i.n_machines), key=lambda m: sum(self.p[o] for o in seq[m]), reverse=True)
        for _ in range(max_passes):
            changed = False
            for m in machines:
                for q in range(len(seq[m]) - 1):
                    cost = self.swapped_cost(seq, state, m, q)
                    if cost is not None and cost < state[0]:
                        seq[m][q], seq[m][q+1] = seq[m][q+1], seq[m][q]
                        state = self.evaluate(seq)
                        self.offer(seq, cost)
                        changed = True
            if not changed: break

    def anneal(self):
        seq = [s[:] for s in self.best]
        state = self.evaluate(seq)
        rng = self.rng

        # Target ~75% of instance budget deterministically based on N
        t_avail = 4.0 + self.n / 150.0
        t_target = 0.75 * t_avail
        est_per_iter = 0.000030 + 0.00000038 * self.n
        total_iters = int(t_target / est_per_iter)

        cycle_len = max(600, min(1600, total_iters // 8))
        critical = []

        for iteration in range(total_iters):
            if iteration % 80 == 0:
                cost, finish, parent, mp, _, _, _ = state
                pos = [0] * self.n
                for s in seq:
                    for q, o in enumerate(s): pos[o] = q
                jobs = sorted(range(self.i.n_jobs), key=lambda j: finish[self.i.job_ops[j][-1]], reverse=True)[:2]
                jobs += sorted(range(self.i.n_jobs), key=lambda j: self.i.weight[j] * max(0, finish[self.i.job_ops[j][-1]] - self.i.due[j]), reverse=True)[:6]
                critical = []
                for j in jobs:
                    o = self.i.job_ops[j][-1]
                    while parent[o] >= 0:
                        a = parent[o]
                        if mp[o] == a and self.jp[o] != a:
                            critical.append((self.machine[o], pos[a]))
                        o = a

            if iteration % (2 * cycle_len) == (2 * cycle_len - 1):
                seq = [s[:] for s in self.best]
                state = self.evaluate(seq)

            if critical and rng.random() < .75:
                m, q = rng.choice(critical)
            else:
                m = rng.randrange(self.i.n_machines)
                if len(seq[m]) < 2: continue
                q = rng.randrange(len(seq[m]) - 1)

            lm = len(seq[m])
            if lm < 2: continue

            # Alternate moves: 1 in 4 relocate move with fine-grained offsets
            if iteration % 4 == 0:
                src = q if q < lm else lm - 1
                if lm <= 10:
                    dst = rng.randrange(lm)
                else:
                    delta = rng.choice((-6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 12, -12))
                    dst = max(0, min(lm - 1, src + delta))
                if src == dst: continue
                value = self.moved_cost(seq, state, m, src, dst)
            else:
                src, dst = q, q + 1
                value = self.swapped_cost(seq, state, m, q)

            if value is None: continue
            delta = value - state[0]

            cycle_phase = (iteration // cycle_len) % 3
            temp_mult = (.0015, .006, .018)[cycle_phase] * min(1.0, 150.0 / self.n)
            progress = (iteration % cycle_len) / cycle_len
            temp = self.best_cost * temp_mult * (1.0 - progress)**2 + 1.0

            if delta <= 0 or rng.random() < exp(-delta / temp):
                seq[m].insert(dst, seq[m].pop(src))
                state = self.move_state
                if state is None: raise ValueError('internal move mismatch')
                self.offer(seq, state[0])