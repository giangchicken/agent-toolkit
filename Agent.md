# Design rules

**How to read this.** Top down: one root sentence, three causes, then the rules hanging under each cause. Every rule has a fixed ID (`H-1`, `C-3`…) for citing in review. The rule text is verbatim from the source document; the framing around it names the principle each rule comes from.

**Enforcement:** `[lint]` a check that fails the build · `[review]` caught by a human reading the diff · `[none]` judgement only, no check exists.

Worked examples and the folded commentary live in `nguyen-tac.vi.md`, same IDs.

---

## Root

> **The price of software is not the first write. It is the Nth change.**

Every rule here exists to make one ordinary change cheaper. None of them is about taste.

A change gets expensive for exactly three reasons — dependencies, obscurity and carrying cost — and the three branches below are those reasons. A fourth, numbered zero, adds no design rule; it keeps the other three from rotting.

---

## Vocabulary

Read this first. These four words carry the rest of the document.

- **Module** — anything with an interface and an implementation — a function, a class, a package, a service — that hides a decision behind that interface. The more it hides behind the smaller interface, the deeper it is. A shallow module — wide interface, little hidden — is a bad sign in logic and in adapter, where there was a decision to hide; in shape, facade and wiring it is the normal state, because there is no decision there to hide.
- **Interface** — everything a caller must know to use it correctly: the signature, the invariants, the ordering, the error modes, the required configuration, the speed. Not just the types. The longer that list runs, the wider the surface a caller is stuck to, even when the signature never moves.
- **Coupling** — what two parts share and how tightly that binds them — the kind and the degree of the dependency between them.
- **Connascence** — one thing forcing another to change with it. It is measured on three axes: strength, how hard the joint change is to get right; locality, how far apart the two ends sit; degree, how many places are involved. This is the part you actually pay for; coupling is the wider word around it.

---

# 1. Dependencies

> A change is expensive when fixing one place forces you to fix another. Two ways out: **hide** what can be hidden, and **weaken** the ties that cannot.

## 1A — Hide the decision

> **Principle.** Each module exists to hide one decision from the rest of the system. The decision inside may change as often as it likes, as long as the interface does not.
>
> *Parnas, 1972.* He argues that starting a decomposition from the flowchart is almost always the wrong move; begin instead from a list of the design decisions that are hard, or that you expect to change, and build each module around hiding one of them. Those decisions outlive any single moment of execution, which is why modules end up not lining up with processing steps at all. This whole branch is that paper.

- **H-1** `[review]` Give each module one decision to hide, and pick the one most likely to change: the file format, the storage engine, the retry policy.
- **H-2** `[review]` Put a lot behind a small interface.
- **H-3** `[review]` Keep the file format and the call order inside the module. A caller that has to know them is already leaked, with or without an import.
- **H-4** `[review]` Write the smallest version by interface, not the shortest by line count. Cutting a function in half to make each half shorter adds an interface, and if the halves stay entangled the reader now has to hold both. An implementation aimed slightly wider than today's single call site usually ends up with a simpler, deeper interface than one specialised to it.
- **H-5** `[lint]` Translate schema, wire and framework types at the outermost layer, in an adapter, so the domain never learns the transport.
- **H-6** `[lint]` Write structured events to standard output, and never manage log files. Logging is a side effect, so the domain returns what happened and the edge writes it down.
- **H-7** `[review]` Do not begin the decomposition from the flow of processing. A step can be a unit of composition — one thing done well, clipped onto the next — but it must not be the unit that hides a decision: a decision usually spans several steps, so splitting by step lands every change in three files. Give the decision its own module and let the steps call it.
- **H-8** `[lint]` Start every module docstring with one of five tags: shape holds nouns and constants; logic holds the decisions over them; facade only re-exports; adapter translates between an outside format and the domain; wiring is the composition root, where the concrete parts get connected. Two answers means the file holds two jobs. Write facade without the cedilla — the tag is read by a machine.

| tag | holds | imported by | may import |
|---|---|---|---|
| `shape` | nouns and constants | everything | nothing |
| `logic` | the decisions over them | adapter, wiring | shape |
| `adapter` | outside ↔ domain translation | wiring | shape, logic |
| `facade` | re-exports only | outside the package | inside the package |
| `wiring` | the composition root | entry point | everything |

The fourth column is the declaration `E-1` enforces. The tag set and the import direction are one thing, not two.

- **H-9** `[review]` Start with one codebase and real module boundaries inside it. Split out a service only when scaling, ownership or reliability genuinely differ — scale itself is bought with indexes, queues and caches. A split does not reduce connascence: both sides still change together, at the same strength and degree. It changes the kind of coupling — an in-process call becomes a network call — and it makes locality worse. If the joint change does not get weaker, you have traded a function call for a network call and bought nothing.

## 1B — Manage the connascence

> **Principle.** Some ties cannot be hidden — two places genuinely have to change together. For those, do two things: pull them **closer**, and trade the strong tie for a **weaker** one.
>
> *Page-Jones, 1992.* You rarely delete connascence. You move it along three axes — strength, locality, degree — and every rule below is one of those moves.

- **C-1** `[review]` The farther apart two elements sit, the weaker the form of connascence between them has to be. If you cannot weaken it, move them closer together — that is why the shape and the logic of one domain share a folder.
- **C-2** `[review]` Group by domain first, technical kind second: things that change together live together. Adding a field to an order should open one folder, not four.
- **C-3** `[lint]` Replace a hard-coded value with a named constant, positional arguments with keyword arguments, an implied ordering with an explicit one. Exempt the values that carry no rule: 0, 1, -1, the empty string, and the status codes the protocol already names. Each swap turns a rule someone has to remember into one the code states, and trades a stronger form of connascence for a weaker one.
- **C-4** `[review]` Make a function when two or more places call it and change together, or when a decision needs its own test. Otherwise a named variable is enough to put the rule on screen.
- **C-5** `[review]` Inline a one-line function with a single caller whose name adds nothing. Three kinds look inlinable but are not: a method implementing an interface, a callback named in configuration or metrics, a decorated function. Removing those still runs while something is lost silently.
- **C-6** `[review]` Keep the decision and the side effect in different functions. One function that validates, writes, calls out and notifies has four reasons to change.
- **C-7** `[review]` Do not split before a second consumer needs half of it, and do not call the split finished while the two halves still depend on each other.

---

# 2. Obscurity

> **Principle.** A change is expensive when you cannot tell **where** to make it, or cannot tell afterwards whether you understood it correctly. This is the cheapest branch to get right and the one most often skipped.
>
> Naming and comments are not decoration here. They are the only part of the interface that carries intent, and intent is exactly what the compiler throws away.

- **R-1** `[review]` Name what is there, not how it got there: what a function returns, what a variable holds, what a file contains. Test a name by reading only the line that uses it and saying what it is.
- **R-2** `[lint]` Never name a thing after its container or its position — user_map, order_list — or reuse a word that already names a step, a command or a table. `main` is the exception the language already owns: fine as an entry point, never as the name of a domain module.
- **R-3** `[lint]` Name a module for what it holds or the job it does, never helpers, utilities, managers or entities. Those name a bucket, not a job.
- **R-4** `[review]` Make the top of the source tree name the domain, not the framework.
- **R-5** `[review]` Comment the abstraction, not the implementation: state the invariants, the units, the ordering and the error modes next to the signature, because none of them are in it. Code does not document itself — if a caller has to read the body to learn the rule, the comment is missing, not redundant.

---

# 3. Carrying cost

> **Principle.** A change is expensive when you have to drag along what you did not need. Every line is debt; every process is a mouth to feed.
>
> This branch is the counterweight to 1A. Hiding a decision is free to keep; a queue is not.

- **T-1** `[review]` Build only what was asked for, and only for the case that exists today. This governs behaviour and features.
- **T-2** `[review]` Boundaries are the exception, and they are decided ahead of the need: hiding a decision is not a feature built early. Decide the boundary early and cut the file late — knowing where the seam runs does not oblige you to split it today.
- **T-3** `[review]` Add something that has to run — a queue, a cache, a scheduled job — only to fix a problem you can name. Everything that runs is something to keep alive.
- **T-4** `[review]` Optimise for clarity first, then for a bottleneck you can name. A bottleneck you cannot name is a guess.
- **T-5** `[review]` Delete a module that nothing gets harder without.

---

# 0. Enforcement

> **Principle.** A rule with no check is a rule already broken — you just have not found out yet. This branch adds no design rule; it keeps the three above from rotting.
>
> *Ford, Parsons and Kua.* Their observation is that the IDE actively encourages the imports that break module discipline, that a written coding standard on its own does not hold, and that the only thing that does hold is a check running in the pipeline.

- **E-1** `[lint]` Declare the import direction once and enforce it with a check that fails the build, not with discipline.
- **E-2** `[none]` Expect a boundary to follow ownership: one that no single team owns will not hold. Draw the modules and divide the teams together rather than fighting the pull.
- **E-3** `[none]` Redraw the boundaries when a typical change keeps touching more files.

Known gap in `E-1`: two modules sharing one database table are coupled through the schema, and schema coupling never shows up in the import graph. No rule here covers it yet.

---

## Sources

- Conway, *How Do Committees Invent?*, 1968 — `E-2`
- Parnas, *On the Criteria To Be Used in Decomposing Systems into Modules*, 1972 — all of 1A
- Constantine and Yourdon, *Structured Design*, 1979 — coupling, cohesion
- Page-Jones, *Comparing Techniques by Means of Encapsulation and Connascence*, 1992 — all of 1B
- Feathers, *Working Effectively with Legacy Code*, 2004
- Martin, *Screaming Architecture*, 2011, and *Clean Architecture* — `R-4`
- Cockburn, *Ports & Adapters*; Palermo, *Onion Architecture* — `H-5`
- Sandin, *Four Strategies for Organizing Code*, 2016 — `C-2`
- Ousterhout, *A Philosophy of Software Design*, 2018 — the root, `H-2`, `H-4`, `R-5`
- Wiggins and others, *The Twelve-Factor App* — `H-6`
- North, *CUPID — for joyful coding*, 2022 — `H-7`, `R-4`
- Ford, Parsons and Kua, *Building Evolutionary Architectures* — branch 0
- Fowler, *MonolithFirst* — `H-9`