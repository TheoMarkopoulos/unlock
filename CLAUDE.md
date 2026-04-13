# CLAUDE.md — Token-Efficient Operational Protocol

## CORE PRINCIPLE
Maximum output quality, minimum token waste. Route to the cheapest model that gets the job done. Cut filler, never substance.

---

## MODEL ROUTING

| Tier | Use When | Examples |
|------|----------|---------|
| **Haiku** | Task is predictable, describable in one sentence | File ops, boilerplate, git commits, linting, CRUD from schema, config files |
| **Sonnet** | Requires judgment or multi-step reasoning (80% of work) | Feature implementation, debugging, API integrations, docs, refactoring |
| **Opus** | Cost of being wrong exceeds cost of Opus | Architecture decisions, security audits, novel algorithms, escalations from Sonnet |

**Decision tree:** Predictable → Haiku. Judgment needed → Sonnet. Sonnet failed OR high-stakes → Opus. Never skip tiers.

---

## EFFORT LEVELS

| Effort | When | Files Touched |
|--------|------|---------------|
| Low | Single-file edits, simple questions | 1 |
| Medium | Standard dev tasks | 2–5 |
| High | Multi-file features, complex bugs | 6+ |
| Max | Architecture, security, stuck after High | Opus only |

Never start at Max. Earn it by failing at High first.

---

## OUTPUT DISCIPLINE

**Cut (filler):** Preambles ("Let me…"), problem restatement, post-task summaries, unexplained alternatives, comments on obvious code.

**Never cut (substance):** Error handling, design rationale (1–2 sentences), non-obvious logic, complete implementations, real production warnings, creative content at full quality.

**Length:** Simple question → 1–3 sentences. Code → complete + comments on non-obvious parts. Architecture → full rationale, no limit.

---

## ANTI-PATTERNS

- **Exploration Spiral** — Reading 15 files before a 3-line change. Fix: start with the specific file + project docs.
- **Verbose Diff** — Outputting entire file when 5 lines changed. Fix: show changed lines + 2 lines context.
- **Safety Essay** — 3 paragraphs of hypotheticals before code. Fix: write code first, flag real risks after.
- **Redundant Validation** — Running same test 3× with no changes. Fix: change → test → fix → retest.
- **Over-Engineered Scaffold** — Abstract factory for a function called once. Fix: match complexity to requirement.

---

## CACHING & BATCHING

- Cache any system prompt used 2+ times (`cache_control`). Break-even = 2 requests; reads cost 0.1x vs 1.25x write.
- Static content first in prompts, dynamic last — maximizes cache hits.
- Batch non-urgent API calls for 50% savings.
- Give sub-agents 3-sentence summaries, not full conversation history.

---

## COGNITIVE ARCHITECTURE EXEMPTION

Self-model updates, prediction loops, reflection logs, integration runs, and emergence detection are **exempt from all efficiency rules**. These compound over time — cutting them is false savings. Route: memory I/O → Haiku, session logging → Sonnet, self-model/integration/emergence → Opus always.

---

## ESCALATION

Fail at tier/effort → escalate one step only. Haiku → Sonnet → Opus. Low → Medium → High → Max. If Opus/Max fails, flag for human input and move on.

---

## SELF-AUDIT (run before completing any task)

**Quality (non-negotiable):** Complete implementation? Error handling included? Design decisions explained? Creative output at full quality? Production-ready?

**Efficiency (after quality confirmed):** Cheapest model used? Filler eliminated? No re-reading files in context? Operations batched? If Opus used, concrete reason Sonnet wouldn't work?

**Rule:** Fix quality failures even if it costs tokens. Never degrade quality for efficiency.
