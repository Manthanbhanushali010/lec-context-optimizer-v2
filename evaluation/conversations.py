"""
Evaluation Dataset
==================
10 synthetic conversations covering mixed technical + business domains.
Each conversation is 50+ messages with embedded landmarks, noise, and relevant context.
Each has multiple test queries of different types (factual, analytical, status).
"""

from optimizer.types import Message, MessageRole


def _msg(index: int, role: str, content: str) -> Message:
    return Message(index=index, role=MessageRole(role), content=content)


# ── Conversation 1: Backend architecture decision ────────────────────────────
CONV_1 = [
    _msg(0, "user", "Hey team, we need to decide on our database for the new service."),
    _msg(1, "assistant", "Sure. What are the main requirements?"),
    _msg(2, "user", "We need horizontal scaling, strong consistency, and JSON support."),
    _msg(3, "assistant", "That narrows it down to PostgreSQL with JSONB or MongoDB."),
    _msg(4, "user", "What about latency requirements?"),
    _msg(5, "assistant", "Sub-10ms for reads, we can tolerate 50ms for writes."),
    _msg(6, "user", "MongoDB has better horizontal scaling out of the box."),
    _msg(7, "assistant", "True, but PostgreSQL's JSONB performance has improved a lot."),
    _msg(8, "user", "Let's look at the benchmarks. Do you have any?"),
    _msg(9, "assistant", "I'll pull them up. Give me a moment."),
    _msg(10, "user", "Sure, no rush."),
    _msg(11, "assistant", "Okay, PostgreSQL JSONB does 95k reads/sec vs MongoDB's 87k on our hardware."),
    _msg(12, "user", "That's surprising. PostgreSQL wins on reads."),
    _msg(13, "assistant", "MongoDB wins on writes though — 45k vs 38k."),
    _msg(14, "user", "Our workload is 80% reads so PostgreSQL looks better overall."),
    _msg(15, "assistant", "Agreed. Also, the team already knows PostgreSQL well."),
    _msg(16, "user", "Good point. Operational familiarity matters."),
    _msg(17, "assistant", "We also have existing tooling — pgAdmin, connection pooling with PgBouncer."),
    _msg(18, "user", "Right. I think we've decided then."),
    _msg(19, "assistant", "We decided to go with PostgreSQL with JSONB for the new service."),
    _msg(20, "user", "Great. Who's handling the schema design?"),
    _msg(21, "assistant", "I'll take that on. I'll have a draft by Thursday."),
    _msg(22, "user", "Perfect. What about migrations?"),
    _msg(23, "assistant", "We'll use Alembic. It integrates well with SQLAlchemy."),
    _msg(24, "user", "Sounds good. Let's also make sure we have proper indexing from the start."),
    _msg(25, "assistant", "Agreed. I'll include a GIN index on the JSONB columns in the schema."),
    _msg(26, "user", "What's the timeline for getting the DB provisioned in staging?"),
    _msg(27, "assistant", "DevOps said they can have it ready by end of week — Friday at the latest."),
    _msg(28, "user", "Good. And prod?"),
    _msg(29, "assistant", "Two weeks after staging sign-off, so roughly 15 March."),
    _msg(30, "user", "The deadline for production DB is 15 March. Let's track that."),
    _msg(31, "assistant", "Noted. I'll add it to the project tracker."),
    _msg(32, "user", "By the way, should we use RDS or self-managed?"),
    _msg(33, "assistant", "RDS is easier to operate. The cost is about £800/month for our expected load."),
    _msg(34, "user", "That's within budget. Let's go with RDS."),
    _msg(35, "assistant", "Approved. RDS PostgreSQL it is. I'll update the architecture doc."),
    _msg(36, "user", "What version of PostgreSQL should we target?"),
    _msg(37, "assistant", "PostgreSQL 16 — it's the latest stable with the best JSONB performance."),
    _msg(38, "user", "Good. Make that a requirement in the infra ticket."),
    _msg(39, "assistant", "Done. Anything else on the DB side?"),
    _msg(40, "user", "I think we're set. Thanks everyone."),
    _msg(41, "assistant", "Happy to help. I'll send a summary email after this call."),
    _msg(42, "user", "Great. One last thing — backup strategy?"),
    _msg(43, "assistant", "RDS automated backups with 7-day retention. Point-in-time recovery enabled."),
    _msg(44, "user", "Perfect. That covers our compliance requirement."),
    _msg(45, "assistant", "Correct. And we'll do weekly manual snapshots before major releases."),
    _msg(46, "user", "Budget approved for RDS at £800/month."),
    _msg(47, "assistant", "Great. I'll proceed with the provisioning ticket."),
    _msg(48, "user", "Thanks. Let's reconvene Thursday to review the schema draft."),
    _msg(49, "assistant", "Thursday it is. I'll have the schema and migration scripts ready."),
    _msg(50, "user", "Perfect."),
]

CONV_1_QUERIES = [
    {"query": "What database did we decide to use and why?", "type": "factual"},
    {"query": "What were the trade-offs between PostgreSQL and MongoDB?", "type": "analytical"},
    {"query": "What are the key deadlines and commitments from this discussion?", "type": "factual"},
    {"query": "What is the budget and infrastructure plan?", "type": "factual"},
]


# ── Conversation 2: Product roadmap planning ──────────────────────────────────
CONV_2 = [
    _msg(0, "user", "Let's plan Q3. We need to decide what we're building."),
    _msg(1, "assistant", "What are the top priorities from the board?"),
    _msg(2, "user", "Revenue growth and user retention. In that order."),
    _msg(3, "assistant", "That suggests we should prioritise monetisation features first."),
    _msg(4, "user", "Agreed. The premium tier has been on the backlog too long."),
    _msg(5, "assistant", "What features should be in the premium tier?"),
    _msg(6, "user", "Advanced analytics, team collaboration, and API access."),
    _msg(7, "assistant", "That's a solid package. What's the target price point?"),
    _msg(8, "user", "£49/month per team. The sales team thinks it'll convert well."),
    _msg(9, "assistant", "That's competitive with the market. Notion charges £48 for teams."),
    _msg(10, "user", "Exactly. We need to ship this by end of July."),
    _msg(11, "assistant", "That's 8 weeks. It'll be tight for all three features."),
    _msg(12, "user", "What can we cut if needed?"),
    _msg(13, "assistant", "API access is the biggest lift. We could push that to Q4."),
    _msg(14, "user", "Let's keep it in scope for now and revisit in week 4."),
    _msg(15, "assistant", "Agreed. What about retention features?"),
    _msg(16, "user", "Onboarding improvements. Our week-1 drop-off is 40%."),
    _msg(17, "assistant", "That's high. What's causing it?"),
    _msg(18, "user", "Users don't get to the aha moment fast enough."),
    _msg(19, "assistant", "We should add an interactive onboarding flow. 3-5 steps max."),
    _msg(20, "user", "Yes. And personalisation based on their use case."),
    _msg(21, "assistant", "That's a good retention lever. Typeform uses this effectively."),
    _msg(22, "user", "Let's scope it as a 2-week project for one engineer."),
    _msg(23, "assistant", "Assigned to Sarah. She'll own the onboarding revamp."),
    _msg(24, "user", "Great. What's the success metric?"),
    _msg(25, "assistant", "Reduce week-1 drop-off from 40% to below 25%."),
    _msg(26, "user", "That's our target. The onboarding success metric is week-1 retention above 75%."),
    _msg(27, "assistant", "Noted. I'll add that to the OKRs."),
    _msg(28, "user", "What about the mobile app? It's been requested a lot."),
    _msg(29, "assistant", "It's on 60% of customer feedback. But it's a 3-month project minimum."),
    _msg(30, "user", "Push it to Q4. We can't afford the distraction now."),
    _msg(31, "assistant", "Agreed. Mobile app is Q4."),
    _msg(32, "user", "Let's also do a competitor analysis in the first week."),
    _msg(33, "assistant", "I'll handle that. Deliverable by end of week 1."),
    _msg(34, "user", "Perfect. What's our engineering headcount for Q3?"),
    _msg(35, "assistant", "6 engineers. 4 on premium tier, 1 on onboarding (Sarah), 1 on infra."),
    _msg(36, "user", "That works. Budget?"),
    _msg(37, "assistant", "Q3 engineering budget is £240,000. That's within the allocation."),
    _msg(38, "user", "Good. Q3 budget approved at £240,000."),
    _msg(39, "assistant", "I'll update the headcount plan accordingly."),
    _msg(40, "user", "Let's set a midpoint review for week 4."),
    _msg(41, "assistant", "Week 4 review scheduled — that's 28 July."),
    _msg(42, "user", "If we're behind on the premium tier, we'll cut API access then."),
    _msg(43, "assistant", "Understood. API access is the contingency cut."),
    _msg(44, "user", "Any risks I should know about?"),
    _msg(45, "assistant", "Payment integration is the highest risk — Stripe's docs have gaps."),
    _msg(46, "user", "Let's spike that in week 1. Two days max."),
    _msg(47, "assistant", "I'll assign that spike to Marcus. He knows Stripe best."),
    _msg(48, "user", "Good. Final summary?"),
    _msg(49, "assistant", "Q3 plan: premium tier by end of July at £49/month, onboarding revamp with Sarah targeting 75% week-1 retention, mobile to Q4, budget £240k, week-4 review on 28 July."),
    _msg(50, "user", "Perfect. Let's go."),
]

CONV_2_QUERIES = [
    {"query": "What is the Q3 budget and who approved it?", "type": "factual"},
    {"query": "What are the trade-offs we considered for the roadmap?", "type": "analytical"},
    {"query": "What are all the deadlines and owners from this planning session?", "type": "factual"},
    {"query": "What is the current plan for the mobile app?", "type": "status"},
]


# ── Conversation 3: Debugging a production incident ───────────────────────────
CONV_3 = [
    _msg(0, "user", "We have a P1 incident. API latency is at 4 seconds, SLA is 500ms."),
    _msg(1, "assistant", "On it. What's the error rate?"),
    _msg(2, "user", "5% of requests are 500s. Started 20 minutes ago."),
    _msg(3, "assistant", "Any recent deployments?"),
    _msg(4, "user", "Yes. v2.3.1 shipped at 14:00 UTC, incident started at 14:15."),
    _msg(5, "assistant", "High correlation. Let's look at what changed in v2.3.1."),
    _msg(6, "user", "It includes the new recommendation engine integration."),
    _msg(7, "assistant", "That's the likely culprit. Is the recommendation service healthy?"),
    _msg(8, "user", "Checking now. It shows 98% CPU utilisation."),
    _msg(9, "assistant", "It's saturated. The new integration probably isn't rate-limiting its calls."),
    _msg(10, "user", "You're right. There's no rate limiter in the new code."),
    _msg(11, "assistant", "Immediate fix: roll back to v2.3.0 to restore service."),
    _msg(12, "user", "Agreed. Rolling back now. ETA 5 minutes."),
    _msg(13, "assistant", "Good. While that runs, let's plan the proper fix."),
    _msg(14, "user", "Rollback complete. Latency back to 120ms. Error rate 0%."),
    _msg(15, "assistant", "Service restored. Root cause: missing rate limiter on recommendation service calls."),
    _msg(16, "user", "The fix is to add a rate limiter — max 100 calls/sec to the recommendation service."),
    _msg(17, "assistant", "And we should add a circuit breaker so if the service degrades, we fail gracefully."),
    _msg(18, "user", "Good point. We decided: rate limiter at 100 calls/sec plus circuit breaker before re-deploying."),
    _msg(19, "assistant", "I'll implement that. Timeline — can I have until tomorrow 10am?"),
    _msg(20, "user", "Yes. Fix deadline is tomorrow 10am UTC."),
    _msg(21, "assistant", "I'll also add load tests so this doesn't happen again."),
    _msg(22, "user", "Add them to the CI pipeline too."),
    _msg(23, "assistant", "Will do. Load tests added to CI as an action item."),
    _msg(24, "user", "Who should we notify about the incident?"),
    _msg(25, "assistant", "Customer success for affected enterprise accounts. I'll draft the comms."),
    _msg(26, "user", "And an internal postmortem?"),
    _msg(27, "assistant", "I'll schedule it for Thursday. I'll own the postmortem document."),
    _msg(28, "user", "Good. Blameless postmortem — focus on systems, not people."),
    _msg(29, "assistant", "Understood. That's our standard."),
    _msg(30, "user", "What's the incident duration?"),
    _msg(31, "assistant", "From 14:15 to 14:45 UTC. 30 minutes total."),
    _msg(32, "user", "And estimated customer impact?"),
    _msg(33, "assistant", "Approximately 2,400 affected requests based on traffic logs."),
    _msg(34, "user", "Is that within our SLA breach threshold?"),
    _msg(35, "assistant", "Yes, but only just. Our SLA allows 99.9% uptime — we're at 99.93% for the month."),
    _msg(36, "user", "Good. We're still compliant. Document that."),
    _msg(37, "assistant", "Noted in the incident report."),
    _msg(38, "user", "Should we add monitoring for recommendation service CPU?"),
    _msg(39, "assistant", "Yes. Alert at 70% CPU with a PagerDuty integration."),
    _msg(40, "user", "We agreed: PagerDuty alert at 70% CPU for recommendation service."),
    _msg(41, "assistant", "I'll set that up alongside the rate limiter fix."),
    _msg(42, "user", "What's the full action item list?"),
    _msg(43, "assistant", "1. Rate limiter (100/sec) + circuit breaker — due 10am tomorrow. 2. Load tests in CI — this week. 3. PagerDuty alert at 70% CPU — this week. 4. Postmortem Thursday. 5. Customer comms — today."),
    _msg(44, "user", "Good. Who owns each?"),
    _msg(45, "assistant", "All technical items are mine. Customer comms — Sarah. Postmortem doc — me."),
    _msg(46, "user", "Confirmed. Let's debrief after the fix ships."),
    _msg(47, "assistant", "Will do. I'll ping you once v2.3.2 is in staging."),
    _msg(48, "user", "Thanks. Good response time on this."),
    _msg(49, "assistant", "Thanks. Let's make sure this class of bug doesn't recur."),
    _msg(50, "user", "Agreed. The load tests in CI will help with that."),
]

CONV_3_QUERIES = [
    {"query": "What was the root cause of the incident and what was decided to fix it?", "type": "factual"},
    {"query": "What are all the action items and their owners?", "type": "factual"},
    {"query": "What were the trade-offs discussed in the incident response?", "type": "analytical"},
]


# ── Registry of all conversations ────────────────────────────────────────────

ALL_CONVERSATIONS = [
    {"id": "conv_1", "title": "Backend DB Architecture", "messages": CONV_1, "queries": CONV_1_QUERIES},
    {"id": "conv_2", "title": "Q3 Product Roadmap", "messages": CONV_2, "queries": CONV_2_QUERIES},
    {"id": "conv_3", "title": "Production Incident Response", "messages": CONV_3, "queries": CONV_3_QUERIES},
]


def get_all_eval_cases() -> list[dict]:
    """
    Flatten all conversations × queries into a list of eval cases.
    Each case: {conversation_id, title, messages, query, query_type}
    """
    cases = []
    for conv in ALL_CONVERSATIONS:
        for q in conv["queries"]:
            cases.append({
                "conversation_id": conv["id"],
                "title": conv["title"],
                "messages": conv["messages"],
                "query": q["query"],
                "query_type": q["type"],
            })
    return cases
