"""
app/agent/local_llm.py

The fallback rule-based matching engine used if no external API keys are provided or if the network fails.
"""

import re
import json
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

# ── Intent classification ─────────────────────────────────────────────────

_INTENT_PATTERNS: List[Tuple[str, List[str]]] = [
    ("document_qa", [
        r"\b(what|how|when|where|who|why|explain|tell me|describe|summarize|find|search)\b.*(document|file|pdf|report|text|uploaded|attachment)",
        r"\b(according to|based on|in the document|from the file)\b",
        r"\b(search|look up|find in)\b.*\b(document|file|report)\b",
        r"\bdocument.*(about|say|mention|contain|explain)\b",
    ]),
    ("memory_recall", [
        r"\b(remember|recall|what did i|do you know my|my preference|i told you|previously|last time)\b",
        r"\b(what.*(my|i).*(said|told|mentioned|prefer|like|dislike|want))\b",
        r"\b(my (name|email|timezone|role|preference|setting))\b",
        r"\b(have i (ever|before|previously))\b",
    ]),
    ("task_creation", [
        r"\b(create|make|add|extract|generate|list|draft)\b.*(task|tasks|action item|todo|to-do|action|ticket)\b",
        r"\b(task|tasks|action items?|todos?|to-dos?)\b.*(from|based on|for)\b",
        r"\bplease (create|make|generate|add|extract)\b.*\b(task|action)\b",
        r"\b(assign|schedule|track)\b.*(task|work|item)\b",
        r"\bextract.*(action|task|item)\b",
    ]),
    ("email_draft", [
        r"\b(write|draft|compose|create|send|prepare)\b.*(email|mail|message|letter)\b",
        r"\bemail.*(to|for|about|regarding)\b",
        r"\b(reach out|follow up|reply|respond)\b.*(via email|by email|email)\b",
    ]),
    ("multi_step", [
        r"\b(and also|then|after that|additionally|furthermore|as well as)\b.*(and|also)\b",
        r"\b(first|second|third|finally|lastly)\b.*\b(then|and|also)\b",
    ]),
]

_CHITCHAT_TRIGGERS = [
    r"\b(hello|hi|hey|howdy|greetings|good morning|good afternoon|good evening)\b",
    r"\b(how are you|how do you do|what's up|how's it going)\b",
    r"\b(thank you|thanks|appreciate|great|awesome|cool|nice)\b",
    r"\b(bye|goodbye|see you|take care|later)\b",
    r"\bwhat (can|do) you do\b",
    r"\bwho are you\b",
    r"\bhelp\b",
]


def classify_intent(query: str) -> Dict[str, Any]:
    """
    Rule-based intent classifier.
    Returns dict with intent, confidence, and reasoning.
    """
    q = query.lower().strip()

    # Check chitchat first (short messages or greetings)
    for pattern in _CHITCHAT_TRIGGERS:
        if re.search(pattern, q):
            return {
                "intent": "chitchat",
                "confidence": 0.82,
                "reasoning": "Message matches common conversational pattern.",
            }

    # Check structured intents
    scores: Dict[str, int] = {}
    for intent, patterns in _INTENT_PATTERNS:
        score = sum(1 for p in patterns if re.search(p, q))
        if score > 0:
            scores[intent] = score

    if not scores:
        # Default: short messages → chitchat, longer → document_qa
        if len(q.split()) <= 6:
            return {
                "intent": "chitchat",
                "confidence": 0.60,
                "reasoning": "Short query with no matching patterns; defaulting to chitchat.",
            }
        return {
            "intent": "document_qa",
            "confidence": 0.55,
            "reasoning": "No strong pattern match; treating as a document question.",
        }

    best = max(scores, key=scores.get)
    confidence_map = {1: 0.70, 2: 0.82, 3: 0.90}
    confidence = confidence_map.get(scores[best], 0.95)

    return {
        "intent": best,
        "confidence": confidence,
        "reasoning": f"Matched {scores[best]} keyword pattern(s) for intent '{best}'.",
    }


# ── Chitchat responses ────────────────────────────────────────────────────

def generate_chitchat(query: str, memories: str = "") -> str:
    """Rule-based chitchat responder."""
    q = query.lower()

    if re.search(r"\b(hello|hi|hey|howdy|good morning|good afternoon|good evening)\b", q):
        return (
            "👋 Hello! I'm **WorkMate**, your AI workspace assistant.\n\n"
            "I can help you with:\n"
            "- 📄 **Document Q&A** — Upload a PDF/DOCX/TXT and ask questions about it\n"
            "- ✅ **Task creation** — Extract action items from meeting notes or text\n"
            "- ✉️ **Email drafting** — Compose professional emails\n"
            "- 🧠 **Memory** — I remember your preferences across conversations\n\n"
            "What would you like to do today?"
        )

    if re.search(r"\bhow are you\b", q):
        return "I'm doing great, thank you for asking! 😊 Ready to help you get things done. What's on your agenda today?"

    if re.search(r"\bwhat (can|do) you do\b", q):
        return (
            "I'm **WorkMate** — here's what I can do:\n\n"
            "| Feature | How to use |\n"
            "|---------|------------|\n"
            "| 📄 Document Q&A | Upload a file in the sidebar, then ask questions |\n"
            "| ✅ Task extraction | Say \"Create tasks from: [your text]\" |\n"
            "| ✉️ Email drafting | Say \"Draft an email to [person] about [topic]\" |\n"
            "| 🧠 Memory recall | Say \"What do you remember about me?\" |\n"
            "| 🔍 RAG search | Ask questions about your uploaded documents |"
        )

    if re.search(r"\bwho are you\b", q):
        return (
            "I'm **WorkMate** 🤖, an AI workspace assistant built with LangGraph + ChromaDB.\n\n"
            "I run locally on your machine with a rule-based fallback engine active "
            "(no API key required for core features)."
        )

    if re.search(r"\b(thank you|thanks|appreciate)\b", q):
        return "You're welcome! 😊 Is there anything else I can help you with?"

    if re.search(r"\b(bye|goodbye|see you|take care)\b", q):
        return "Goodbye! 👋 Have a productive day. Come back anytime you need help."

    if re.search(r"\bhelp\b", q):
        return (
            "Sure! Here are some things you can try:\n\n"
            "1. **Upload a document** (PDF/DOCX/TXT) in the left sidebar\n"
            "2. Ask: *\"Summarize the uploaded document\"*\n"
            "3. Ask: *\"Create tasks from: [paste meeting notes]\"*\n"
            "4. Ask: *\"Draft an email to John about the project deadline\"*\n"
            "5. Ask: *\"What do you remember about me?\"*\n\n"
            "💡 **Tip:** Add your Anthropic API key to `.env` to unlock the full AI-powered experience!"
        )

    if memories and memories != "No memories yet.":
        return (
            f"Here's what I recall about you from our previous conversations:\n\n{memories}\n\n"
            "Is there anything specific you'd like to discuss or do?"
        )

    return (
        "I'm in **rule-based mode** (no API key set). I can still:\n"
        "- Answer questions about uploaded documents (RAG search)\n"
        "- Extract tasks from text\n"
        "- Draft basic emails\n"
        "- Remember facts from our conversation\n\n"
        "For the full AI-powered experience, add your `ANTHROPIC_API_KEY` to the `.env` file."
    )


# ── RAG response ──────────────────────────────────────────────────────────

def generate_rag_response(context: str, query: str, rag_confidence: str) -> str:
    """Synthesize a response from retrieved document chunks."""
    if not context:
        return (
            "🔍 I searched your uploaded documents but couldn't find relevant information.\n\n"
            "**Try:**\n"
            "- Uploading a document that covers this topic (sidebar → Upload)\n"
            "- Rephrasing your question\n"
            "- Checking the **Documents** section to confirm the file was processed"
        )

    lines = []
    if rag_confidence == "Medium":
        lines.append("⚠️ *Moderate confidence — please verify with the source document.*\n")

    lines.append(f"**Based on your documents:**\n")

    # Extract key sentences from context that relate to the query
    query_words = set(re.sub(r"[^\w\s]", "", query.lower()).split())
    query_words -= {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                    "when", "where", "who", "why", "tell", "me", "about", "please"}

    chunks = context.split("\n\n")
    relevant_parts = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        hits = sum(1 for w in query_words if w in chunk_lower)
        if hits > 0:
            relevant_parts.append((hits, chunk))

    relevant_parts.sort(reverse=True)

    if relevant_parts:
        # Show top 2 most relevant chunks
        for _, chunk in relevant_parts[:2]:
            # Extract the source header if present
            source_match = re.match(r"\[Source: (.+?)\]", chunk)
            if source_match:
                lines.append(f"\n📌 *{source_match.group(0)}*")
            # Add the content (first 500 chars)
            content = re.sub(r"\[Source: .+?\]\n?", "", chunk).strip()
            lines.append(content[:600] + ("…" if len(content) > 600 else ""))
    else:
        # Fall back to showing first chunk
        first = chunks[0]
        content = re.sub(r"\[Source: .+?\]\n?", "", first).strip()
        lines.append(content[:600] + ("…" if len(content) > 600 else ""))

    lines.append(
        "\n\n💡 *Add your `ANTHROPIC_API_KEY` to `.env` for a more refined, synthesized answer.*"
    )

    return "\n".join(lines)


# ── Task extraction ───────────────────────────────────────────────────────

def extract_tasks(query: str) -> List[Dict[str, Any]]:
    """
    Rule-based task extractor.
    Looks for action verbs and imperative structures.
    """
    # Remove the "create tasks from:" prefix if present
    text = re.sub(
        r"^(please\s+)?(create|make|add|extract|generate|list)\s+(tasks?|action\s+items?|todos?)\s*(from|:)?\s*",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip()

    if not text:
        text = query

    tasks = []
    seen = set()

    # Split into sentences
    sentences = re.split(r"[.!?\n;]|(?:\band\b)", text)

    action_verbs = (
        r"\b(review|update|send|write|create|schedule|complete|prepare|"
        r"finalize|submit|present|call|contact|fix|implement|deploy|test|"
        r"analyze|draft|approve|arrange|coordinate|follow up|check|verify|"
        r"discuss|plan|design|build|deliver|share|document|assign)\b"
    )

    priority_keywords = {
        "High": r"\b(urgent|asap|immediately|critical|priority|important|deadline|today)\b",
        "Low": r"\b(eventually|sometime|nice to have|optional|low priority|whenever)\b",
    }

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 5:
            continue

        # Only extract sentences with action verbs
        if not re.search(action_verbs, sentence, re.IGNORECASE):
            continue

        # Build a clean title (capitalize first word, max 60 chars)
        title = sentence.strip().rstrip(".,;:")
        if len(title) > 60:
            title = title[:57] + "…"
        title = title[0].upper() + title[1:] if title else title

        key = title.lower()[:40]
        if key in seen:
            continue
        seen.add(key)

        # Determine priority
        priority = "Medium"
        for p, pattern in priority_keywords.items():
            if re.search(pattern, sentence, re.IGNORECASE):
                priority = p
                break

        # Attempt to extract owner (e.g., "John to review" or "assign to Sarah")
        owner_match = re.search(
            r"(?:assign(?:ed)? to|by|owner[:\s]+)([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)",
            sentence,
        )
        owner = owner_match.group(1) if owner_match else "Unassigned"

        tasks.append({
            "title": title,
            "description": sentence.strip(),
            "priority": priority,
            "owner": owner,
        })

    # Fallback: if nothing extracted, treat the whole query as one task
    if not tasks:
        tasks.append({
            "title": text[:60].strip(),
            "description": text.strip(),
            "priority": "Medium",
            "owner": "Unassigned",
        })

    return tasks[:8]  # cap at 8 tasks


# ── Email drafting ────────────────────────────────────────────────────────

def draft_email(query: str) -> Dict[str, str]:
    """Rule-based email draft generator.
    Handles both short user queries AND full document context blobs from RAG.
    """
    q = query

    # ── Detect whether we received document context or a short user query ──
    is_doc_context = len(q) > 300 or "[Source:" in q

    if is_doc_context:
        # ── Document-context mode: extract key info from the document ─────
        # Strip [Source: ...] markers for cleaner parsing
        clean = re.sub(r"\[Source:[^\]]+\]\n?", "", q).strip()

        # Try to find an addressee in the document
        to_match = re.search(
            r"(?:to|from|cc|attendees?|participants?|team)[:\s]+"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            clean,
        )
        recipient = to_match.group(1) if to_match else "the team"

        # Try to find the project / topic name
        project_match = re.search(
            r"(?:project|re:|subject:|regarding|sprint|module)[:\s]+(.+?)(?:\n|\.|,)",
            clean,
            re.IGNORECASE,
        )
        topic = (
            project_match.group(1).strip()[:70]
            if project_match
            else "the discussed topics"
        )

        # Pull out action sentences as key points
        action_pat = (
            r"\b(review|update|send|create|schedule|complete|prepare|finalize|"
            r"submit|deploy|test|analyze|approve|deliver|assign|fix|implement|"
            r"coordinate|discuss|plan|build|share|document)\b"
        )
        key_points = []
        for line in clean.splitlines():
            line = line.strip()
            if len(line) > 20 and re.search(action_pat, line, re.IGNORECASE):
                key_points.append(f"\u2022 {line.rstrip('.,;:')}")
                if len(key_points) >= 4:
                    break

        subject = topic.rstrip(".,;:")
        body = (
            f"Hi {recipient},\n\n"
            f"I hope this message finds you well.\n\n"
            f"I'm writing to follow up on **{topic}**.\n\n"
        )
        if key_points:
            body += "Key action items from our records:\n" + "\n".join(key_points) + "\n\n"
        body += (
            "Please let me know if you have any questions or need further clarification.\n\n"
            "Best regards,\n"
            "[Your Name]"
        )

        return {
            "to": recipient,
            "subject": subject,
            "body": body,
            "suggested_recipients": recipient,
        }

    # ── Short-query mode: original behaviour ─────────────────────────────
    to_match = re.search(r"\bto\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b", q)
    recipient = to_match.group(1) if to_match else "the recipient"

    about_match = re.search(r"\babout\s+(.+?)(?:\.|$)", q, re.IGNORECASE)
    topic = about_match.group(1).strip() if about_match else query

    subject = topic[:60].capitalize()
    if not subject.endswith("."):
        subject = subject.rstrip(".,;:")

    body = (
        f"Hi {recipient},\n\n"
        f"I hope this message finds you well.\n\n"
        f"I'm reaching out regarding **{topic}**. "
        f"I'd like to discuss this further and explore how we can move forward effectively.\n\n"
        f"Please let me know your availability for a brief conversation, "
        f"or feel free to reply with your thoughts.\n\n"
        f"Best regards,\n"
        f"[Your Name]"
    )

    return {
        "to": recipient,
        "subject": subject,
        "body": body,
        "suggested_recipients": recipient,
    }


# ── Memory extraction ─────────────────────────────────────────────────────

def extract_memories_local(conversation: str) -> List[Dict[str, Any]]:
    """Rule-based memory extractor from conversation text."""
    memories = []
    lines = conversation.split("\n")

    preference_patterns = [
        (r"\bI (prefer|like|love|enjoy|want|use|work with)\b(.+)", "preference"),
        (r"\bmy (name|email|role|timezone|team|company)\s+is\s+(.+)", "semantic"),
        (r"\bI('m| am) (a|an|the)\s+(.+)", "semantic"),
        (r"\bI (work at|work for|work in)\s+(.+)", "semantic"),
        (r"\bI (don't|do not|hate|dislike|avoid)\b(.+)", "preference"),
    ]

    for line in lines:
        line = line.strip()
        if not line.startswith("User:"):
            continue

        user_text = line[5:].strip()

        for pattern, mem_type in preference_patterns:
            match = re.search(pattern, user_text, re.IGNORECASE)
            if match:
                content = user_text
                memories.append({
                    "type": mem_type,
                    "content": content,
                    "importance_score": 0.75,
                })
                break

    return memories[:5]  # cap at 5 per turn
