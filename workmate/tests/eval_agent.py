"""
tests/eval_agent.py

Evaluates the Supervisor Agent's routing accuracy.
The supervisor must correctly route queries to the right sub-agent.
"""
from app.agent.state import AgentState
from app.agent.supervisor import supervisor_node, AGENT_RESEARCH, AGENT_PRODUCTIVITY, AGENT_CHITCHAT

SAMPLES = [
    # (query, expected_next_agent)
    ("What is the launch date according to the project plan document?", AGENT_RESEARCH),
    ("What was the timezone I said I prefer?",                          AGENT_RESEARCH),
    ("Please extract all action items from this meeting transcript.",   AGENT_PRODUCTIVITY),
    ("Draft an email to John summarizing our discussion.",              AGENT_PRODUCTIVITY),
    ("Hello, how are you today?",                                       AGENT_CHITCHAT),
    ("What can you do?",                                                AGENT_CHITCHAT),
]


def run_eval():
    correct = 0
    total = len(SAMPLES)
    print("=" * 60)
    print("Running Supervisor Routing Evaluation")
    print("=" * 60)

    for idx, (query, expected) in enumerate(SAMPLES, 1):
        state = AgentState(current_query=query)
        try:
            result = supervisor_node(state)
            predicted = result.get("next_agent", "unknown")
            status = "PASS" if predicted == expected else "FAIL"
            if predicted == expected:
                correct += 1
            print(
                f"[{idx}/{total}] {status}\n"
                f"  Query    : {query}\n"
                f"  Expected : {expected}\n"
                f"  Got      : {predicted}\n"
            )
        except Exception as exc:
            print(f"[{idx}/{total}] ERROR -- {exc}\n")

    print("=" * 60)
    print(f"Accuracy: {correct}/{total} ({(correct / total) * 100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    run_eval()

