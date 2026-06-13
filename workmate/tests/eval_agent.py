import json
from app.agent.state import AgentState
from app.agent.nodes import intent_detection

def run_eval():
    samples = [
        {"query": "What is the launch date according to the project plan document?", "expected": "document_qa"},
        {"query": "What was the timezone I said I prefer?", "expected": "memory_recall"},
        {"query": "Please extract all the action items from this meeting transcript and create tasks.", "expected": "task_creation"},
        {"query": "Draft an email to John summarizing our discussion.", "expected": "email_draft"},
        {"query": "Summarize the requirements doc AND create tasks for the team.", "expected": "multi_step"},
        {"query": "Hello, how are you today?", "expected": "chitchat"}
    ]
    
    correct = 0
    print("Running Agent Intent Evaluation...")
    for idx, sample in enumerate(samples):
        state = AgentState(
            current_query=sample["query"],
            intent="",
            confidence=0.0,
            reasoning=""
        )
        try:
            result_state = intent_detection(state)
            predicted = result_state["intent"]
            
            if predicted == sample["expected"]:
                correct += 1
                print(f"[{idx+1}/{len(samples)}] PASS | Expected: {sample['expected']} | Got: {predicted}")
            else:
                print(f"[{idx+1}/{len(samples)}] FAIL | Expected: {sample['expected']} | Got: {predicted} | Reasoning: {result_state.get('reasoning')}")
        except Exception as e:
            print(f"[{idx+1}/{len(samples)}] ERROR: {e}")
            
    print(f"\nAccuracy: {correct}/{len(samples)} ({(correct/len(samples))*100:.1f}%)")

if __name__ == "__main__":
    run_eval()
