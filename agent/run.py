from state import AgentState


def main():
    state: AgentState = {
        "messages": [],
        "goal": "Test state creation",
        "iterations": 0,
        "final_answer": None,
    }
    print(state)


if __name__ == "__main__":
    main()