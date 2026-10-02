#!/usr/bin/env python3
"""Interactive CLI for WikiRAG."""

import os
import sys
import argparse
from src.wikirag import (
    create_index,
    create_react_agent,
    run_agent,
    SUPPORTED_MODELS,
    get_api_key
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="WikiRAG: Autonomous ReAct Agent & Multi-Topic Wikipedia RAG"
    )
    parser.add_argument(
        "--pages",
        type=str,
        default="Paris, Batman, Python",
        help="Comma-separated Wikipedia page titles to index (default: 'Paris, Batman, Python')"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=SUPPORTED_MODELS,
        default="mock-mode" if not get_api_key() else "gpt-4o-mini",
        help="Model choice (defaults to mock-mode if OPENAI_API_KEY is unset)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single question to query the agent with (if omitted, starts interactive chat)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 65)
    print("  WikiRAG -- Autonomous Wikipedia ReAct Agent")
    print("=" * 65)
    print(f"Model       : {args.model}")
    print(f"Target Pages: {args.pages}")
    print("=" * 65)

    print(f"\n[1/2] Indexing Wikipedia pages: '{args.pages}'...")
    try:
        index = create_index(args.pages, model_name=args.model)
        print("[OK] Indexing completed successfully!")
    except Exception as e:
        print(f"[ERROR] Indexing failed: {e}")
        sys.exit(1)

    print(f"\n[2/2] Initializing ReAct Agent with '{args.model}'...")
    agent = create_react_agent(args.model, index)
    print("[OK] Agent ready!\n")

    # If single query supplied
    if args.query:
        print(f"User: {args.query}\n")
        response = run_agent(agent, args.query)
        print(f"Agent:\n{response}")
        return

    # Interactive chat loop
    print("Type your question below (or 'exit' / 'quit' to stop):")
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting WikiRAG. Goodbye!")
                break

            response = run_agent(agent, user_input)
            print(f"\nAgent:\n{response}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

if __name__ == "__main__":
    main()
