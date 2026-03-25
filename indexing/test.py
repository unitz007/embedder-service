from full_pipeline import build_knowledge_base

if __name__ == "__main__":
    repo = "/Users/charles/.dotfiles"
    question = "Implement a dotfile syncing system, that syncs across devices if there is a push to the dotfile main github repository"
    kb = build_knowledge_base(repo_path=repo, include_dotfiles=True)
    answer = kb.ask(question, model='gpt-oss:20b-cloud')

    print(answer)