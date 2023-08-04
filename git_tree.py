import requests

def fetch_repository_tree(owner, repo, access_token):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/master?recursive=1"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch repository tree. Status code: {response.status_code}")

def generate_ascii_tree(node, indent="", is_last=True):
    connector = "└─" if is_last else "├─"
    tree_repr = f"{indent}{connector} {node['path']}\n"

    if 'tree' in node:
        for idx, child in enumerate(node['tree']):
            is_last_child = idx == len(node['tree']) - 1
            tree_repr += generate_ascii_tree(child, indent + ("    " if is_last else "│   "), is_last_child)

    return tree_repr

def main():
    owner = "faisal-saddique"
    repo = "LawyerAI"
    access_token = "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"
    output_file = "repository_tree.txt"

    try:
        repo_tree = fetch_repository_tree(owner, repo, access_token)
        ascii_tree = generate_ascii_tree(repo_tree)

        with open(output_file, "w") as file:
            file.write(ascii_tree)

        print(f"ASCII tree diagram saved to {output_file}")

    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()
