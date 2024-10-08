import os
import re
import json
from collections import defaultdict
import subprocess


def get_tracked_files():
    result = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    return set(result.stdout.splitlines())


def extract_info_from_md(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    match = re.search(r"---\s*title:\s*(.*?)\s*type:\s*(.*?)\s*---", content, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None


def extract_info_from_ipynb(file_path):
    with open(file_path, "r") as f:
        notebook = json.load(f)

    first_cell = notebook["cells"][0]
    if first_cell["cell_type"] == "markdown":
        content = "".join(first_cell["source"])
        match = re.search(
            r"<!--\s*---\s*title:\s*(.*?)\s*type:\s*(.*?)\s*---\s*-->",
            content,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip(), match.group(2).strip()
    return None, None


def get_service(path):
    if "gke" in path:
        return "GKE"
    elif "vertex-ai" in path:
        return "Vertex AI"
    elif "cloud-run" in path:
        return "Cloud Run"
    return None


def generate_tables():
    examples = defaultdict(lambda: defaultdict(list))
    root_dir = "examples"
    tracked_files = get_tracked_files()

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename in ["README.md", "vertex-notebook.ipynb"]:
                file_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(file_path, start=".")

                if relative_path not in tracked_files:
                    continue

                dir_path = os.path.dirname(relative_path)

                if filename.endswith(".md"):
                    title, example_type = extract_info_from_md(file_path)
                elif filename.endswith(".ipynb"):
                    title, example_type = extract_info_from_ipynb(file_path)

                if title and example_type:  # type: ignore
                    service = get_service(relative_path)
                    if service:
                        examples[service][example_type].append((dir_path, title))

    return examples


def update_readme(examples):
    with open("README.md", "r") as f:
        content = f.read()

    ordered_services = ["Vertex AI", "GKE", "Cloud Run"]

    for example_type in ["training", "inference", "evaluation"]:
        table_rows = []
        for service in ordered_services:
            if examples[service].get(example_type):
                for path, title in sorted(
                    examples[service][example_type], key=lambda x: x[1]
                ):
                    # Format the path to include 'examples/<service>'
                    formatted_path = (
                        f"examples/{service.lower().replace(' ', '-')}/{path}"
                    )
                    table_rows.append(
                        (
                            service,
                            f"[{formatted_path}](./{formatted_path})",
                            title,
                        )
                    )

        if table_rows:
            table = format_table(["Service", "Example", "Title"], table_rows)
            pattern = (
                rf"(### {example_type.capitalize()} Examples\n\n)[\s\S]*?(\n\n###|\Z)"
            )
            replacement = rf"\1{table}\2"
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    with open("README.md", "w") as f:
        f.write(content.rstrip() + "\n")


def update_docs(examples):
    with open("docs/source/resources.mdx", "r") as f:
        content = f.read()

    new_content = []
    ordered_services = ["Vertex AI", "GKE", "Cloud Run"]
    ordered_types = ["inference", "training", "evaluation"]

    for service in ordered_services:
        service_name = f"(Preview) {service}" if service == "Cloud Run" else service
        new_content.append(f"\n### {service_name}\n")

        for example_type in ordered_types:
            if examples[service].get(example_type):
                new_content.append(f"\n- {example_type.capitalize()}\n\n")
                for path, title in sorted(
                    examples[service][example_type], key=lambda x: x[1]
                ):
                    github_url = f"https://github.com/huggingface/Google-Cloud-Containers/tree/main/{path}"
                    new_content.append(f"  - [{title}]({github_url})\n")

    new_examples_content = "".join(new_content)

    # Replace the Examples section in the original content
    pattern = r"(## Examples\n\n- \[All examples\].*?\n)[\s\S]*"
    updated_content = re.sub(
        pattern, rf"\1{new_examples_content}", content, flags=re.DOTALL
    )

    with open("docs/source/resources.mdx", "w") as f:
        f.write(updated_content)


def update_cloud_run_examples(examples):
    file_path = "examples/cloud-run/README.md"

    with open(file_path, "r") as f:
        content = f.read()

    # Update Inference Examples
    inference_examples = examples.get("Cloud Run", {}).get("inference", [])
    inference_table = format_table(
        ["Example", "Title"],
        [
            (f"[{os.path.basename(path)}](./{os.path.basename(path)})", title)
            for path, title in sorted(inference_examples, key=lambda x: x[1])
        ],
    )

    inference_pattern = r"(## Inference Examples\n\n)[\s\S]*?(\n\n## Training Examples)"
    inference_replacement = rf"\1{inference_table}\2"
    content = re.sub(inference_pattern, inference_replacement, content, flags=re.DOTALL)

    # Update Training Examples
    training_pattern = r"(## Training Examples\n\n)[\s\S]*"
    training_replacement = r"\1Coming soon!"
    content = re.sub(training_pattern, training_replacement, content, flags=re.DOTALL)

    with open(file_path, "w") as f:
        f.write(content)


def update_gke_examples(examples):
    file_path = "examples/gke/README.md"

    with open(file_path, "r") as f:
        content = f.read()

    for example_type in ["Training", "Inference"]:
        examples_list = examples.get("GKE", {}).get(example_type.lower(), [])
        pattern = rf"(## {example_type} Examples\n\n)[\s\S]*?(\n\n##|\Z)"

        if examples_list:
            # Sort examples alphabetically by their basename
            sorted_examples = sorted(
                examples_list, key=lambda x: os.path.basename(x[0])
            )
            table = format_table(
                ["Example", "Title"],
                [
                    (f"[{os.path.basename(path)}](./{os.path.basename(path)})", title)
                    for path, title in sorted_examples
                ],
            )
            replacement = rf"\1{table}\2"
        else:
            replacement = rf"\1No {example_type.lower()} examples available yet.\2"

        content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    with open(file_path, "w") as f:
        f.write(content.rstrip() + "\n")


def update_vertex_ai_examples(examples):
    file_path = "examples/vertex-ai/README.md"

    with open(file_path, "r") as f:
        content = f.read()

    new_content = []
    for line in content.split("\n"):
        if line.startswith("## Notebooks"):
            new_content.append(line)
            break
        new_content.append(line)

    for example_type in ["Training", "Inference", "Evaluation"]:
        examples_list = examples.get("Vertex AI", {}).get(example_type.lower(), [])
        new_content.append(f"\n### {example_type} Examples\n")
        if examples_list:
            table = format_table(
                ["Example", "Title"],
                [
                    (
                        f"[notebooks/{os.path.basename(path)}](./notebooks/{os.path.basename(path)})",
                        title,
                    )
                    for path, title in sorted(examples_list, key=lambda x: x[1])
                ],
            )
            new_content.append(table)
        else:
            new_content.append("Coming soon!")

    # Handle Pipelines section
    new_content.append("\n## Pipelines\n")
    pipeline_examples = examples.get("Vertex AI", {}).get("pipeline", [])
    if pipeline_examples:
        table = format_table(
            ["Example", "Title"],
            [
                (
                    f"[pipelines/{os.path.basename(path)}](./pipelines/{os.path.basename(path)})",
                    title,
                )
                for path, title in sorted(pipeline_examples, key=lambda x: x[1])
            ],
        )
        new_content.append(table)
    else:
        new_content.append("Coming soon!")

    with open(file_path, "w") as f:
        f.write("\n".join(new_content).strip())


def format_table(headers, rows):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    header = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + " |"
    separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
    body = [
        "| " + " | ".join(f"{cell:<{w}}" for cell, w in zip(row, col_widths)) + " |"
        for row in rows
    ]

    return "\n".join([header, separator] + body)


if __name__ == "__main__":
    examples = generate_tables()
    update_readme(examples)
    update_docs(examples)
    update_cloud_run_examples(examples)
    update_gke_examples(examples)
    update_vertex_ai_examples(examples)
    print(
        "README.md, docs/source/resources.mdx, and example README files have been updated."
    )
